# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""
import os
import uuid
import json
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, RayPPOTrainer, _timer, apply_kl_penalty, compute_advantage, compute_response_mask
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto


class RayConstraintTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize constraint manager if enabled
        self.use_constraints = self.config.algorithm.get("use_constraints", False)
        print(f"\033[93mDAPOTrainer: use_constraints = {self.use_constraints}\033[0m")
        self.constraint_manager = None
        
        if self.use_constraints:
            from .constraint_reward_manager import CalashRewardManager
            
            constraint_config = self.config.algorithm.constraint_config
            self.constraint_manager = CalashRewardManager(
                target_brier_score=constraint_config.target_brier_score,
                lambda_init=constraint_config.lambda_init,
                lambda_lr=constraint_config.lambda_lr,
                lambda_max=constraint_config.lambda_max,
                lambda_min=constraint_config.lambda_min,
                constraint_type=constraint_config.constraint_type,
            )
            if self.constraint_manager is not None:
                print("\033[91mLoad ConstrainedRewardManager Successful\033[0m")

    def _validate(self):
        print("Validation: Generation Begin.")
        
        reward_acc_lst = []
        data_source_lst = []
        length_lst = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            # test_batch = test_batch.to('cuda')

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            n_val_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)
            test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            test_batch = test_batch.union(test_output_gen_batch)
            # evaluate using reward_function
            # for certain reward function (e.g. sandbox), the generation can overlap with reward
            reward_result = self.val_reward_fn(test_batch, return_dict=True)
            reward_acc = reward_result["reward_extra_info"]["acc"]

            # obtain response length
            def obtain_reponse_length(output_batch):
                prompt_length = output_batch.batch['prompts'].shape[-1]
                response_length = output_batch.batch['attention_mask'][:,prompt_length:].sum(1).numpy()
                return response_length
            
            length_lst.append(obtain_reponse_length(test_output_gen_batch))
            reward_acc_lst.append(reward_acc)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * len(reward_acc)))

        print('Validation: Generation end.')

        reward_acc = np.concatenate(reward_acc_lst, axis=0) # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        lengths = np.concatenate(length_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}
        data_source_response_lengths = {}
        for i in range(len(reward_acc)):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_acc[i])

            if data_source not in data_source_response_lengths:
                data_source_response_lengths[data_source] = []
            data_source_response_lengths[data_source].append(lengths[i])

        metric_dict = {}
        test_score_vals = []
        test_length_vals = []
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)
            test_score_vals.append(np.mean(rewards))

        for data_source, lengths in data_source_response_lengths.items():
            metric_dict[f'val/test_length/{data_source}'] = np.mean(lengths)
            test_length_vals.append(np.mean(lengths))

        metric_dict['result/avg_acc'] = np.mean(test_score_vals)
        metric_dict['result/avg_len'] = np.mean(test_length_vals)

        return metric_dict

    def _validate_with_save(self, output_path):
        """
        执行验证，并按 question_id 保存每个样本的多个生成响应及其准确率。
        指标计算将按 data_source 分组。
        支持 resume 功能：如果输出文件已存在，将跳过已处理的 question_ids。
        """
        print("Validation with Save: Generation Begin.")

        # 修改数据结构以存储 data_source
        # results_by_question 的结构将变为:
        # {
        #   "qid1": {"data_source": "source_A", "responses": [{"response": "...", "acc": 1.0, "tokens": 123}, ...]},
        #   "qid2": {"data_source": "source_B", "responses": [{"response": "...", "acc": 0.0, "tokens": 456}, ...]}
        # }
        results_by_question = {}
        processed_qids = set()  # 存储已处理的 question_ids

        # 检查输出文件是否存在，如果存在则读取已处理的 question_ids
        if os.path.exists(output_path):
            print(f"Found existing results file: {output_path}")
            print("Loading processed question_ids for resume...")
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        qid = data["question_id"]
                        processed_qids.add(qid)
                        # 将已有数据加载到内存中，用于后续的指标计算
                        results_by_question[qid] = {
                            "data_source": data["data_source"],
                            "responses": data["responses"]
                        }
                    except json.JSONDecodeError:
                        continue
            print(f"Found {len(processed_qids)} already processed question_ids.")

        # Calculate total samples for progress bar (excluding already processed ones)
        total_samples = 0
        skipped_samples = 0
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            # 检查这个 batch 中有多少需要处理
            if 'question_id' in test_batch.non_tensor_batch:
                batch_qids = test_batch.non_tensor_batch.get('question_id')
                unprocessed_in_batch = sum(1 for qid in batch_qids if qid not in processed_qids)
                total_samples += unprocessed_in_batch
                skipped_samples += len(batch_qids) - unprocessed_in_batch
            else:
                total_samples += len(test_batch)

        print(f"Total samples to process: {total_samples} (skipping {skipped_samples} already processed)")

        # Create progress bar with actual samples to process
        pbar = tqdm(total=total_samples, desc="Validating samples")

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                print("Skipping validation for model-based reward model.")
                continue

            # 检查 batch 中的 question_ids，过滤出需要处理的样本
            if 'question_id' in test_batch.non_tensor_batch:
                batch_qids = test_batch.non_tensor_batch.get('question_id')
                # 找出需要处理的样本索引
                indices_to_process = [i for i, qid in enumerate(batch_qids) if qid not in processed_qids]

                if not indices_to_process:
                    # 整个 batch 都已处理过，跳过
                    continue

                # 只保留需要处理的样本
                test_batch = test_batch[indices_to_process]

            n_val_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n
            repeated_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)

            gen_batch = repeated_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }

            gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_wg.world_size)
            output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(gen_batch_padded)
            output_gen_batch = unpad_dataproto(output_gen_batch_padded, pad_size=pad_size)

            final_batch = repeated_batch.union(output_gen_batch)

            reward_result = self.val_reward_fn(final_batch, return_dict=True)
            accuracies = reward_result["reward_extra_info"]["acc"]

            # --- 新增: 提取 question_id 和 data_source ---
            question_ids = final_batch.non_tensor_batch.get('question_id')
            # 假设每个样本都有 data_source，如果没有则提供默认值 'unknown'
            data_sources = final_batch.non_tensor_batch.get('data_source', ['unknown'] * len(question_ids))

            response_ids = output_gen_batch.batch['responses']

            # 计算实际的 token 长度（参考 obtain_reponse_length）
            prompt_length = output_gen_batch.batch['prompts'].shape[-1]
            response_lengths = output_gen_batch.batch['attention_mask'][:, prompt_length:].sum(1).numpy()

            new_results_for_save = {}  # 存储这个 batch 的新结果

            for i in range(len(question_ids)):
                qid = question_ids[i]
                source = data_sources[i]
                acc = accuracies[i]
                token_length = int(response_lengths[i])

                response_text = self.tokenizer.decode(response_ids[i], skip_special_tokens=True)

                # --- 修改: 更新数据保存结构 ---
                # 如果是第一次遇到这个 qid，则初始化其条目
                if qid not in results_by_question:
                    results_by_question[qid] = {
                        "data_source": source,
                        "responses": []
                    }
                    # 标记为需要保存的新结果
                    new_results_for_save[qid] = {
                        "data_source": source,
                        "responses": []
                    }

                # 将结果添加到字典中，包含 tokens 字段
                results_by_question[qid]["responses"].append({
                    "response": response_text.strip(),
                    "acc": float(acc),
                    "tokens": token_length
                })

                # 如果是新结果，也添加到待保存列表
                if qid in new_results_for_save:
                    new_results_for_save[qid]["responses"].append({
                        "response": response_text.strip(),
                        "acc": float(acc),
                        "tokens": token_length
                    })

            # 实时追加新结果到文件（避免程序中断丢失数据）
            if new_results_for_save:
                with open(output_path, 'a', encoding='utf-8') as f:
                    for qid, data in new_results_for_save.items():
                        json_line = json.dumps({
                            "question_id": qid,
                            "data_source": data["data_source"],
                            "responses": data["responses"]
                        }, ensure_ascii=False)
                        f.write(json_line + '\n')

            # Update progress bar
            pbar.update(len(test_batch))

        pbar.close()
        print('Validation with Save: Generation end.')

        # 不再需要在最后保存，因为已经实时保存了
        print(f"Validation results saved to {output_path} (JSONL format)")

        # --- 修改: 计算并返回指标以用于日志记录 (与 _validate 逻辑对齐) ---
        data_source_reward = defaultdict(list)
        data_source_response_lengths = defaultdict(list)

        # 从已保存的结果中收集数据
        for qid, data in results_by_question.items():
            source = data['data_source']
            for res in data['responses']:
                data_source_reward[source].append(res['acc'])
                # 使用实际的 token 长度而不是字符串长度
                data_source_response_lengths[source].append(res['tokens'])

        metric_dict = {}
        test_score_vals = []
        test_length_vals = []

        # 计算每个 data_source 的平均准确率
        for data_source, rewards in data_source_reward.items():
            mean_reward = np.mean(rewards)
            metric_dict[f'val/test_score/{data_source}'] = mean_reward
            test_score_vals.append(mean_reward)

        # 计算每个 data_source 的平均长度
        for data_source, lengths in data_source_response_lengths.items():
            mean_length = np.mean(lengths)
            metric_dict[f'val/test_length/{data_source}'] = mean_length
            test_length_vals.append(mean_length)

        # 计算总体平均准确率和长度
        if test_score_vals:
            metric_dict['result/avg_acc'] = np.mean(test_score_vals)
        if test_length_vals:
            metric_dict['result/avg_len'] = np.mean(test_length_vals)

        return metric_dict

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        if self.global_steps != 0 and self.use_constraints:
            self.constraint_manager.load_state(os.path.join(self.config.trainer.default_local_dir,f"{self.global_steps}_constraint_manager_state.pt"))

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            if self.config.trainer.get("val_only", False):
                print(f"Validation only, val_save_path: {self.config.trainer.val_save_path}")
                val_metrics = self._validate_with_save(self.config.trainer.val_save_path)
            else:
                val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # pop those keys for generation
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    with _timer("reward", timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result["reward_extra_info"]
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor
                        new_batch.batch["confidence_tensor"] = torch.tensor(reward_extra_infos_dict["confidence"])
                        new_batch.batch["acc_tensor"] = torch.tensor(reward_extra_infos_dict["acc"])

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # Compute response_mask before applying constraints
                        if "response_mask" not in new_batch.batch:
                            new_batch.batch["response_mask"] = compute_response_mask(new_batch)

                        #breakpoint()

                        # Apply Lagrangian constraints if enabled
                        if self.use_constraints and self.constraint_manager is not None:

                            new_batch.batch["original_token_level_scores"] = new_batch.batch["token_level_scores"].clone()

                            constraint_result = self.constraint_manager.compute_constrained_reward(
                                new_batch,
                                return_dict=True
                            )
                            
                            # Update rewards with constrained values
                            new_batch.batch["token_level_scores"] = constraint_result["reward_tensor"]
                            metrics.update({"calash/num_format_errors_batch": constraint_result["reward_extra_info"]["calash/num_format_errors_batch"]})
                            
                            # Update metrics with constraint info
                            constraint_metrics = self.constraint_manager.get_metrics()
                            print(f"\033[92mConstraint metrics: {list(constraint_metrics.keys())}\033[0m")
                            metrics.update(constraint_metrics)


                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = new_batch.batch["token_level_scores"].sum(dim=-1).numpy()

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name]):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [uid for uid, std in prompt_uid2metric_std.items() if std > 0 or len(prompt_uid2metric_vals[uid]) == 1]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                progress_bar.update(1)
                                continue
                            else:
                                raise ValueError(f"{num_gen_batches=} >= {max_num_gen_batches=}." + " Generated too many. Please check if your data are too difficult." + " You could also try set max_num_gen_batches=0 to enable endless trials.")
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    #breakpoint()
                    metrics.update({"critic/original_rewards/mean": batch.batch['original_token_level_scores'].mean().item()})

                    # === Updating ===
                    if "response_mask" not in batch.batch:
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                            if self.use_constraints:                            
                                self.constraint_manager.save_state(os.path.join(self.config.trainer.default_local_dir,f"{self.global_steps}_constraint_manager_state.pt"))

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
