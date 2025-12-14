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
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

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

from verl.trainer.ppo import core_algos

def compute_auroc(y_true, y_score):
    """
    Compute Area under ROC curve (AUROC).
    
    Args:
        y_true: True binary labels (0 or 1)
        y_score: Predicted probabilities
        
    Returns:
        AUROC score, or 0.0 if computation fails (e.g., only one class present)
    """
    try:
        # Check if we have both classes
        if len(np.unique(y_true)) < 2:
            return 0.0
        return roc_auc_score(y_true, y_score)
    except Exception:
        return 0.0

def compute_ece(y_true, y_score, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).
    
    ECE = sum(|Bm|/N * |acc(Bm) - conf(Bm)|)
    where M is the number of bins, Bm is the set of samples in bin m, and N is the number of samples.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_score: Predicted probabilities
        n_bins: Number of bins (default: 10)
        
    Returns:
        ECE score
    """
    if len(y_true) == 0:
        return 0.0
    
    # Bin the predictions
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    n = len(y_true)
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        # For the first bin, include the lower boundary (0)
        if bin_lower == 0:
            in_bin = (y_score >= bin_lower) & (y_score <= bin_upper)
        else:
            in_bin = (y_score > bin_lower) & (y_score <= bin_upper)
        prop = np.mean(in_bin)
        
        if prop > 0:
            # Accuracy in this bin
            accuracy_in_bin = np.mean(y_true[in_bin])
            # Average confidence in this bin
            avg_confidence_in_bin = np.mean(y_score[in_bin])
            # Add to ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop
    
    return ece

def compute_confidence_advantage(data: DataProto, adv_estimator, norm_adv_by_std_in_grpo=True, group_by_acc=False):
    if adv_estimator == "GRPO":
        # TODO: test on more adv estimator type
        grpo_calculation_mask = data.batch["response_mask"]
        
        # Determine grouping index: by acc values or by uid (prompt)
        if group_by_acc:
            # Group by acc values: convert acc_tensor to numpy array of strings for grouping
            acc_values = data.batch["acc_tensor"].cpu().numpy()
            # Convert to string array for use as index (ensures hashable keys)
            grouping_index = np.array([str(acc) for acc in acc_values], dtype=object)
        else:
            # Original behavior: group by uid (prompt identifier)
            grouping_index = data.non_tensor_batch["uid"]
        
        # Compute reward: when format=0, use the value of token_level_scores; otherwise use 1 - brier_score
        # format_tensor is (batch_size,), brier_score is (batch_size,)
        # We need to expand format_tensor to token level for token_level_rewards
        format_tensor = data.batch["format_tensor"]  # (batch_size,)
        brier_score = data.batch["brier_score"]  # (batch_size,)
        
        # Compute base reward: 1 - brier_score
        base_reward = 1 - brier_score  # (batch_size,)
        
        # When format=0, set reward to the value of token_level_scores; otherwise use base_reward
        sequence_reward = torch.where(format_tensor == 0, data.batch["token_level_scores"].sum(dim=-1), base_reward)
        
        # Expand to token level: (batch_size,) -> (batch_size, 1) for unsqueeze
        token_level_rewards = sequence_reward.unsqueeze(-1)  # (batch_size, 1)
        
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=grpo_calculation_mask,
            index=grouping_index,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError

    return data

class RaySplitAdvConfidenceTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config.actor_rollout_ref.actor.ppo_mini_batch_size *= 2
        self.group_confidence_adv_by_acc = self.config.algorithm.get("group_confidence_adv_by_acc", False)

    def _validate(self):
        print("Validation: Generation Begin.")
        
        reward_acc_lst = []
        data_source_lst = []
        length_lst = []
        confidence_lst = []
        brier_score_lst = [] 
        format_lst = [] # 0/1, 0: incorrect, 1: correct
        answer_tokens_length_lst = []
        confidence_tokens_length_lst = []
        
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
            reward_acc = reward_result["reward_extra_info"]["acc"]  # True label y, assumed to be 0 or 1
            confidence = reward_result["reward_extra_info"]["confidence"] # Predicted probability p
            format_scores_batch = reward_result["reward_extra_info"]["format"] # Format, 0 or 1
        
            brier_scores = (np.array(confidence) - np.array(reward_acc))**2 
            
            # obtain response length
            def obtain_response_length(output_batch):
                prompt_length = output_batch.batch['prompts'].shape[-1]
                response_length = output_batch.batch['attention_mask'][:,prompt_length:].sum(1).numpy()
                return response_length
                
            length_lst.append(obtain_response_length(test_output_gen_batch))
            
            # compute answer_tokens_length and confidence_tokens_length
            if "response_mask" not in test_output_gen_batch.batch:
                test_output_gen_batch.batch["response_mask"] = compute_response_mask(test_output_gen_batch)
            acc_response_mask, confidence_response_mask = self.split_mask_by_answer_tag(
                test_output_gen_batch.batch, self.tokenizer
            )
            answer_tokens_length_lst.append(acc_response_mask.sum(dim=-1).float().mean().item())
            confidence_tokens_length_lst.append(confidence_response_mask.sum(dim=-1).float().mean().item())
            
            reward_acc_lst.append(reward_acc)
            confidence_lst.append(confidence)
            brier_score_lst.append(brier_scores)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * len(reward_acc)))
            format_lst.append(format_scores_batch)
        
        print('Validation: Generation end.')
        
        reward_acc = np.concatenate(reward_acc_lst, axis=0) # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        lengths = np.concatenate(length_lst, axis=0)
        confidences = np.concatenate(confidence_lst, axis=0)

        brier_scores = np.concatenate(brier_score_lst, axis=0) 
        format_scores = np.concatenate(format_lst, axis=0) # 0 or 1
        
        # evaluate test_score based on data source
        data_source_reward = {}
        data_source_response_lengths = {}
        data_source_confidence = {}
        data_source_brier_scores = {}
        data_source_format_scores = {}
        
        for i in range(len(reward_acc)):
            data_source = data_sources[i]
            
            # Accuracy
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_acc[i])
    
            # Length
            if data_source not in data_source_response_lengths:
                data_source_response_lengths[data_source] = []
            data_source_response_lengths[data_source].append(lengths[i])
    
            if data_source not in data_source_format_scores:
                data_source_format_scores[data_source] = []
            data_source_format_scores[data_source].append(format_scores[i])
    
            if data_source not in data_source_confidence:
                data_source_confidence[data_source] = []
            data_source_confidence[data_source].append(confidences[i])
            
            if data_source not in data_source_brier_scores:
                data_source_brier_scores[data_source] = []
            data_source_brier_scores[data_source].append(brier_scores[i])
        
        
        metric_dict = {}
        test_score_vals = []
        test_length_vals = []
        test_confidence_vals = []
        test_brier_score_vals  = [] 
        test_format_acc_vals = []
        test_auroc_vals = []
        test_ece_vals = []
        
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)
            test_score_vals.append(np.mean(rewards))
    
        for data_source, formats in data_source_format_scores.items():
            format_acc = np.mean(formats)
            metric_dict[f'val/format_acc/{data_source}'] = format_acc
            test_format_acc_vals.append(format_acc)
    
        for data_source, confidence in data_source_confidence.items():
            mean_confidence = np.mean(confidence)
            metric_dict[f'val/test_confidence/{data_source}'] = mean_confidence
            test_confidence_vals.append(mean_confidence)
        
        for data_source, brier_terms in data_source_brier_scores.items():
            mean_brier = np.mean(brier_terms)
            metric_dict[f'val/test_brier_score/{data_source}'] = mean_brier
            test_brier_score_vals.append(mean_brier)
    
        for data_source, lengths in data_source_response_lengths.items():
            metric_dict[f'val/test_length/{data_source}'] = np.mean(lengths)
            test_length_vals.append(np.mean(lengths))
        
        # Compute AUROC and ECE for each data source
        for data_source in data_source_reward.keys():
            # Get corresponding rewards (acc) and confidences for this data source
            acc_values = np.array(data_source_reward[data_source])
            conf_values = np.array(data_source_confidence[data_source])
            
            # Compute AUROC
            auroc = compute_auroc(acc_values, conf_values)
            metric_dict[f'val/confidence_metrics/auroc/{data_source}'] = auroc
            test_auroc_vals.append(auroc)
            
            # Compute ECE
            ece = compute_ece(acc_values, conf_values, n_bins=10)
            metric_dict[f'val/confidence_metrics/ece/{data_source}'] = ece
            test_ece_vals.append(ece)
       
        metric_dict['result/avg_acc'] = np.mean(test_score_vals)
        metric_dict['result/avg_len'] = np.mean(test_length_vals)
        
        metric_dict['result/avg_format_acc'] = np.mean(test_format_acc_vals)
    
        metric_dict['result/avg_confidence'] = np.mean(test_confidence_vals)
        metric_dict['result/avg_brier_score'] = np.mean(test_brier_score_vals)
        metric_dict['result/avg_auroc'] = np.mean(test_auroc_vals) if len(test_auroc_vals) > 0 else 0.0
        metric_dict['result/avg_ece'] = np.mean(test_ece_vals) if len(test_ece_vals) > 0 else 0.0
        
        # Compute average answer_tokens_length and confidence_tokens_length
        metric_dict['result/answer_tokens_length'] = np.mean(answer_tokens_length_lst) if len(answer_tokens_length_lst) > 0 else 0.0
        metric_dict['result/confidence_tokens_length'] = np.mean(confidence_tokens_length_lst) if len(confidence_tokens_length_lst) > 0 else 0.0
          
        return metric_dict

    def split_mask_by_answer_tag(self, batch, tokenizer):
        responses = batch['responses']
        original_mask = batch['response_mask']
        device = responses.device
        batch_size, seq_len = responses.size()

        answer_tokens = tokenizer.encode("</answer>", add_special_tokens=False)
        answer_tokens_tensor = torch.tensor(answer_tokens, device=device)
        tag_len = len(answer_tokens)

        windows = responses.unfold(1, tag_len, 1)
        matches = (windows == answer_tokens_tensor).all(dim=-1)
        
        found_mask = matches.any(dim=1)

        positions = torch.arange(matches.size(1), device=matches.device).unsqueeze(0).expand_as(matches)
        last_match_index = torch.where(matches, positions, -1).max(dim=1).values
        
        start_indices = torch.where(
            found_mask, 
            last_match_index, 
            torch.tensor(seq_len, device=device, dtype=torch.long)
        )

        end_indices = torch.clamp(start_indices + tag_len, max=seq_len)
        seq_range = torch.arange(seq_len, device=device).unsqueeze(0)

        acc_pos_mask = seq_range < end_indices.unsqueeze(1)
        acc_response_mask = original_mask & acc_pos_mask.type_as(original_mask)

        conf_pos_mask = seq_range >= end_indices.unsqueeze(1)
        conf_response_mask = original_mask & conf_pos_mask.type_as(original_mask)

        return acc_response_mask, conf_response_mask

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
                        reward_extra_infos_dict: dict[str, list] = {}
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
                        new_batch.batch["format_tensor"] = torch.tensor(reward_extra_infos_dict["format"])
                        new_batch.batch["acc_tensor"] = torch.tensor(reward_extra_infos_dict["acc"])

                        # compute brier score
                        new_batch.batch["brier_score"] =  1.0 * (new_batch.batch["acc_tensor"] - new_batch.batch["confidence_tensor"]) ** 2

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # Compute response_mask
                        if "response_mask" not in new_batch.batch:
                            new_batch.batch["response_mask"] = compute_response_mask(new_batch)

                        #breakpoint()

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

                    metrics.update({"format/valid_num": batch.batch['format_tensor'].sum().item()})
                    metrics.update({"confidence_metrics/brier_score/mean": batch.batch['brier_score'].mean().item()})
                    metrics.update({"confidence_metrics/brier_score/max": batch.batch['brier_score'].max().item()})
                    metrics.update({"confidence_metrics/brier_score/min": batch.batch['brier_score'].min().item()})
                    
                    # Compute AUROC and ECE for training batch
                    acc_tensor_np = batch.batch['acc_tensor'].cpu().numpy()
                    confidence_tensor_np = batch.batch['confidence_tensor'].cpu().numpy()
                    
                    train_auroc = compute_auroc(acc_tensor_np, confidence_tensor_np)
                    train_ece = compute_ece(acc_tensor_np, confidence_tensor_np, n_bins=10)
                    
                    metrics.update({"confidence_metrics/auroc": train_auroc})
                    metrics.update({"confidence_metrics/ece": train_ece})
                        
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

                    # split mask by answer tag
                    acc_response_mask, confidence_response_mask = self.split_mask_by_answer_tag(batch.batch, self.tokenizer)

                    # compute answer_tokens and confidence_tokens length
                    answer_tokens_length = acc_response_mask.sum(dim=-1).float().mean().item()
                    confidence_tokens_length = confidence_response_mask.sum(dim=-1).float().mean().item()
                    
                    # Add answer_tokens and confidence_tokens length statistics to metrics
                    metrics.update({"tokens_length/answer_tokens/mean": answer_tokens_length})
                    metrics.update({"tokens_length/confidence_tokens/mean": confidence_tokens_length})

                    # clone and set response mask
                    # FIX: acc_batch also needs deepcopy, otherwise modifying its mask will affect the original batch
                    acc_batch = deepcopy(batch)
                    confidence_batch = deepcopy(batch)
                    acc_batch.batch["response_mask"] = acc_response_mask
                    confidence_batch.batch["response_mask"] = confidence_response_mask

                    with _timer("adv", timing_raw):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        group_confidence_adv_by_acc = self.config.algorithm.get("group_confidence_adv_by_acc", False)
                        
                        acc_batch = compute_advantage(
                            acc_batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                        confidence_batch = compute_confidence_advantage(
                            confidence_batch,
                            adv_estimator="GRPO",
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            group_by_acc=group_confidence_adv_by_acc,
                        )

                        # TODO: concat acc and confidence batch in a better way
                        # Currently, we concat them directly, which might cause some problems
                        # 交错concat，或者根据acc和confidence的顺序，交替concat
                        
                        batch = DataProto.concat([acc_batch, confidence_batch])
                        n = len(acc_batch)
                        indices = []
                        for i in range(n):
                            indices.append(i)      # from data1
                            indices.append(i + n)  # from data2
                        batch = batch.select_idxs(indices)

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
