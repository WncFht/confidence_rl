import torch
import numpy as np
import os
from typing import Dict, Optional, Union
from verl import DataProto


class DynamicConfidenceRewardManager:
    """
    Manages Lagrangian constraints for DAPO, adapted for Confidence Calibration (Dynamic Confidence RL).
    
    This class implements Dynamic Confidence RL.
    Its goal is to maximize task rewards while satisfying a constraint on the
    model's calibration error, specifically the Brier Score.
    
    Mathematical formulation (Brier Score-based):
    $$\mathcal{L}(\theta, \lambda) = \mathbb{E}_{\pi_\theta}[r_{task}(x,y)] - \lambda \cdot g(\pi_\theta)$$
    
    where the constraint $g(\pi_\theta)$ is the "average Brier Score violation":
    $$g(\pi_\theta) = \mathbb{E}_{\pi_\theta}[(c - y)^2] - L_{Brier}$$
    $c$ = model confidence, $y \in \{0, 1\}$ = task correctness, $L_{Brier}$ = target Brier score.
    
    The one-sided augmented reward for a sample $(x, y, c)$ becomes:
    $$\tilde{R}(x,y,c) = r_{task}(x,y) - \lambda \cdot \max(0, (c - y)^2 - L_{Brier})$$
    """
    
    def __init__(
        self,
        target_brier_score: float = 0.25,  # Target for the average Brier score (e.g., 0.1)
        lambda_init: float = 0.1,
        lambda_lr: float = 0.01,
        lambda_max: float = 5.0,  # Max penalty coefficient
        lambda_min: float = 0.0,
        constraint_type: str = "average",  # Only "average" Brier score is supported
    ):
        if constraint_type != "average":
            raise ValueError("Dynamic Confidence RL only supports 'average' constraint on Brier score.")
            
        self.target_brier_score = target_brier_score
        self.lambda_lr = lambda_lr
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min
        self.constraint_type = constraint_type
        
        # Lagrange multiplier
        self.lagrange_multiplier = lambda_init
        
        # Tracking statistics (keys here MUST match get_metrics)
        self.stats = {
            "avg_brier": [],
            "max_brier": [],
            "min_brier": [],
            "std_brier": [],
            "avg_confidence": [],
            "std_confidence": [],
            "avg_task_reward": [], # Tracks original task reward of valid samples
            "constraint_violation": [],
            "lagrange_multiplier": [],
            "num_violations": [], # Fraction of valid samples exceeding target Brier
            "penalty_magnitude": [], # Average penalty applied per valid sample
            "satisfaction_rate": [], # Fraction of valid samples <= target Brier
            "penalty_active_rate": [],
            "avg_active_penalty": [],
            "lambda_change_rate": [],
        }

    def compute_constrained_reward(
        self, 
        batch: DataProto,
        return_dict: bool = True,
        FORMAT_PENALTY: float = -2.0,
    ) -> Union[torch.Tensor, Dict[str, Union[torch.Tensor, Dict]]]:
        """
        Apply Lagrangian constraint (Dynamic Confidence RL) to rewards based on Brier Score.
        
        Formulas:
        - Brier Score: $B_i = (c_i - y_i)^2$ (where $y_i=1$ if $r_{task}>0$ else $0$)
        - Sample Violation: $v_i = B_i - L_{Brier}$
        - Penalty: $p_i = \lambda \cdot \max(0, v_i)$
        - Augmented reward: $\tilde{r}_T = r_T - p_i$ (applied at last token)
        
        - Batch Violation: $g = \mathbb{E}[B_i] - L_{Brier}$
        - Lambda update: $\lambda = \text{clip}(\lambda + \eta_\lambda \cdot g, \lambda_{min}, \lambda_{max})$
        """
        # 1. Get rewards and masks
        rewards = batch.batch["token_level_scores"]
        response_mask = batch.batch["response_mask"]
        format_tensor = batch.batch["format_tensor"]
        confidence_tensor = batch.batch["confidence_tensor"] # Shape: (batch_size,)

        # Clone to avoid in-place modification
        reward_tensor = rewards.clone()
        
        # 3. Loop through samples to compute Brier score and apply penalty
        valid_brier_scores = []
        valid_sample_violations = [] # Stores *positive* violations
        valid_original_task_rewards = []
        valid_confidences = []
        num_violations = 0
        total_penalty = 0.0
          
        for i in range(len(reward_tensor)):
            valid_response_length = int(torch.sum(response_mask[i]).item())
        
            # Skip samples with no response
            if valid_response_length == 0:
                continue
                
            last_token_idx = valid_response_length - 1
            r_task = rewards[i, last_token_idx].item()

            # Check if the format is correct (i.e., not a format penalty)
            if format_tensor[i]:
                # --- 3a. Format is correct: Apply Dynamic Confidence RL logic ---
                
                # Get task reward and confidence
                c_i = confidence_tensor[i].item()
                y_i = 1.0 if r_task > 0 else 0.0
                
                # Statistics for valid samples
                valid_original_task_rewards.append(r_task)
                valid_confidences.append(c_i)

                # Calculate Brier Score and Dynamic Confidence RL penalty
                brier_i = (c_i - y_i) ** 2
                violation_i = brier_i - self.target_brier_score
                
                valid_brier_scores.append(brier_i) # Add all valid brier scores

                if violation_i > 0:
                    num_violations += 1
                    valid_sample_violations.append(violation_i)
                    penalty = self.lagrange_multiplier * violation_i
                else:
                    penalty = 0.0
                
                # Apply penalty and clamp
                final_reward = r_task - penalty
                reward_tensor[i, last_token_idx] = torch.clamp(
                    torch.tensor(final_reward), min=-1.0, max=1.0
                )
                total_penalty += penalty

            else:
                # --- 3b. Format is incorrect ---
                # The reward is already FORMAT_PENALTY in the original `rewards`
                # and thus in `reward_tensor`. No action needed, but we
                # explicitly *don't* include it in Brier score stats.
                pass
                
        # --- 5. Calculate batch statistics (based *only* on valid samples) ---
        num_total_samples = len(reward_tensor)
        num_valid_samples = len(valid_brier_scores)

        # Average Brier (only on valid samples) - Used for Lambda update
        avg_valid_brier = np.mean(valid_brier_scores) if valid_brier_scores else 0.0
        
        # Average confidence and task rewards (only on valid samples) - Used for logging
        avg_valid_confidence = np.mean(valid_confidences) if valid_confidences else 0.0
        avg_valid_original_task_reward = np.mean(valid_original_task_rewards) if valid_original_task_rewards else 0.0
        
        # $g = \mathbb{E}[B_i] - L_{Brier}$
        current_constraint_violation = avg_valid_brier - self.target_brier_score
            
        # --- 6. Update Lambda ---
        # We update lambda even if num_valid_samples is 0.
        # In that case, avg_valid_brier is 0, and the violation will be
        # -self.target_brier_score, causing lambda to decrease (which is reasonable).
        lambda_change = self.lambda_lr * current_constraint_violation
        self.lagrange_multiplier += lambda_change
        self.lagrange_multiplier = np.clip(
            self.lagrange_multiplier, self.lambda_min, self.lambda_max
        )

        # --- 7. Update running statistics ---
        if num_total_samples > 0:
            # --- DEBUG FIX: Write to the correct stat keys ---
            self.stats["avg_brier"].append(avg_valid_brier) 
            
            # --- DEBUG FIX: Use 'valid_brier_scores' and handle empty list ---
            self.stats["std_brier"].append(np.std(valid_brier_scores) if valid_brier_scores else 0.0)
            self.stats["max_brier"].append(np.max(valid_brier_scores) if valid_brier_scores else 0.0)
            self.stats["min_brier"].append(np.min(valid_brier_scores) if valid_brier_scores else 0.0)
            
            # --- DEBUG FIX: Write to the correct stat keys ---
            self.stats["avg_confidence"].append(avg_valid_confidence)
            self.stats["std_confidence"].append(np.std(valid_confidences) if valid_confidences else 0.0)
            self.stats["avg_task_reward"].append(avg_valid_original_task_reward)
            
            self.stats["constraint_violation"].append(current_constraint_violation)
            self.stats["lagrange_multiplier"].append(self.lagrange_multiplier)
            
            # Statistics based on valid samples
            if num_valid_samples > 0:
                self.stats["num_violations"].append(num_violations / num_valid_samples)
                self.stats["penalty_magnitude"].append(total_penalty / num_valid_samples)
                
                # --- DEBUG FIX: 'satisfaction_rate' logic ---
                # Check how many valid brier scores were at or below the target
                within_target = sum(1 for b in valid_brier_scores if b <= self.target_brier_score)
                self.stats["satisfaction_rate"].append(within_target / num_valid_samples)
                
                # 'valid_sample_violations' only contains violations > 0
                non_zero_penalties = valid_sample_violations
                if non_zero_penalties:
                    self.stats["penalty_active_rate"].append(len(non_zero_penalties) / num_valid_samples)
                    self.stats["avg_active_penalty"].append(
                        np.mean([self.lagrange_multiplier * v for v in non_zero_penalties])
                    )
                else:
                    self.stats["penalty_active_rate"].append(0.0)
                    self.stats["avg_active_penalty"].append(0.0)
            else: # If entire batch had format errors or was empty
                self.stats["num_violations"].append(0.0)
                self.stats["penalty_magnitude"].append(0.0)
                self.stats["satisfaction_rate"].append(0.0)
                self.stats["penalty_active_rate"].append(0.0)
                self.stats["avg_active_penalty"].append(0.0)

            # Lambda change rate
            self.stats["lambda_change_rate"].append(lambda_change)
        
        # --- 8. Return ---
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {
                    # Log batch-specific, valid-only stats
                    "dynamic_confidence_rl/avg_brier_valid_batch": avg_valid_brier,
                    "dynamic_confidence_rl/avg_confidence_valid_batch": avg_valid_confidence,
                    "dynamic_confidence_rl/lambda": self.lagrange_multiplier,
                    "dynamic_confidence_rl/num_format_errors_batch": num_total_samples - num_valid_samples,
                }
            }
        else:
            return reward_tensor
            
    def get_metrics(self) -> Dict[str, float]:
        """Get current constraint optimization metrics for logging."""
        metrics = {}
        
        # Use the last recorded values from stats
        if self.stats["avg_brier"]:
            metrics["dynamic_confidence_rl/avg_brier"] = self.stats["avg_brier"][-1]
            metrics["dynamic_confidence_rl/std_brier"] = self.stats["std_brier"][-1]
            metrics["dynamic_confidence_rl/min_brier"] = self.stats["min_brier"][-1]
            metrics["dynamic_confidence_rl/max_brier"] = self.stats["max_brier"][-1]
            metrics["dynamic_confidence_rl/avg_confidence"] = self.stats["avg_confidence"][-1]
            metrics["dynamic_confidence_rl/avg_task_reward"] = self.stats["avg_task_reward"][-1]

            # Constraint satisfaction metrics
            metrics["dynamic_confidence_rl/satisfaction_rate"] = self.stats["satisfaction_rate"][-1]
            metrics["dynamic_confidence_rl/num_violations_rate"] = self.stats["num_violations"][-1]
            
            # Distance from target
            metrics["dynamic_confidence_rl/avg_violation"] = self.stats["constraint_violation"][-1]
            
        # Penalty statistics
        if self.stats["penalty_magnitude"]:
            metrics["dynamic_confidence_rl/avg_penalty"] = self.stats["penalty_magnitude"][-1]
            metrics["dynamic_confidence_rl/penalty_active_rate"] = self.stats["penalty_active_rate"][-1]
            
            # --- DEBUG FIX: Move this check inside and add its own list check ---
            if self.stats["avg_active_penalty"]:
                metrics["dynamic_confidence_rl/avg_active_penalty"] = self.stats["avg_active_penalty"][-1]
                    
        # Lambda tracking
        if self.stats["lagrange_multiplier"]:
            metrics["dynamic_confidence_rl/lambda"] = self.stats["lagrange_multiplier"][-1]
            if self.stats["lambda_change_rate"]:
                metrics["dynamic_confidence_rl/lambda_change_rate"] = self.stats["lambda_change_rate"][-1]
                
        # Add current target
        metrics["dynamic_confidence_rl/target_brier_score"] = self.target_brier_score

        return metrics
        
    def reset_stats(self):
        """Reset tracking statistics."""
        for key in self.stats:
            self.stats[key] = []
        
    def get_summary_string(self) -> str:
        """Get a human-readable summary of current metrics."""
        metrics = self.get_metrics()
        
        summary_parts = []
        
        if "dynamic_confidence_rl/avg_brier" in metrics:
            summary_parts.append(
                f"Avg Brier: {metrics['dynamic_confidence_rl/avg_brier']:.3f} "
                f"(target: {self.target_brier_score:.3f}, "
                f"sat_rate: {metrics.get('dynamic_confidence_rl/satisfaction_rate', 0):.2%})"
            )
        
        if "dynamic_confidence_rl/avg_confidence" in metrics:
            summary_parts.append(f"Avg Conf: {metrics['dynamic_confidence_rl/avg_confidence']:.3f}")
            
        if "dynamic_confidence_rl/lambda" in metrics:
            summary_parts.append(f"λ: {metrics['dynamic_confidence_rl/lambda']:.4f}")
            
        if "dynamic_confidence_rl/avg_penalty" in metrics:
            summary_parts.append(
                f"Penalty: {metrics['dynamic_confidence_rl/avg_penalty']:.3f} "
                f"(active: {metrics.get('dynamic_confidence_rl/penalty_active_rate', 0):.2%})"
            )
            
        if "dynamic_confidence_rl/avg_task_reward" in metrics:
            summary_parts.append(f"Task R: {metrics['dynamic_confidence_rl/avg_task_reward']:.3f}")

        return " | ".join(summary_parts)


    def save_state(self, file_path: str):
        """
        Saves the current state of the manager to a file for checkpointing.

        Args:
            file_path (str): The path to the file where the state will be saved.
        """
        # Ensure the directory exists
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        state_dict = {
            'lagrange_multiplier': self.lagrange_multiplier,
            'stats': self.stats,  # Save full stats for continuous logging
        }
        
        try:
            torch.save(state_dict, file_path)
            print(f"\033[92m✅ Dynamic Confidence RL Manager state saved to {file_path}\033[0m")
        except Exception as e:
            print(f"\033[91m❌ Error saving Dynamic Confidence RL Manager state: {e}\033[0m")

    def load_state(self, file_path: str):
        """
        Loads the state of the manager from a file to resume training.

        Args:
            file_path (str): The path to the file from which to load the state.
        """
        if not os.path.exists(file_path):
            print(f"\033[93m⚠️ [WARNING] Dynamic Confidence RL checkpoint file not found at {file_path}. Starting with a fresh state.\033[0m")
            # Do not exit, just start fresh
            return

        try:
            state_dict = torch.load(file_path, map_location='cpu')
            
            # Restore core state variables using .get() for safety
            self.lagrange_multiplier = state_dict.get('lagrange_multiplier', self.lagrange_multiplier)
            
            # Restore stats dictionary for complete historical data
            loaded_stats = state_dict.get('stats', {})
            for key in self.stats:
                if key in loaded_stats:
                    self.stats[key] = loaded_stats[key]

            print(f"\033[92m✅ Dynamic Confidence RL Manager state loaded from {file_path}\033[0m")
            print(f"  - Resumed λ: {self.lagrange_multiplier:.4f}")

        except Exception as e:
            print(f"\033[91m❌ Error loading Dynamic Confidence RL Manager state from {file_path}: {e}. \033[0m")
            # Don't raise error, just warn and continue with fresh state
            print(f"\033[93m⚠️ [WARNING] Proceeding with a fresh Dynamic Confidence RL state.\033[0m")

