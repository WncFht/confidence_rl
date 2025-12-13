"""
RLCR recipe: Reward Learning with Calibration Regularization.
Reward = acc_reward - brier_score, where:
- acc_reward: whether the answer is correct (0 or 1)
- brier_score: (acc - confidence)^2
"""

from .rlcr_ray_trainer import RayRLCRTrainer

__all__ = ["RayRLCRTrainer"]

