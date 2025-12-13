# Dynamic Confidence RL (Direct Alignment from Preference Optimization)

This module implements a constrained optimization approach for DAPO using Lagrangian multipliers to control confidence calibration (Brier Score) in language model training.

## Mathematical Formulations

### 1. Lagrangian Formulation

The main objective function with constraints:

$$\mathcal{L}(\theta, \lambda) = \mathbb{E}_{\pi_\theta}[r_{task}(x,y)] - \lambda \cdot g(\pi_\theta)$$

where:
- $\theta$ represents the model parameters
- $\lambda$ is the Lagrange multiplier
- $r_{task}(x,y)$ is the task reward function
- $g(\pi_\theta)$ is the constraint function (Brier Score violation)

### 2. Brier Score Constraint

The constraint is defined based on Brier Score:

$$g(\pi_\theta) = \mathbb{E}_{\pi_\theta}[(c - y)^2] - L_{Brier}$$

where:
- $c$ is the model confidence (predicted probability)
- $y \in \{0, 1\}$ is the task correctness (1 if $r_{task} > 0$, else 0)
- $L_{Brier}$ is the target Brier score
- The constraint violation measures calibration error

### 3. Augmented Reward

The reward is augmented with the constraint penalty:

$$\tilde{R}(x,y,c) = r_{task}(x,y) - \lambda \cdot \max(0, (c - y)^2 - L_{Brier})$$

At the token level, the penalty is applied at the last token:

$$\tilde{r}_T = r_T - \lambda \cdot \max(0, (c - y)^2 - L_{Brier})$$

where:
- $r_T$ is the task reward at the last token
- The penalty is only applied when the Brier score exceeds the target

### 4. Constraint Violation

#### Average Constraint
$$g = \mathbb{E}[(c - y)^2] - L_{Brier}$$

The constraint violation is computed as the average Brier score across valid samples minus the target.

### 5. Lambda Update Rule

The Lagrange multiplier is updated based on constraint satisfaction:

$$\lambda^{(t+1)} = \text{clip}(\lambda^{(t)} + \eta_\lambda \cdot g^{(t)}, \lambda_{min}, \lambda_{max})$$

where:
- $\eta_\lambda$ is the learning rate for lambda updates
- $g^{(t)}$ is the current constraint violation
- $\lambda_{min}, \lambda_{max}$ are the bounds for the multiplier

### 6. Constraint Satisfaction Metrics

#### Satisfaction Rate
Percentage of samples with Brier score within target:

$$\text{Satisfaction Rate} = \frac{|\{i : (c_i - y_i)^2 \leq L_{Brier}\}|}{N}$$

#### Average Brier Score
Mean Brier score across valid samples:

$$\text{Avg Brier Score} = \frac{1}{N}\sum_{i=1}^{N} (c_i - y_i)^2$$

#### Penalty Active Rate
Fraction of samples with non-zero penalties:

$$\text{Penalty Active Rate} = \frac{|\{i : (c_i - y_i)^2 > L_{Brier}\}|}{N}$$

where $v_i = (c_i - y_i)^2 - L_{Brier}$ is the constraint violation for sample $i$.

## Implementation Details

### Key Components

1. **DynamicConfidenceRewardManager**: Manages the Lagrangian constraint optimization for confidence calibration
   - Computes constrained rewards based on Brier Score
   - Updates Lagrange multipliers
   - Tracks optimization metrics (Brier score, confidence, violations, etc.)

2. **RayDynamicConfidenceTrainer**: Extends the base PPO trainer with dynamic confidence constraint support
   - Integrates constraint manager into training loop
   - Applies constraints during reward computation
   - Saves/loads constraint manager state for checkpointing

### Hyperparameters

- `target_brier_score`: Target Brier score (default: 0.5)
- `lambda_init`: Initial Lagrange multiplier (default: 0.01)
- `lambda_lr`: Learning rate for lambda updates (default: 0.02)
- `lambda_max`: Maximum lambda value (default: 2.0)
- `lambda_min`: Minimum lambda value (default: 0.0)
- `constraint_type`: Type of constraint (default: "average")

## Usage

The constraint optimization is enabled by setting `use_constraints: true` in the configuration and providing the constraint configuration parameters. The system will automatically apply Lagrangian constraints to control confidence calibration (Brier Score) during training.