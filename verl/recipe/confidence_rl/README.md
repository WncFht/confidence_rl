# Constraint-based DAPO (Direct Alignment from Preference Optimization)

This module implements a constrained optimization approach for DAPO using Lagrangian multipliers to control output length in language model training.

## Mathematical Formulations

### 1. Lagrangian Formulation

The main objective function with constraints:

$$\mathcal{L}(\theta, \lambda) = -\mathbb{E}_{\pi_\theta}[R(x,y)] + \lambda \cdot g(\pi_\theta)$$

where:
- $\theta$ represents the model parameters
- $\lambda$ is the Lagrange multiplier
- $R(x,y)$ is the reward function
- $g(\pi_\theta)$ is the constraint function

### 2. Normalized Constraint Function

The constraint is defined as a ratio-based violation:

$$g(\pi_\theta) = \frac{\mathbb{E}_{\pi_\theta}[|y|]}{L_{target}} - 1$$

where:
- $|y|$ is the response length
- $L_{target}$ is the target length
- The constraint violation is normalized as a ratio

### 3. Augmented Reward

The reward is augmented with the constraint penalty:

$$\tilde{R}(x,y) = R(x,y) - \lambda \cdot \left(\frac{|y|}{L_{target}} - 1\right)$$

At the token level, the penalty is applied at the last token:

$$\tilde{r}_t = r_t - \lambda \cdot \mathbb{1}[t = T] \cdot \left(\frac{T}{L_{target}} - 1\right)$$

where:
- $\mathbb{1}[t = T]$ is an indicator function that equals 1 when $t = T$ (last token)
- $T$ is the total response length

### 4. Constraint Violation Types

#### Average Constraint
$$g = \frac{\mathbb{E}[|y|]}{L_{target}} - 1$$

#### Max Constraint
$$g = \frac{\max(|y|)}{L_{target}} - 1$$

### 5. Exponential Moving Average (EMA) Update

The constraint violation is smoothed using EMA:

$$\tilde{g}^{(t+1)} = \alpha \cdot \tilde{g}^{(t)} + (1-\alpha) \cdot g^{(t)}$$

where:
- $\alpha$ is the EMA smoothing parameter (typically 0.95)
- $\tilde{g}^{(t)}$ is the smoothed violation at time $t$

### 6. Momentum Update for Lambda

The Lagrange multiplier is updated with momentum:

$$m^{(t+1)} = \beta \cdot m^{(t)} + (1-\beta) \cdot \tilde{g}^{(t+1)}$$

where:
- $\beta$ is the momentum parameter (typically 0.9)
- $m^{(t)}$ is the momentum at time $t$

### 7. Lambda Update Rule

The Lagrange multiplier is updated based on constraint satisfaction:

$$\lambda^{(t+1)} = \text{clip}(\lambda^{(t)} + \eta_\lambda \cdot m^{(t+1)}, \lambda_{min}, \lambda_{max})$$

The update is triggered when:
- $\tilde{g} > \tau$: Increase $\lambda$ (constraint violated, length ratio exceeds tolerance)
- $\tilde{g} < -\tau$: Increase $\lambda$ (too far below target)

where:
- $\eta_\lambda$ is the learning rate for lambda updates
- $\tau$ is the tolerance threshold (as a ratio, e.g., 0.125 means 12.5% deviation)
- $\lambda_{min}, \lambda_{max}$ are the bounds for the multiplier

### 8. Adaptive Tolerance (Optional)

The tolerance can be adjusted based on performance:

$$\tau = \max\left(\tau_{min}, 2 \cdot \frac{\sigma_{lengths}}{L_{target}}\right)$$

where:
- $\tau_{min}$ is the minimum tolerance ratio
- $\sigma_{lengths}$ is the standard deviation of response lengths

### 9. Constraint Satisfaction Metrics

#### Satisfaction Rate
Percentage of samples within tolerance:

$$\text{Satisfaction Rate} = \frac{|\{y : |\frac{|y|}{L_{target}} - 1| \leq \tau\}|}{N}$$

#### Average Relative Distance
Mean deviation from target:

$$\text{Avg Relative Distance} = \frac{1}{N}\sum_{i=1}^{N} \left|\frac{|y_i|}{L_{target}} - 1\right|$$

#### Penalty Active Rate
Fraction of samples with non-zero penalties:

$$\text{Penalty Active Rate} = \frac{|\{i : |\lambda \cdot v_i| > \epsilon\}|}{N}$$

where $v_i$ is the constraint violation for sample $i$.

## Implementation Details

### Key Components

1. **ConstraintRewardManager**: Manages the Lagrangian constraint optimization
   - Computes constrained rewards
   - Updates Lagrange multipliers
   - Tracks optimization metrics

2. **RayConstraintTrainer**: Extends the base PPO trainer with constraint support
   - Integrates constraint manager into training loop
   - Applies constraints during reward computation
   - Saves/loads constraint manager state

### Hyperparameters

- `target_length`: Target response length (default: 4096)
- `tolerance`: Allowed deviation ratio (default: 0.125 = 12.5%)
- `lambda_init`: Initial Lagrange multiplier (default: 0.01)
- `lambda_lr`: Learning rate for lambda updates (default: 0.02)
- `lambda_max`: Maximum lambda value (default: 2.0)
- `lambda_min`: Minimum lambda value (default: 0.0)
- `ema_alpha`: EMA smoothing factor (default: 0.95)
- `momentum_beta`: Momentum factor (default: 0.9)

### Window Size Calculation

The rolling window for metrics is calculated as:

$$\text{window\_size} = 2 \times \text{batch\_size} \times \text{n\_responses\_per\_prompt}$$

This ensures metrics are computed over approximately 2 batches of data.

## Usage

The constraint optimization is enabled by setting `use_constraints: true` in the configuration and providing the constraint configuration parameters. The system will automatically apply Lagrangian constraints to control response length during training.