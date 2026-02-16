# CSCI 4922/5922: Deep Learning - Lab Assignment 2

## Training Deep Convolutional Neural Networks: Optimization Techniques and Residual Learning

**Student:** [Your Name]  
**Date:** February 15, 2026  
**Course:** Neural Networks and Deep Learning  

---

## Abstract

This report investigates the training dynamics of convolutional neural networks on the Fashion-MNIST image classification task, focusing on activation functions, batch size, momentum optimization, and skip connections for deep architectures. We demonstrate that proper activation function selection (Leaky ReLU + Tanh) dramatically improves network training compared to sigmoid-based networks. Mini-batch gradient descent provides substantial speedup with acceptable accuracy trade-offs. Momentum optimization (α=0.95) yields the best 5-layer performance at 91.05% test accuracy. For deep 15-layer networks, skip connections provide meaningful improvements, with longer skip configurations (Config 2) achieving 90.35% test accuracy. Gradient flow analysis reveals that longer skip connections provide 65.6× stronger gradients compared to networks without skips, explaining their superior optimization behavior.

---

## 1. Methods

### 1.1 Dataset and Preprocessing

We used the Fashion-MNIST dataset containing 70,000 grayscale images (28×28 pixels, 1 channel) across 10 clothing categories: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot. The dataset was split into 60,000 training images and 10,000 test images following the standard partitioning. All images were normalized using PyTorch's standard transform pipeline with pixel values centered at 0.5 (mean=0.5, std=0.5). No validation set was used; hyperparameters were selected based on test set performance for pedagogical purposes only (not recommended for production systems).

### 1.2 Model Architectures

We implemented and evaluated six distinct architectures:

**2-Layer Baseline Network:**
- Architecture: Input (784) → FC1 (256) + Sigmoid → FC2 (10) + Softmax
- Parameters: 203,530
- Activation: Sigmoid for hidden layer
- Purpose: Establish baseline performance with shallow network

**5-Layer CNN (Sigmoid):**
- Block 1: Conv 1→16 (3×3, stride=1, pad=1) + Sigmoid + MaxPool (2×2)
- Block 2: Conv 16→32 (3×3, stride=1, pad=1) + Sigmoid + MaxPool (2×2)
- Block 3: Conv 32→64 (3×3, stride=1, pad=1) + Sigmoid + MaxPool (2×2)
- Fully Connected: Flatten → FC 576→256 + Sigmoid → FC 256→10
- Parameters: 169,610
- Purpose: Demonstrate vanishing gradient problem

**5-Layer CNN (Leaky ReLU + Tanh):**
- Block 1: Conv 1→16 (3×3, stride=1, pad=1) + Leaky ReLU (α=0.1) + MaxPool (2×2)
- Block 2: Conv 16→32 (3×3, stride=1, pad=1) + Leaky ReLU (α=0.1) + MaxPool (2×2)
- Block 3: Conv 32→64 (3×3, stride=1, pad=1) + Leaky ReLU (α=0.1) + MaxPool (2×2)
- Fully Connected: Flatten → FC 576→256 + Tanh → FC 256→10
- Parameters: 169,610
- Purpose: Mitigate vanishing gradients with improved activations

**15-Layer Deep CNN (No Skip):**
- Block 1: Conv 1→16, 3× Conv 16→16, each with Leaky ReLU, MaxPool → 14×14×16
- Block 2: Conv 16→32, 3× Conv 32→32, each with Leaky ReLU, MaxPool → 7×7×32
- Block 3: Conv 32→64, 3× Conv 64→64, each with Leaky ReLU, MaxPool → 3×3×64
- Fully Connected: Flatten → FC 576→256 + Tanh → FC 256→256 + Tanh → FC 256→10
- Total: 12 convolutional layers + 3 fully connected = 15 parameterized layers
- Parameters: 281,226
- Purpose: Demonstrate degradation problem in plain deep networks

**15-Layer CNN with Skip Connections (Config 1 - Short):**
- Same architecture as 15-Layer No Skip
- Skip connections: 3 residual connections, each spanning 1 layer
  - Skip 1: conv1 output → conv2 output (y = x + f(x))
  - Skip 2: conv5 output → conv6 output (y = x + f(x))
  - Skip 3: conv9 output → conv10 output (y = x + f(x))
- Parameters: 281,226 (identical to no-skip version)
- Purpose: Test frequent, short skip connections (ResNet-style)

**15-Layer CNN with Skip Connections (Config 2 - Longer):**
- Same architecture as 15-Layer No Skip
- Skip connections: 3 residual connections, lengths 2-3
  - Skip 1: conv1 output → conv3 output (length 2)
  - Skip 2: conv5 output → conv8 output (length 3)
  - Skip 3: conv9 output → conv12 output (length 3)
- Parameters: 281,226
- Purpose: Test sparser, longer skip connections

All convolutional layers used 3×3 kernels with stride=1 and padding=1. All max pooling operations used 2×2 windows with stride=2.

### 1.3 Training Configuration

**Loss Function:** Cross-entropy loss implemented manually:
```
L = -∑ yᵢ log(ŷᵢ) where ŷᵢ = softmax(zᵢ)
```

**Optimizer:** Stochastic Gradient Descent (SGD) with optional momentum:
```
vₜ = α·vₜ₋₁ + ∇L(θₜ)
θₜ₊₁ = θₜ - η·vₜ
```
where α is momentum coefficient, η is learning rate, and v is velocity.

**Hyperparameters (systematic variation):**
- Learning rate: 0.005 (fixed for all experiments)
- Batch size: {1, 32} (Part 2 comparison)
- Momentum: {0.0, 0.5, 0.9, 0.95, 0.99} (Part 2 study)
- Epochs: 10 (all experiments)
- Weight initialization: PyTorch default (Kaiming uniform for conv, uniform for FC)
- Random seed: Not fixed (stochastic variation accepted)
- Device: Apple Silicon MPS (GPU acceleration)

### 1.4 Experimental Design

We conducted three sets of experiments:

**Experiment 1: Architecture and Activation Functions**
- Fixed: batch_size=1, momentum=0.0, lr=0.005, epochs=10
- Varied: Model architecture (2-layer vs 5-layer) and activation functions (Sigmoid vs Leaky ReLU+Tanh)
- Measured: Test accuracy, training time
- Purpose: Quantify impact of depth and activation choice

**Experiment 2: Batch Size and Momentum**
- Fixed: 5-layer CNN with Leaky ReLU+Tanh, lr=0.005, epochs=10
- Varied: 
  - Batch size: 1 vs 32 (holding momentum=0.0)
  - Momentum: {0.0, 0.5, 0.9, 0.95, 0.99} (holding batch_size=32)
- Measured: Test accuracy, training time
- Purpose: Optimize training efficiency and convergence

**Experiment 3: Deep Networks and Skip Connections**
- Fixed: batch_size=32, momentum=0.9, lr=0.005, epochs=10
- Varied: Skip connection configuration (none, config 1, config 2)
- Measured: Test accuracy, training accuracy, average gradient L1-norm during epoch 1
- Purpose: Demonstrate degradation problem and solution via residual learning

**Gradient Measurement Protocol:**
For Experiment 3, we quantified gradient flow by computing the L1-norm of all parameter gradients during each batch of the first training epoch:
```
gradient_norm_batch = ∑ ∑ |∇Wᵢⱼ|
                     layers params

avg_gradient_norm = mean(gradient_norm_batch for all batches in epoch 1)
```

This metric quantifies the magnitude of gradient signals reaching early layers, with larger values indicating stronger gradient flow and better optimization conditions.

### 1.5 Evaluation

All models were evaluated on the held-out test set of 10,000 images. We report classification accuracy as the primary metric. For deep network experiments, we additionally report the generalization gap (training accuracy - test accuracy) and gradient norms to diagnose optimization behavior. Training time was measured in wall-clock minutes on Apple Silicon M-series processors with MPS acceleration.

---

## 2. Results

### 2.1 Architecture and Activation Function Effects

Table 1 presents test accuracy for different architectures and activation functions.

| Model | Activation | Test Accuracy | Training Time |
|-------|-----------|--------------|---------------|
| 2-Layer Net | Sigmoid | 81.37% | ~2 min |
| 5-Layer CNN | Sigmoid | 82.62% | 39.3 min |
| 5-Layer CNN | Leaky ReLU + Tanh | 90.42% | 40.9 min |

**Table 1:** Test accuracy for baseline architectures. Both 5-layer CNNs used batch_size=1 and trained for 10 epochs.

The 2-layer baseline with sigmoid activation achieved 81.37% accuracy on Fashion-MNIST. The 5-layer CNN with sigmoid activations achieved 82.62% accuracy after experiencing slow initial learning (stuck at ~10% for 3 epochs), demonstrating the vanishing gradient problem. Replacing sigmoid with Leaky ReLU (convolutional layers) and Tanh (fully connected layers) increased accuracy to 90.42%, a 7.80 percentage point improvement. Training times were similar (39.3 vs 40.9 minutes) for both 5-layer variants with batch_size=1, confirming that the performance difference arises from optimization dynamics rather than computational cost.

Figure 1 would visualize these results. The 5-layer sigmoid network shows delayed learning with eventual moderate success (82.62%), while Leaky ReLU + Tanh enables immediate and superior learning (90.42%). Unlike CIFAR-10 where sigmoid completely fails, Fashion-MNIST's simpler patterns allow sigmoid to eventually optimize, though modern activations remain substantially better.

**Note:** Figure needs regeneration with Fashion-MNIST data.

### 2.2 Batch Size Impact

Table 2 compares training efficiency for different batch sizes on the 5-layer CNN with Leaky ReLU + Tanh activations.

| Batch Size | Test Accuracy | Training Time | Speedup |
|-----------|--------------|---------------|---------|
| 1 (SGD) | 90.42% | 40.9 min | 1.0× |
| 32 (Mini-batch) | 85.57% | 2.8 min | 14.6× |

**Table 2:** Effect of batch size on test accuracy and training time. Both configurations used momentum=0.0 and trained for 10 epochs on 5-layer CNN with Leaky ReLU+Tanh.

Mini-batch SGD with batch_size=32 reduced training time from 40.9 minutes to 2.8 minutes, a 14.6× speedup. Test accuracy decreased from 90.42% to 85.57%, a 4.85 percentage point reduction. This represents a favorable trade-off between computational efficiency and model performance.

Figure 2 would illustrate this trade-off, showing both the accuracy reduction and substantial training time savings from mini-batch processing.

**Note:** Figure needs regeneration with Fashion-MNIST data.
| 0.0 | 85.57% | — |
| 0.5 | 88.88% | +3.31% |
| 0.7 | 89.30% | +3.73% |
| 0.9 | 90.88% | +5.31% |
| 0.95 | 91.15% | +5.58% |
| 0.99 | 11.71% | -73.86% |

**Table 3:** Effect of momentum coefficient on test accuracy. All models used batch_size=32, lr=0.005, and trained for 10 epochs.

Test accuracy increased monotonically from α=0.0 (85.57%) through α=0.95 (91.15%), achieving the best performance at α=0.95. However, α=0.99 experienced catastrophic failure, collapsing to 11.71% accuracy. This demonstrates the critical importance of momentum tuning: too low provides insufficient acceleration (α=0.5: 88.88%), while optimal values (α=0.95: 91.15%) provide maximum benefit, but excessive momentum (α=0.99: 11.71%) causes severe optimization instability and overshooting.

The optimal momentum coefficient α=0.95 improved test accuracy by 5.58 percentage points compared to no momentum. The dramatic failure at α=0.99 validates theoretical predictions about momentum instability at extreme values.

Figure 3 would illustrate the non-monotonic relationship between momentum and performance, with steady improvement from 0.0 to 0.95, then catastrophic collapse at 0.99.

**Note:** Figure needs regeneration with Fashion-MNIST data.

### 2.4 Deep Networks and the Degradation Problem

Table 4 presents results for 15-layer networks with and without skip connections.

| Model | Test Acc | Train Acc | Gradient Norm | Time |
|-------|---------|-----------|---------------|------|
| 5-Layer CNN (baseline) | 91.15% | — | — | 3.4 min |
| 15-Layer No Skip | 86.02% | 85.64% | 10.62 | 4.9 min |
| 15-Layer Skip Config 1 | 89.68% | 91.49% | 18.59 | 5.0 min |
| 15-Layer Skip Config 2 | 90.35% | 93.49% | 696.42 | 4.9 min |

**Table 4:** Performance of deep networks with varying skip connection strategies. All models used batch_size=32, momentum=0.9, and trained for 10 epochs. Gradient norm is averaged over epoch 1.

On Fashion-MNIST, the 15-layer network without skip connections achieved 86.02% test accuracy. While this is respectable performance, it falls 5.13 percentage points short of the 5-layer baseline (91.15%). Training accuracy (85.64%) was nearly identical to test accuracy, indicating the network successfully optimized but did not benefit from additional depth.

Skip connection configuration 1 (three short skips of length 1) achieved 89.68% test accuracy, a 3.66 percentage point improvement over the no-skip variant. This narrows the gap with the 5-layer baseline to just 1.47 points while using 3× the depth.

Skip connection configuration 2 (three longer skips of lengths 2-3) achieved 90.35% test accuracy, only 0.80 percentage points below the 5-layer baseline. This configuration nearly matched shallow network performance while being 3× deeper. Training accuracy (93.49%) exceeded test accuracy by 3.14 points, indicating mild overfitting.

Average gradient L1-norm during the first epoch increased from 10.62 (no skip) to 18.59 (config 1, 1.75× improvement) to 696.42 (config 2, 65.6× improvement), demonstrating dramatically improved gradient flow with longer skip connections.

Figure 4 visualizes the degradation mitigation from skip connections.

**Note:** Figures need to be regenerated with Fashion-MNIST data.

### 2.5 Progressive Performance Summary

The progressive improvements across configurations show:

- 2-layer baseline (sigmoid): 81.37%
- 5-layer sigmoid (batch_size=1): 82.62%
- 5-layer Leaky ReLU + Tanh (batch_size=1): 90.42%
- 5-layer Leaky ReLU + Tanh (batch_size=32, no momentum): 85.57%
- 5-layer with momentum α=0.5: 88.88%
- 5-layer with momentum α=0.7: 89.30%
- 5-layer with momentum α=0.9: 90.88%
- 5-layer with momentum α=0.95 (best): 91.15%
- 5-layer with momentum α=0.99 (collapsed): 11.71%
- 15-layer with skip Config 2: 90.35%

Fashion-MNIST proves substantially more learnable than CIFAR-10, with even sigmoid-based networks achieving >80% accuracy (vs ~10% on CIFAR-10). The best overall performance (91.15%) was achieved with the 5-layer CNN using Leaky ReLU+Tanh, mini-batch SGD, and momentum α=0.95. The catastrophic failure at α=0.99 (11.71%) demonstrates the critical importance of momentum tuning. The 15-layer network with longer skip connections (90.35%) nearly matched the shallow baseline, demonstrating the feasibility of training deeper architectures with appropriate gradient flow pathways.

**Note:** Figures showing progressive improvements need regeneration with actual Fashion-MNIST results.

---

## 3. Analysis

### 3.1 Activation Functions and Gradient Flow

The comparison between sigmoid (82.62%) and Leaky ReLU + Tanh (90.42%) on Fashion-MNIST reveals important insights about the vanishing gradient problem's dataset dependence. Unlike CIFAR-10 where sigmoid-based networks completely fail (~10% accuracy), Fashion-MNIST's simpler grayscale patterns allow sigmoid to eventually learn after a slow start. The sigmoid network remained stuck at ~10% accuracy for the first 3 epochs before suddenly improving, demonstrating the optimization difficulty caused by vanishing gradients. However, once the network escaped this poor initialization, it continued improving steadily.

Leaky ReLU maintains a constant gradient of 1.0 for positive inputs and a small negative slope (α=0.1) for negative inputs, preventing complete gradient death during backpropagation. This enabled immediate learning from epoch 1 (83% training accuracy) without the lengthy initialization phase. The Tanh function, while sigmoid-shaped, has a steeper maximum gradient (1.0 vs 0.25 for sigmoid) and zero-centered outputs.

The 7.80 percentage point improvement (82.62% → 90.42%) with identical training time (39-40 minutes) confirms that the performance difference arises from optimization quality rather than computational cost. The combination of Leaky ReLU and Tanh enables effective learning across all architectures tested, as evidenced by strong performance throughout our experiments.

### 3.2 Batch Size and Training Efficiency

Mini-batch SGD with batch_size=32 achieved a 14.6× training speedup (40.9 min → 2.8 min) at the cost of 4.85% accuracy reduction (90.42% → 85.57%). The efficiency gain from mini-batch processing comes from:

**Computational efficiency:** Processing 32 samples simultaneously leverages GPU parallelism through vectorized operations, dramatically reducing wall-clock time per epoch compared to sequential single-sample updates.

**Optimization dynamics:** Single-sample SGD provides noisy but unbiased gradient estimates with high variance, which can help escape local minima and explore the loss landscape more thoroughly. Mini-batch gradients are lower variance but may converge to worse local optima within a fixed epoch budget. The effective number of parameter updates is reduced from 60,000 per epoch (batch_size=1) to 1,875 per epoch (batch_size=32), meaning 32× fewer weight adjustment opportunities within 10 epochs.

The 4.85% accuracy reduction is acceptable given the order-of-magnitude speedup, particularly for early-stage experimentation where fast iteration times are valuable. Fashion-MNIST's higher signal-to-noise ratio compared to CIFAR-10 makes it more amenable to mini-batch optimization, explaining the strong 85.57% accuracy even without momentum.

### 3.3 Momentum Optimization Dynamics

The Fashion-MNIST results reveal a clear non-monotonic relationship between momentum and performance, with steady improvement from α=0.0 (85.57%) through α=0.95 (91.15%), followed by catastrophic failure at α=0.99 (11.71%).

**Low momentum (α=0.5):** Achieved 88.88% accuracy, a 3.31 percentage point improvement over no momentum. This demonstrates that even modest velocity accumulation helps acceleration, though not optimally.

**Intermediate momentum (α=0.7, 0.9):** Showed progressive improvement (89.30%, 90.88%), indicating that higher momentum values provide stronger acceleration benefits within the 10-epoch training window.

**Optimal momentum (α=0.95):** Achieved the best performance at 91.15%, a 5.58 percentage point gain over no momentum. This value provides strong velocity accumulation while maintaining sufficient damping to prevent oscillations. The near-optimal range of 0.9-0.95 aligns with established deep learning practice.

**Excessive momentum (α=0.99):** Experienced complete training collapse at 11.71% accuracy, equivalent to random guessing. This catastrophic failure occurs because the velocity term vₜ becomes dominated by historical gradients rather than current gradient information. With α=0.99, accumulated velocity from early training phases causes the optimizer to overshoot, oscillate wildly, and prevent any meaningful convergence. The network never escapes poor local configurations.

This dramatic demonstration of momentum instability (α=0.95: 91.15% → α=0.99: 11.71%, a 79.44 percentage point drop) validates theoretical warnings about excessive momentum. The sensitivity to this hyperparameter underscores the importance of careful tuning for optimization algorithms.

### 3.4 Deep Networks Performance on Fashion-MNIST

Unlike the severe degradation observed on CIFAR-10, the 15-layer network without skip connections achieved reasonable performance on Fashion-MNIST (86.02% test accuracy). However, it still underperformed the 5-layer baseline by 5.03 percentage points despite being 3× deeper.

The gradient norm analysis provides mechanistic insight: the no-skip network exhibited an average L1-norm of 10.62 during the first epoch. While sufficient for basic optimization (unlike the complete failure on CIFAR-10), these gradients were still suboptimal for efficiently training deep networks.

Fashion-MNIST's simpler patterns (grayscale clothing items vs complex color objects) and higher inherent separability likely explain why plain deep networks can optimize at all. However, the fact that additional depth fails to improve performance indicates persistent gradient flow limitations even on easier datasets.

### 3.5 Skip Connections Improve Deep Network Optimization

Skip connections improve gradient flow in deep networks by providing highways that bypass multiple layers of nonlinear transformations. In residual learning, each block learns F(x) while the skip connection passes x unchanged:

y = F(x) + x

During backpropagation:
∂L/∂x = ∂L/∂y · (∂F/∂x + I)

The identity term (I) ensures gradients flow directly backward through the skip connection regardless of F's derivatives, strengthening gradient signals to early layers.

**Configuration 1 vs Configuration 2:** The performance difference between short skips (config 1: 89.68%) and longer skips (config 2: 90.35%) demonstrates optimal skip connection strategy:

Config 1 used three length-1 skips, bypassing single convolutional layers. This improved gradient norms by 1.75× (18.59 vs 10.62) and increased test accuracy by 3.66 percentage points over the no-skip baseline. The improvement is meaningful but modest.

Config 2 used three skips of length 2-3, bypassing multiple layers. This increased gradient norms 65.6× (696.42 vs 10.62) and achieved 90.35% test accuracy, nearly matching the 5-layer baseline (0.70% gap). Longer skips create more direct pathways from loss to early layers, dramatically improving gradient magnitude and enabling stable optimization.

The substantial gradient flow improvement (**65.6×**) directly correlates with better optimization, supporting the hypothesis that skip connections work by combating vanishing gradients rather than merely providing additional capacity.

**Comparison with CIFAR-10:** On the harder CIFAR-10 dataset, skip connections were essential for any learning in deep networks (10% → 73.66%). On Fashion-MNIST, they provide incremental improvements (86.02% → 90.35%), suggesting their importance scales with task difficulty and optimization landscape complexity.

### 3.6 Limitations and Future Work

Several limitations warrant discussion:

**Limited training budget:** All experiments used only 10 epochs, which may be insufficient for deeper networks to converge fully, particularly for skip connection configurations.

**No validation set:** Hyperparameters were selected based on test set performance rather than a held-out validation set. This violates standard machine learning practice and may overestimate generalization to truly unseen data.

**Single dataset:** Results are specific to Fashion-MNIST's 28×28 grayscale clothing images. Fashion-MNIST is considerably easier than CIFAR-10, as evidenced by sigmoid achieving 82.62% (vs ~10% on CIFAR-10). Generalization to other modalities (RGB images, text, audio), resolutions, or task complexities remains untested. The reduced severity of optimization challenges on Fashion-MNIST may underestimate the importance of architectural innovations like skip connections.

**Absent regularization:** No batch normalization, dropout, weight decay, or data augmentation was used. These techniques are standard in modern deep learning and would likely improve performance.

**Single random seed:** All experiments used stochastic initialization without fixing random seeds or running multiple trials. Performance metrics represent single runs and do not account for optimization variance.

Future work should address these limitations by using proper validation splits, regularization techniques, and multiple-trial statistics. Additionally, investigating deeper architectures (50+ layers) on more challenging datasets would provide clearer evidence of skip connection benefits. The dramatic difference between Fashion-MNIST and CIFAR-10 results suggests that architectural insights should be validated across multiple difficulty levels.

---

## 4. Conclusions

This work systematically investigated training dynamics of convolutional neural networks on Fashion-MNIST, with the following findings:

**1. Activation functions significantly affect optimization, with dataset-dependent severity.** Sigmoid activations demonstrated slow learning (stuck at ~10% for 3 epochs) before achieving 82.62% accuracy on Fashion-MNIST. Leaky ReLU + Tanh enabled immediate and superior learning at 90.42%, a 7.80 percentage point improvement with identical training time. Unlike CIFAR-10 where sigmoid completely fails, Fashion-MNIST's simpler patterns allow sigmoid to eventually optimize, though modern activations remain substantially better.

**2. Mini-batch SGD enables highly efficient training.** Batch_size=32 achieved 14.6× speedup (40.9 min → 2.8 min) with acceptable 4.85% accuracy trade-off (90.42% → 85.57%), demonstrating the importance of hardware-aware optimization.

**3. Momentum substantially improves convergence with critical tuning requirements.** Optimal momentum (α=0.95) improved test accuracy from 85.57% to 91.15%, a 5.58 percentage point gain. However, excessive momentum (α=0.99) caused catastrophic training collapse to 11.71% accuracy, a 79.44 percentage point drop from optimal. This dramatic failure validates theoretical warnings about momentum instability and demonstrates that careful hyperparameter tuning is essential.

**4. Skip connections improve deep network optimization on easier datasets.** Plain 15-layer networks achieved modest 86.02% accuracy, underperforming the 5-layer baseline (91.15%) by 5.13 points despite 3× depth. Skip connections with longer spans (Config 2) nearly closed this gap at 90.35% test accuracy. Gradient flow analysis revealed that longer skips provide **65.6× stronger gradients** (696.42 vs 10.62), mechanistically explaining their superior optimization behavior.

**5. Task difficulty profoundly influences architectural requirements and optimization challenges.** Unlike CIFAR-10 where plain deep networks fail catastrophically and skip connections are essential, Fashion-MNIST allows plain deep networks to optimize reasonably (86.02%). Similarly, sigmoid fails completely on CIFAR-10 (~10%) but achieves 82.62% on Fashion-MNIST. This suggests architectural and algorithmic innovations provide greater benefit on harder tasks with more complex optimization landscapes.

Our best model (5-layer CNN with Leaky ReLU+Tanh, batch_size=32, momentum α=0.95) achieved **91.15% test accuracy** on Fashion-MNIST within a 3.4-minute training budget, representing an excellent balance between performance, efficiency, and architecture complexity for this task. The comprehensive experimental validation, including the dramatic momentum collapse at α=0.99, provides strong empirical evidence for established optimization principles in deep learning.

---

## References

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770-778.

Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *Proceedings of the International Conference on Artificial Intelligence and Statistics (AISTATS)*, 249-256.

Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013). On the importance of initialization and momentum in deep learning. *International Conference on Machine Learning (ICML)*, 1139-1147.

---

**End of Report**
