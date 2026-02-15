# Assignment 2: Step-by-Step Guide
## Neural Networks and Deep Learning

**Total Points:** 50 (+ 5 extra credit)  
**Due Date:** [Check Canvas]

---

## 📋 QUICK OVERVIEW

This assignment has 3 main parts:
1. **Part 1:** Build baseline models (10 points)
2. **Part 2:** Test activation functions & optimizers (20 points)
3. **Part 3:** Add skip connections (20 points)
4. **Extra Credit:** Implement weight decay (5 points)

---

## 🎯 RECOMMENDED DATASET: CIFAR-10

**Why CIFAR-10?**
- ✅ 10 classes (multi-class classification)
- ✅ 60,000 images (50k train, 10k test)
- ✅ Small 32×32 color images (manageable)
- ✅ Built into PyTorch (easy to load)
- ✅ Challenging for shallow networks (~35-40% accuracy)
- ✅ Well-documented and commonly used

**How to load:**
```python
from torchvision import datasets, transforms

# Normalization values for CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
```

---

## 📝 PROGRAMMING RESTRICTIONS

### ✅ ALLOWED:
- `torch.nn.Conv2d()` and `torch.nn.MaxPool2d()`
- `torch.nn.Conv1d()` and `torch.nn.MaxPool1d()`
- `torch.nn.RNNCell()` and `torch.nn.Embedding()`
- `torch.nn.Linear()` and `torch.nn.Bilinear()`
- `loss.backward()` for gradient computation
- Tensor operations (flatten, matmul, etc.)

### ❌ NOT ALLOWED:
- Pre-built activation functions (must implement forward pass yourself)
- PyTorch optimizers (`torch.optim.*`)
- Pre-built loss functions (implement from scratch)
- Pre-built normalization layers

### ⚠️ IMPORTANT:
- Implement ALL activation functions manually
- Implement ALL optimizer update rules manually
- Use `loss.backward()` but manually update parameters

---

## 🚀 PART 1: BASELINE MODELS [10 Points]

### STEP 1: Load and Preprocess CIFAR-10

**What to do:**
1. Download CIFAR-10
2. Normalize to mean=0, std=1 (or use standard transform)
3. Create train/test DataLoaders
4. Verify data shapes (should be [batch, 3, 32, 32])

**Code structure:**
```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms

# Load data with normalization
# Create DataLoaders
```

---

### STEP 2: Build 2-Layer Network (Verify Difficulty)

**Purpose:** Prove that CIFAR-10 needs deep learning

**Architecture:**
```
Input: 3072 (32×32×3 flattened)
  ↓
Linear(3072 → 128) + Sigmoid
  ↓
Linear(128 → 10) + Softmax
```

**Implementation details:**
- Use `torch.nn.Linear()` (allowed)
- Implement sigmoid activation manually: `1 / (1 + torch.exp(-x))`
- Implement softmax manually: `torch.exp(x) / torch.exp(x).sum()`
- Implement cross-entropy loss manually

**Training setup:**
- Optimizer: SGD (implement manually)
- Batch size: 1
- Learning rate: 0.01 or 0.001
- Epochs: 10-20

**Expected result:** 30-45% test accuracy

**Deliverable:** Report this architecture and accuracy in your report

---

### STEP 3: Build Baseline Deep Network (5+ Layers)

**Recommended Architecture (Simple but Effective):**

```
INPUT: [batch, 3, 32, 32]
  ↓
Conv2d(3→16, kernel=3, padding=1) + Sigmoid
  ↓ [batch, 16, 32, 32]
MaxPool2d(kernel=2)
  ↓ [batch, 16, 16, 16]
Conv2d(16→32, kernel=3, padding=1) + Sigmoid
  ↓ [batch, 32, 16, 16]
MaxPool2d(kernel=2)
  ↓ [batch, 32, 8, 8]
Conv2d(32→64, kernel=3, padding=1) + Sigmoid
  ↓ [batch, 64, 8, 8]
MaxPool2d(kernel=2)
  ↓ [batch, 64, 4, 4]
Flatten
  ↓ [batch, 1024]
Linear(1024 → 256) + Sigmoid
  ↓
Linear(256 → 10) + Softmax
```

**This gives you 5 parameterized layers!**

**Implementation checklist:**
- [ ] Create model class
- [ ] Implement sigmoid activation: `1 / (1 + torch.exp(-x))`
- [ ] Implement softmax activation: `torch.exp(x) / torch.exp(x).sum(dim=1, keepdim=True)`
- [ ] Implement cross-entropy loss manually
- [ ] Implement SGD update rule: `param = param - lr * param.grad`

**Training setup:**
- Optimizer: SGD (manual implementation)
- Batch size: 1
- Learning rate: 0.01 (adjust if needed)
- Epochs: 30-50

**Expected result:** 45-55% test accuracy

**Deliverable:** Report architecture, hyperparameters, and test accuracy

---

## 🔬 PART 2: ACTIVATION FUNCTIONS & OPTIMIZERS [20 Points]

### STEP 4: Test Different Activation Functions

**Pick 2 from this list (recommendations in bold):**

1. **Leaky ReLU** (RECOMMENDED - easiest)
   ```python
   def leaky_relu(x):
       return torch.maximum(x, 0.1 * x)
   ```

2. **Tanh** (RECOMMENDED - stable)
   ```python
   def tanh(x):
       return torch.tanh(x)  # Can use built-in for forward pass
   ```

3. SiLU (Swish)
   ```python
   def silu(x):
       return x / (1 + torch.exp(-x))
   ```

4. Gaussian
   ```python
   def gaussian(x):
       return torch.exp(-x**2)
   ```

**What to do:**
1. Copy your baseline architecture twice
2. Replace ALL sigmoid activations with Leaky ReLU in model 1
3. Replace ALL sigmoid activations with Tanh in model 2
4. Keep softmax for output layer
5. Train both with SGD (batch size = 1)
6. Use SAME number of epochs as baseline
7. Compare test accuracies

**Hypothesis to include in report:**
- "Leaky ReLU will perform better because it doesn't saturate for positive values, allowing gradients to flow better in deep networks"
- "Tanh will perform better than sigmoid because it's zero-centered"

**Deliverable:** Report both activation functions, hypothesis, and test accuracies

---

### STEP 5: Implement Mini-Batch SGD

**Key insight:** Instead of updating after every sample, accumulate gradients over multiple samples

**Implementation approach:**
```python
# Instead of computing loss for one sample:
for epoch in epochs:
    # OLD WAY (batch size = 1):
    for x, y in dataset:
        output = model(x)
        loss = compute_loss(output, y)
        loss.backward()
        update_parameters()  # Update after EVERY sample
        
    # NEW WAY (mini-batch):
    for batch_x, batch_y in dataloader:  # DataLoader handles batching
        output = model(batch_x)  # Forward pass entire batch
        loss = compute_loss(output, batch_y).mean()  # Average loss
        loss.backward()  # Gradients accumulated automatically
        update_parameters()  # Update after BATCH
```

**What to do:**
1. Take your BEST model from Step 4 (best activation function)
2. Modify training loop to use batches
3. Test these batch sizes: **[16, 32, 64, 128]**
4. Record test accuracy for each

**Tips:**
- For CIFAR-10, batch size 32 or 64 usually works best
- Larger batch = more stable but slower convergence
- Smaller batch = faster but noisier

**Deliverable:** Report mini-batch SGD implementation and which batch size worked best

---

### STEP 6: Add Momentum

**What is momentum?**
Momentum helps the optimizer "remember" previous gradient directions, smoothing out noisy updates.

**Update rule:**
```python
# Initialize
momentum_buffer = {param: torch.zeros_like(param) for param in parameters}
alpha = 0.9  # momentum coefficient

# During training:
for param in parameters:
    grad = param.grad
    momentum_buffer[param] = alpha * momentum_buffer[param] + grad
    param.data = param.data - lr * momentum_buffer[param]
    param.grad = None  # Reset gradient
```

**What to do:**
1. Take your best model from Step 5 (with best batch size)
2. Implement momentum optimizer
3. Test these alpha values: **[0.85, 0.9, 0.95]**
4. Record test accuracy for each

**Tips:**
- Alpha = 0.9 is usually optimal
- Higher alpha = more momentum (smoother but slower to adapt)

**Deliverable:** Report momentum implementation and best alpha value

---

## 🔗 PART 3: SKIP CONNECTIONS [20 Points]

### STEP 7: Extend Your Model (Add 10 Layers)

**Goal:** Add 10 layers that DON'T change tensor shape (needed for skip connections)

**Strategy for Conv2D models:**
Use `padding=1` with `kernel=3` to maintain spatial dimensions

**Example extended architecture:**
```
INPUT: [batch, 3, 32, 32]
  ↓
Conv2d(3→16, k=3, p=1) + Act      ← Layer 1
  ↓ [batch, 16, 32, 32]
Conv2d(16→16, k=3, p=1) + Act     ← Layer 2 (ADDED - same shape)
  ↓ [batch, 16, 32, 32]
Conv2d(16→16, k=3, p=1) + Act     ← Layer 3 (ADDED - same shape)
  ↓ [batch, 16, 32, 32]
MaxPool2d(2)
  ↓ [batch, 16, 16, 16]
Conv2d(16→32, k=3, p=1) + Act     ← Layer 4
  ↓ [batch, 32, 16, 16]
Conv2d(32→32, k=3, p=1) + Act     ← Layer 5 (ADDED - same shape)
  ↓ [batch, 32, 16, 16]
Conv2d(32→32, k=3, p=1) + Act     ← Layer 6 (ADDED - same shape)
  ↓ [batch, 32, 16, 16]
Conv2d(32→32, k=3, p=1) + Act     ← Layer 7 (ADDED - same shape)
  ↓ [batch, 32, 16, 16]
MaxPool2d(2)
  ↓ [batch, 32, 8, 8]
Conv2d(32→64, k=3, p=1) + Act     ← Layer 8
  ↓ [batch, 64, 8, 8]
Conv2d(64→64, k=3, p=1) + Act     ← Layer 9 (ADDED - same shape)
  ↓ [batch, 64, 8, 8]
Conv2d(64→64, k=3, p=1) + Act     ← Layer 10 (ADDED - same shape)
  ↓ [batch, 64, 8, 8]
Conv2d(64→64, k=3, p=1) + Act     ← Layer 11 (ADDED - same shape)
  ↓ [batch, 64, 8, 8]
Conv2d(64→64, k=3, p=1) + Act     ← Layer 12 (ADDED - same shape)
  ↓ [batch, 64, 8, 8]
MaxPool2d(2)
  ↓ [batch, 64, 4, 4]
Flatten → [batch, 1024]
Linear(1024→256) + Act             ← Layer 13
Linear(256→128) + Act              ← Layer 14 (ADDED)
Linear(128→10) + Softmax           ← Layer 15
```

**Total: 15 parameterized layers!**

**Training:**
- Use best optimizer from Part 2 (mini-batch SGD with momentum)
- Use best hyperparameters (batch size, alpha)
- Same number of epochs

**Important:** Record average gradient L1-norm for first epoch:
```python
total_grad_norm = 0
count = 0
for param in model.parameters():
    if param.grad is not None:
        total_grad_norm += param.grad.abs().sum().item()
        count += 1
avg_grad_norm = total_grad_norm / count
```

**Deliverable:** Report extended architecture, test accuracy, and gradient norms

---

### STEP 8: Add Skip Connections (2 Configurations)

**What is a skip connection?**
```python
# Without skip:
x = layer1(x)
x = activation(x)
x = layer2(x)
x = activation(x)

# With skip (adds input to output):
identity = x
x = layer1(x)
x = activation(x)
x = layer2(x)
x = x + identity  # Skip connection!
x = activation(x)
```

**Configuration 1: Short Skips (ResNet-style)**
Add 3 skip connections that skip 1 layer each:
```
x1 = conv1(x)
x1 = act(x1)

x2 = conv2(x1)
x2 = x2 + x1  # Skip 1 ← SKIP CONNECTION 1
x2 = act(x2)

x3 = conv3(x2)
x3 = x3 + x2  # Skip 1 ← SKIP CONNECTION 2
x3 = act(x3)

x4 = conv4(x3)
x4 = x4 + x3  # Skip 1 ← SKIP CONNECTION 3
x4 = act(x4)
```

**Configuration 2: Medium Skips**
Add 3 skip connections that skip 2 layers each:
```
x0 = conv1(x)
x0 = act(x0)

x1 = conv2(x0)
x1 = act(x1)

x2 = conv3(x1)
x2 = x2 + x0  # Skip 2 ← SKIP CONNECTION 1
x2 = act(x2)

x3 = conv4(x2)
x3 = act(x3)

x4 = conv5(x3)
x4 = x4 + x2  # Skip 2 ← SKIP CONNECTION 2
x4 = act(x4)

x5 = conv6(x4)
x5 = act(x5)

x6 = conv7(x5)
x6 = x6 + x4  # Skip 2 ← SKIP CONNECTION 3
x6 = act(x6)
```

**KEY REQUIREMENT:** Layers connected by skip must have SAME SHAPE!

**Training:**
- Same optimizer and hyperparameters as Step 7
- Record gradient norms for each configuration

**What to compare:**
1. Extended model WITHOUT skips (from Step 7)
2. Extended model WITH Config 1 skips
3. Extended model WITH Config 2 skips

**Deliverable:** 
- Report both configurations
- Test accuracies for all 3 models
- Gradient norms for all 3 models
- Analysis of which worked best and why

---

## 💎 EXTRA CREDIT: WEIGHT DECAY [5 Points]

**What is weight decay?**
Regularization technique that penalizes large weights by pushing them toward zero.

**Modified momentum update rule:**
```python
# Standard momentum:
momentum_buffer[param] = alpha * momentum_buffer[param] + grad
param.data = param.data - lr * momentum_buffer[param]

# With weight decay:
momentum_buffer[param] = alpha * momentum_buffer[param] + grad + beta * param.data
param.data = param.data - lr * momentum_buffer[param]
```

**What to do:**
1. Take your BEST model WITHOUT skip connections (from Part 2)
2. Take your BEST model WITH skip connections (from Part 3)
3. Add weight decay to training
4. Test beta values: **[0.0001, 0.0005, 0.001]**
5. Compare with and without weight decay

**Deliverable:**
- Report weight decay implementation
- Test accuracies with different beta values
- Analysis of weight decay's effect

---

## 📊 EVALUATION METRICS

For EVERY model, report:

### 1. Test Accuracy
```python
correct = 0
total = 0
with torch.no_grad():
    for x, y in test_loader:
        output = model(x)
        pred = output.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
accuracy = correct / total
```

### 2. Precision (average across classes)
```python
# For each class:
precision_c = true_positives_c / (true_positives_c + false_positives_c)
# Average:
avg_precision = sum(precision_c) / num_classes
```

### 3. Recall (average across classes)
```python
# For each class:
recall_c = true_positives_c / (true_positives_c + false_negatives_c)
# Average:
avg_recall = sum(recall_c) / num_classes
```

**Tip:** Use confusion matrix to compute precision/recall easily

---

## 📝 REPORT STRUCTURE

### Methods Section
**What to include:**
- Dataset choice and why (CIFAR-10)
- Preprocessing steps
- ALL architectural details:
  - Number of layers
  - Layer types (Conv2d, Linear)
  - Number of channels/neurons
  - Activation functions
  - Pooling locations
- ALL hyperparameters:
  - Learning rate
  - Batch size
  - Momentum (alpha)
  - Weight decay (beta) if applicable
  - Number of epochs
- Hypotheses for each experiment
- Could someone reproduce your work from this section?

### Results Section
**What to include:**
- Test accuracy for ALL models (tables or plots)
- Precision and recall for ALL models
- Gradient norms (Part 3)
- Well-formatted figures with:
  - Clear labels
  - Legends
  - Titles/captions
  - No ambiguity

**Suggested format:**
```
Table 1: Model Performance Comparison

Model                          | Test Acc | Precision | Recall
-------------------------------|----------|-----------|-------
2-Layer Baseline               | 38.2%    | 0.35      | 0.38
5-Layer Sigmoid Baseline       | 52.1%    | 0.50      | 0.51
5-Layer Leaky ReLU            | 61.3%    | 0.60      | 0.61
5-Layer Tanh                   | 58.7%    | 0.57      | 0.58
Best + Mini-batch (32)         | 65.4%    | 0.64      | 0.65
Best + Momentum (α=0.9)        | 68.2%    | 0.67      | 0.68
15-Layer Extended              | 64.5%    | 0.63      | 0.64
15-Layer + Skip Config 1       | 71.8%    | 0.70      | 0.71
15-Layer + Skip Config 2       | 69.3%    | 0.68      | 0.69
```

### Analysis Section
**What to discuss (minimum 2 trends):**

**Example Trend 1: Activation Functions**
- "Leaky ReLU outperformed both Sigmoid and Tanh by 9.2% and 2.6% respectively"
- "This is because Leaky ReLU doesn't saturate for positive values, maintaining gradient flow in deep networks"
- "Sigmoid suffers from vanishing gradients in deep architectures"

**Example Trend 2: Skip Connections**
- "Skip connections improved test accuracy by 7.3%"
- "Average gradient norm increased from 0.0023 to 0.0145 with skip connections"
- "This demonstrates skip connections help combat vanishing gradients"
- "Short skips (Config 1) outperformed medium skips (Config 2) because..."

**Other possible trends:**
- Effect of batch size on training stability
- Impact of momentum on convergence speed
- Relationship between network depth and performance
- Trade-offs between different techniques

---

## ✅ CHECKLIST BEFORE SUBMISSION

### Code Requirements:
- [ ] All activation functions implemented manually (forward pass)
- [ ] All optimizers implemented manually (no torch.optim)
- [ ] Used only permitted PyTorch modules
- [ ] Cross-entropy loss implemented manually
- [ ] Training loops implemented from scratch
- [ ] Gradient resetting implemented manually
- [ ] Parameter updates implemented manually

### Report Requirements:
- [ ] Methods section complete with all details
- [ ] Hypotheses provided for experiments
- [ ] Results section with accuracy, precision, recall
- [ ] Figures are well-formatted and clear
- [ ] Analysis discusses at least 2 trends in depth
- [ ] Report is 3-5 pages (can be longer)
- [ ] Follows methods/results/analysis structure

### Submission Format:
- [ ] Single PDF file
- [ ] Lab report first
- [ ] Code appendix (plain text, NOT screenshots)
- [ ] All deliverables addressed

---

## 🎯 QUICK TIPS FOR SUCCESS

1. **Start early** - Deep learning experiments take time to run
2. **Keep organized** - Use clear variable names and comments
3. **Track everything** - Save all results in a spreadsheet/notebook
4. **Test incrementally** - Make sure each part works before moving on
5. **Use reasonable epochs** - 20-50 epochs is usually sufficient
6. **Monitor training** - Print loss/accuracy during training to catch issues
7. **Save models** - Use torch.save() to save trained models
8. **Verify shapes** - Print tensor shapes frequently to catch bugs
9. **Start simple** - Get basic version working, then optimize
10. **Document as you go** - Don't wait until the end to write report

---

## 🐛 COMMON PITFALLS TO AVOID

1. **Forgetting to reset gradients** between batches
2. **Using torch.optim** (not allowed!)
3. **Not implementing activations manually** (required!)
4. **Skip connections between different shapes** (won't work!)
5. **Running too few epochs** (might not converge)
6. **Not normalizing data** (training will be unstable)
7. **Batch size too large** (might not fit in memory)
8. **Learning rate too high** (training diverges)
9. **Not recording metrics** (can't write report!)
10. **Submitting code screenshots** (use plain text!)

---

## 📈 EXPECTED PERFORMANCE RANGES (CIFAR-10)

- 2-Layer Network: 30-45%
- 5-Layer Sigmoid Baseline: 45-55%
- 5-Layer with ReLU: 55-65%
- With Mini-batch + Momentum: 65-75%
- Extended (15 layers) no skips: 60-70%
- Extended with skip connections: 70-80%

**Don't worry if you don't hit these exactly!** Focus on:
1. Showing clear improvement from baseline
2. Understanding WHY changes help/hurt
3. Proper implementation of all techniques

---

## 🔥 FINAL STRATEGY FOR FULL MARKS

1. **Choose CIFAR-10** (simple, well-documented)
2. **Use Leaky ReLU** (works well, easy to implement)
3. **Batch size 32-64** (good balance)
4. **Momentum α=0.9** (standard value)
5. **Short skip connections** (easier to implement, usually work well)
6. **Be thorough in report** (explain everything clearly)
7. **Analyze deeply** (don't just state results, explain WHY)

---

## 📚 CODE STRUCTURE TEMPLATE

```python
# main.py structure:
# 1. Imports
# 2. Data loading
# 3. Model definition
# 4. Activation functions
# 5. Loss function
# 6. Optimizer functions
# 7. Training loop
# 8. Evaluation functions
# 9. Main execution

# Keep everything organized and commented!
```

---

## 🚀 YOU GOT THIS!

Remember:
- This guide simplifies everything for you
- Follow steps in order
- Keep it simple
- Document everything
- Ask for help when stuck

**Good luck! You're going to do great! 🎓**
