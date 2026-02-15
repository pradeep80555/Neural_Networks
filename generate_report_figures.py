"""
Generate figures for Assignment 2 Report
Creates publication-quality matplotlib figures comparing various experiments
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

# Create figures directory
os.makedirs('report_figures', exist_ok=True)

# ============================================================================
# FIGURE 1: Architecture Comparison (2-layer vs 5-layer)
# ============================================================================
fig, ax = plt.subplots(figsize=(6, 4))

architectures = ['2-Layer\nNet', '5-Layer CNN\n(Sigmoid)', '5-Layer CNN\n(Leaky ReLU\n+ Tanh)']
accuracies = [48.04, 10.00, 65.24]
colors = ['#3498db', '#e74c3c', '#2ecc71']

bars = ax.bar(architectures, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{acc:.2f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

ax.set_ylabel('Test Accuracy (%)', fontweight='bold')
ax.set_title('Effect of Architecture and Activation Functions', fontweight='bold', pad=15)
ax.set_ylim(0, 75)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=10, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random baseline (10%)')
ax.legend()

plt.tight_layout()
plt.savefig('report_figures/fig1_architecture_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# FIGURE 2: Batch Size Effect
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Accuracy comparison
batch_sizes = ['Batch=1\n(SGD)', 'Batch=32\n(Mini-Batch)']
accuracies_batch = [65.24, 58.86]
times = [29.9, 2.8]

bars1 = ax1.bar(batch_sizes, accuracies_batch, color=['#9b59b6', '#e67e22'], 
                alpha=0.8, edgecolor='black', linewidth=1.2)
for bar, acc in zip(bars1, accuracies_batch):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{acc:.2f}%',
             ha='center', va='bottom', fontweight='bold')

ax1.set_ylabel('Test Accuracy (%)', fontweight='bold')
ax1.set_title('Batch Size: Accuracy Comparison', fontweight='bold')
ax1.set_ylim(0, 75)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Training time comparison
bars2 = ax2.bar(batch_sizes, times, color=['#9b59b6', '#e67e22'], 
                alpha=0.8, edgecolor='black', linewidth=1.2)
for bar, time in zip(bars2, times):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{time:.1f} min\n({times[0]/time:.1f}×)',
             ha='center', va='bottom', fontweight='bold', fontsize=9)

ax2.set_ylabel('Training Time (minutes)', fontweight='bold')
ax2.set_title('Batch Size: Training Time Comparison', fontweight='bold')
ax2.set_ylim(0, 35)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('report_figures/fig2_batch_size_effect.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# FIGURE 3: Momentum Hyperparameter Study
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 5))

momentum_values = [0.0, 0.5, 0.9, 0.95, 0.99]
accuracies_momentum = [58.86, 69.26, 74.02, 72.88, 71.40]

# Line plot with markers
line = ax.plot(momentum_values, accuracies_momentum, marker='o', markersize=8, 
               linewidth=2.5, color='#2ecc71', label='Test Accuracy')

# Highlight best value
best_idx = np.argmax(accuracies_momentum)
ax.plot(momentum_values[best_idx], accuracies_momentum[best_idx], 
        marker='*', markersize=20, color='#f39c12', 
        markeredgecolor='black', markeredgewidth=1.5,
        label=f'Best: α={momentum_values[best_idx]} ({accuracies_momentum[best_idx]:.2f}%)')

# Add value labels
for x, y in zip(momentum_values, accuracies_momentum):
    ax.text(x, y + 0.8, f'{y:.2f}%', ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Momentum Coefficient (α)', fontweight='bold')
ax.set_ylabel('Test Accuracy (%)', fontweight='bold')
ax.set_title('Effect of Momentum on 5-Layer CNN Performance', fontweight='bold', pad=15)
ax.set_xlim(-0.05, 1.04)
ax.set_ylim(55, 77)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='lower right')

# Add baseline reference
ax.axhline(y=58.86, color='gray', linestyle=':', linewidth=1.5, alpha=0.6, 
           label='No momentum baseline')

plt.tight_layout()
plt.savefig('report_figures/fig3_momentum_study.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# FIGURE 4: Skip Connections - Accuracy Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 5))

models = ['5-Layer\nBaseline', '15-Layer\nNo Skip', '15-Layer\nSkip Config 1\n(Short)', 
          '15-Layer\nSkip Config 2\n(Longer)']
accuracies_skip = [74.02, 10.00, 52.25, 73.66]
colors_skip = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']

bars = ax.bar(models, accuracies_skip, color=colors_skip, alpha=0.8, 
              edgecolor='black', linewidth=1.2)

# Add value labels
for bar, acc in zip(bars, accuracies_skip):
    height = bar.get_height()
    y_pos = height + 2 if height > 15 else height + 1
    ax.text(bar.get_x() + bar.get_width()/2., y_pos,
            f'{acc:.2f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

# Highlight degradation problem
ax.annotate('', xy=(1, 10), xytext=(0, 74), 
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax.text(0.5, 42, 'Degradation\n-64.02%', ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', linewidth=2),
        fontweight='bold', color='red', fontsize=9)

ax.set_ylabel('Test Accuracy (%)', fontweight='bold')
ax.set_title('Skip Connections Solve the Degradation Problem', fontweight='bold', pad=15)
ax.set_ylim(0, 82)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('report_figures/fig4_skip_connections_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# FIGURE 5: Gradient Flow Analysis
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 5))

models_grad = ['No Skip\nConnections', 'Skip Config 1\n(Short)', 'Skip Config 2\n(Longer)']
grad_norms = [10.13, 20.77, 689.55]
colors_grad = ['#e74c3c', '#f39c12', '#2ecc71']

bars = ax.bar(models_grad, grad_norms, color=colors_grad, alpha=0.8, 
              edgecolor='black', linewidth=1.2)

# Add value labels and ratios
ratios = [1.0, 2.05, 68.09]
for bar, norm, ratio in zip(bars, grad_norms, ratios):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 20,
            f'{norm:.2f}',
            ha='center', va='bottom', fontweight='bold', fontsize=10)
    if ratio > 1.0:
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{ratio:.1f}×',
                ha='center', va='center', fontweight='bold', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax.set_ylabel('Average Gradient L1-Norm (Epoch 1)', fontweight='bold')
ax.set_title('Skip Connections Dramatically Improve Gradient Flow', fontweight='bold', pad=15)
ax.set_ylim(0, 750)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Use log scale for better visualization
ax.set_yscale('log')
ax.set_ylim(5, 1000)
ax.set_ylabel('Average Gradient L1-Norm (Epoch 1, log scale)', fontweight='bold')

plt.tight_layout()
plt.savefig('report_figures/fig5_gradient_flow.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# FIGURE 6: Progressive Improvements Summary
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

improvements = [
    '2-Layer Baseline',
    '5-Layer CNN\n+ Sigmoid',
    '5-Layer CNN\n+ Leaky ReLU/Tanh',
    '+ Mini-batch\n(batch=32)',
    '+ Momentum\n(α=0.9)',
    '15-Layer\n+ Skip Config 2'
]
accuracies_prog = [48.04, 10.00, 65.24, 58.86, 74.02, 73.66]
colors_prog = ['#95a5a6', '#e74c3c', '#f39c12', '#9b59b6', '#2ecc71', '#3498db']

bars = ax.barh(improvements, accuracies_prog, color=colors_prog, alpha=0.8, 
               edgecolor='black', linewidth=1.2)

# Add value labels
for bar, acc in zip(bars, accuracies_prog):
    width = bar.get_width()
    x_pos = width + 1.5
    ax.text(x_pos, bar.get_y() + bar.get_height()/2.,
            f'{acc:.2f}%',
            ha='left', va='center', fontweight='bold', fontsize=10)

ax.set_xlabel('Test Accuracy (%)', fontweight='bold')
ax.set_title('Progressive Performance Improvements Throughout Assignment', fontweight='bold', pad=15)
ax.set_xlim(0, 82)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Add milestone markers
ax.axvline(x=74.02, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Best Performance')

plt.tight_layout()
plt.savefig('report_figures/fig6_progressive_improvements.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ All figures generated successfully!")
print("   Figures saved in: report_figures/")
print("   - fig1_architecture_comparison.png")
print("   - fig2_batch_size_effect.png")
print("   - fig3_momentum_study.png")
print("   - fig4_skip_connections_accuracy.png")
print("   - fig5_gradient_flow.png")
print("   - fig6_progressive_improvements.png")
