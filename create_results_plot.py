import matplotlib.pyplot as plt

# Hard-coded results gathered from training logs
models = [
    ("LightGBM", 0.9994, 0.9984),
    ("LogReg",  0.9985, 0.9954),
    ("RandForest", 0.9875, 0.9664),
    ("MLP", 0.9847, 0.9606),
]

names = [m[0] for m in models]
roc = [m[1] for m in models]
pr  = [m[2] for m in models]

x = range(len(names))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x, roc, width, label='ROC-AUC', color='skyblue')
plt.bar([i + width for i in x], pr, width, label='PR-AUC', color='salmon')
plt.ylim(0.95, 1.0)
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks([i + width/2 for i in x], names)
plt.legend()
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
print('Saved plot to model_comparison.png') 