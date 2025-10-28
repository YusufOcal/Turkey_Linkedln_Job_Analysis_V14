import matplotlib.pyplot as plt
import pandas as pd

# Define leakage columns and rationale
leakage_info = [
    ("apply_rate", "Direct target metric (apply_rate > Q3 label)"),
    ("pop_views_log", "Contains future popularity (views) info"),
    ("pop_applies_log", "Contains future popularity (applies) info"),
]

cols, reasons = zip(*leakage_info)

fig, ax = plt.subplots(figsize=(8, 2))
ax.axis('off')

table_data = [[c, r] for c, r in leakage_info]

table = ax.table(cellText=table_data, colLabels=["Column", "Why Leakage?"], loc='center')

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('black')
    cell.set_fontsize(10)
    if row == 0:
        cell.set_facecolor('#CCCCCC')

fig.tight_layout()
plt.savefig('leakage_columns.png', dpi=150)
print('Saved leakage_columns.png') 