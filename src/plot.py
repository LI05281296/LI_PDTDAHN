import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# 读取数据
data = pd.read_excel('plot_data\\triple_neg0.xlsx')
# 绘制ROC曲线
plt.figure(figsize=(10, 8))

for model, data in data.groupby('Model'):
    fpr, tpr = roc_curve(data['Recall'], data['AUC'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model} (AUC = {roc_auc:.2f})')

plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

# 绘制PR曲线
plt.figure(figsize=(10, 8))

for model, data in data.groupby('Model'):
    precision, recall = precision_recall_curve(data['Recall'], data['Precision'])
    avg_precision = average_precision_score(data['Recall'], data['Precision'])
    plt.plot(recall, precision, label=f'{model} (Avg Precision = {avg_precision:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()