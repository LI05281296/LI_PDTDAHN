import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, f1_score, roc_curve, auc, \
    precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

def evaluate_combined_relationships(positive_negative_samples, drug_disease_res, drug_target_res, disease_target_res):
    results = []
    for index, row in positive_negative_samples.iterrows():
        drug_index = row['drug']
        target_index = row['target']
        disease_index = row['disease']
        label = row['label']

        score_drug_disease = drug_disease_res.iloc[drug_index, disease_index]
        score_drug_target = drug_target_res.iloc[drug_index, target_index]
        score_disease_target = disease_target_res.iloc[disease_index, target_index]

        average_score = np.mean([score_drug_disease, score_drug_target, score_disease_target])
        results.append([score_drug_disease, score_drug_target, score_disease_target, label, average_score])
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'../../data/predict_label/label_{k}.csv', header=False, index=False)
    return pd.DataFrame(results, columns=['score_drug_disease', 'score_drug_target', 'score_disease_target', 'label', 'average_score'])

def plot_roc_pr_curves(combined_results, k):
    mean_fpr = np.linspace(0, 1, 1000)
    tprs = []
    true_labels = []
    average_scores = []
    y = []
    for index, row in combined_results.iterrows():
        score_drug_disease = row['score_drug_disease']
        score_drug_target = row['score_drug_target']
        score_disease_target = row['score_disease_target']
        true_label = row['label']
        average_score = row['average_score']
        # 三个模型的结果都为正时，认为有关系
        if score_drug_disease > 0.55 and score_drug_target > 0.55 and score_disease_target > 0.55:
            prediction = 1
        else:
            prediction = 0
        y.append(prediction)
        true_labels.append(true_label)
        average_scores.append(average_score)
    y = np.array(y)
    true_labels = np.array(true_labels)
    average_scores = np.array(average_scores)
    # 评估模型性能
    accuracy = accuracy_score(true_labels, y)

    precision = precision_score(true_labels, y)

    recall = recall_score(true_labels, y)

    f1 = f1_score(true_labels, y)

    mcc = matthews_corrcoef(true_labels, y)

    fpr, tpr, _ = roc_curve(true_labels, average_scores)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    # 保存ROC曲线数据neg7
    roc_curve_data = np.column_stack((fpr, tpr, _))
    np.savetxt(f'../../data/predict_label/neg7/fold_{k}_roc.txt', roc_curve_data, delimiter=',', fmt='%.8f')

    # 保存PR曲线数据
    precisions_, recalls_, _ = precision_recall_curve(true_labels, average_scores)
    # AUPR
    aupr = auc(recalls_, precisions_)
    pr_curve_data = np.column_stack((precisions_, recalls_))
    np.savetxt(f'../../data/predict_label/neg7/fold_{k}_pr.txt', pr_curve_data, delimiter=',', fmt='%.8f')

    result_dict = {
        "Fold": k,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "MCC": mcc,
        "F1 Score": f1,
        "AUC": roc_auc,
        "AUPR": aupr
    }
    return result_dict

# 得分矩阵标准化
def normalize_matrix(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    normalized_matrix = (matrix - min_val) / (max_val - min_val)
    return normalized_matrix

if __name__ == "__main__":
    # Load positive and negative samples
    positive_samples = pd.read_csv('../../data/data_ddt/pos.csv', names=['drug', 'disease', 'target', 'label'], header=None)
    '''
    negative_samples0 = pd.read_csv('../../data/data_ddt/neg0.csv', names=['drug', 'disease', 'target', 'label'], header=None)
    negative_samples1 = pd.read_csv('../../data/data_ddt/neg1.csv', names=['drug', 'disease', 'target', 'label'],
                                   header=None)
    negative_samples2 = pd.read_csv('../../data/data_ddt/neg2.csv', names=['drug', 'disease', 'target', 'label'],
                                   header=None)
    negative_samples3 = pd.read_csv('../../data/data_ddt/neg3.csv', names=['drug', 'disease', 'target', 'label'],
                                   header=None)
    negative_samples4 = pd.read_csv('../../data/data_ddt/neg4.csv', names=['drug', 'disease', 'target', 'label'],
                                   header=None)
    negative_samples5 = pd.read_csv('../../data/data_ddt/neg5.csv', names=['drug', 'disease', 'target', 'label'],
                                   header=None)
    negative_samples6 = pd.read_csv('../../data/data_ddt/neg6.csv', names=['drug', 'disease', 'target', 'label'],
                                   header=None)
    negative_samples = pd.concat([negative_samples0,negative_samples1,negative_samples2,negative_samples3,negative_samples4,negative_samples5,negative_samples6])

    negative_samples = negative_samples.sample(n=len(positive_samples), replace=False, random_state=42)
    '''
    negative_samples = pd.read_csv('../../data/data_ddt/neg6.csv', names=['drug', 'disease', 'target', 'label'], header=None)
    positive_negative_samples = pd.concat([positive_samples, negative_samples])
    positive_negative_samples['drug'] -= 1
    positive_negative_samples['disease'] -= 125
    positive_negative_samples['target'] -= 302
    AllResult = []
    # metric_dict = {"Fold": '', "Accuracy": 'Accuracy', "Precision": 'Precision', "Recall": 'Recall', "MCC": 'MCC',
    #               "F1 Score": 'F1 Score', "AUC": 'AUC'}
    # AllResult.append(metric_dict)
    for k in range(5):
        drug_disease_res = pd.read_csv(f'../../data/data_ddt/drug_disease_predict_{k}.csv', header=None)
        drug_target_res = pd.read_csv(f'../../data/data_ddt/drug_target_predict_{k}.csv', header=None)
        disease_target_res = pd.read_csv(f'../../data/data_ddt/disease_target_predict_{k}.csv', header=None)
        # Evaluate combined relationships
        combined_results = evaluate_combined_relationships(positive_negative_samples, drug_disease_res, drug_target_res,
                                                           disease_target_res)
        metrics = plot_roc_pr_curves(combined_results, k)
        AllResult.append(metrics)
    # Save results to CSV
    result_df = pd.DataFrame(AllResult)
    # 计算每一列的平均值
    mean_values = result_df.mean(numeric_only=True)
    # 创建一个包含平均值的新行
    new_row = {}
    new_row.update(mean_values.to_dict())
    new_row["Fold"] = "Mean"
    AllResult.append(new_row)
    # 将新行添加到 DataFrame
    result_df = pd.DataFrame(AllResult)
    result_df.to_csv('results6.csv', index=False)
