import math
import random
import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, f1_score, roc_curve, auc, \
    precision_recall_curve
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('../mashup_data/pos.csv', header=None)

# 特征融合
neg_flag = '_all'
negative_samples0 = pd.read_csv('../mashup_data/neg0.csv', header=None)
negative_samples1 = pd.read_csv('../mashup_data/neg1.csv', header=None)
negative_samples2 = pd.read_csv('../mashup_data/neg2.csv', header=None)
negative_samples3 = pd.read_csv('../mashup_data/neg3.csv', header=None)
negative_samples4 = pd.read_csv('../mashup_data/neg4.csv', header=None)
negative_samples5 = pd.read_csv('../mashup_data/neg5.csv', header=None)
negative_samples6 = pd.read_csv('../mashup_data/neg6.csv', header=None)
negative_samples = pd.concat([negative_samples0, negative_samples1, negative_samples2, negative_samples3,
                              negative_samples4, negative_samples5,negative_samples6])

negative_samples = negative_samples.sample(n=len(data), replace=False, random_state=42)
Negative = negative_samples.reset_index(drop=True)
'''
neg_flag = 3
Negative = pd.read_csv(f'../mashup_data/neg{neg_flag}.csv', header=None)
'''
Nindex = pd.read_csv('../mashup_data/NewRandomList.csv', header=None)

creat_var = globals()

dimension_dr = 100
dimension_di = 100
dimension_tar = 100
AllResult = []

dimension_dict = {"dimension":  [dimension_dr, dimension_di, dimension_tar]}
AllResult.append(dimension_dict)
metric_dict = {"Fold": '',
               "Accuracy": 'Accuracy',
               "Precision": 'Precision',
               "Recall": 'Recall',
               "MCC": 'MCC',
               "F1 Score": 'F1 Score',
               "AUC": 'AUC',
               "AUPR": "AUPR"
               }
AllResult.append(metric_dict)
#  训练集加上负样本

#  训练集加上负样本
for i in range(5):

    Attribute_drug = pd.read_csv(
        f'../../mashup/triple-net_dimension/dimension{dimension_dr}/dimension{dimension_dr}_feature_train{i}.csv',
        header=None, index_col=None)
    Attribute_disease = pd.read_csv(
        f'../../mashup/triple-net_dimension/dimension{dimension_di}/dimension{dimension_di}_feature_train{i}.csv',
        header=None, index_col=None)
    Attribute_target = pd.read_csv(
        f'../../mashup/triple-net_dimension/dimension{dimension_tar}/dimension{dimension_tar}_feature_train{i}.csv',
        header=None, index_col=None)
    drug_Attribute = Attribute_drug.iloc[:124, :]
    disease_Attribute = Attribute_disease.iloc[124:301, :]
    target_Attribute = Attribute_target.iloc[301:, :]
    drug_Attribute.index = drug_Attribute.index + 1
    disease_Attribute.index = disease_Attribute.index + 1
    target_Attribute.index = target_Attribute.index + 1

    train_data = pd.read_csv('../mashup_sample/ddt_train' + str(i) + '.csv', header=None)
    kk = []
    for j in range(5):
        if j != i:
            kk.append(j)
    index = np.hstack(
        [np.array(Nindex)[kk[0]], np.array(Nindex)[kk[1]], np.array(Nindex)[kk[2]], np.array(Nindex)[kk[3]]])
    result = train_data._append(pd.DataFrame(np.array(Negative)[index]))
    labels_train = result[3]
    #  将特征组合
    data_train_feature = pd.concat([drug_Attribute.loc[result[0].values.tolist()].reset_index(drop=True),
                                    disease_Attribute.loc[result[1].values.tolist()].reset_index(drop=True),
                                    target_Attribute.loc[result[2].values.tolist()].reset_index(drop=True)],
                                   axis=1)
    creat_var['data_train' + str(i)] = data_train_feature.values.tolist()
    creat_var['labels_train' + str(i)] = labels_train
    print(len(data_train_feature))
    print(len(labels_train))

    del labels_train, result, data_train_feature
    test_data = pd.read_csv('../mashup_sample/ddt_test' + str(i) + '.csv', header=None)
    result = test_data._append(pd.DataFrame(np.array(Negative)[np.array(Nindex)[i]]))
    labels_test = result[3]
    data_test_feature = pd.concat([drug_Attribute.loc[result[0].values.tolist()].reset_index(drop=True),
                                   disease_Attribute.loc[result[1].values.tolist()].reset_index(drop=True),
                                   target_Attribute.loc[result[2].values.tolist()].reset_index(drop=True)],
                                  axis=1)
    creat_var['data_test' + str(i)] = data_test_feature.values.tolist()
    creat_var['labels_test' + str(i)] = labels_test
    print(len(labels_test))
    del train_data, test_data, labels_test, result, data_test_feature
    print(i)

data_train = [data_train0, data_train1, data_train2, data_train3, data_train4]
data_test = [data_test0, data_test1, data_test2, data_test3, data_test4]
labels_train = [labels_train0, labels_train1, labels_train2, labels_train3, labels_train4]
labels_test = [labels_test0, labels_test1, labels_test2, labels_test3, labels_test4]

print(str(5) + "-CV")
tprs = []
aucs = []
auprs = []
accuracy_scores = []
precision_scores = []
recall_scores = []
mcc_scores = []
f1_scores = []
mean_fpr = np.linspace(0, 1, 1000)
for i in range(5):
    X_train, X_test = data_train[i], data_test[i]

    label_encoder = LabelEncoder()
    Y_train_encoded = label_encoder.fit_transform(labels_train[i])
    Y_test_encoded = label_encoder.transform(labels_test[i])

    best_XGB = XGBClassifier(n_estimators=999)
    best_XGB .fit(np.array(X_train), Y_train_encoded)
    y_score0 = best_XGB.predict(np.array(X_test))
    y_score_XGB = best_XGB.predict_proba(np.array(X_test))
    fpr, tpr, thresholds = roc_curve(Y_test_encoded, y_score_XGB[:, 1])
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    # auc
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

    precision_, recall_, _ = precision_recall_curve(Y_test_encoded, y_score_XGB[:, 1])
    # AUPR
    aupr = auc(recall_, precision_)
    auprs.append(aupr)

    # Accuracy
    accuracy = accuracy_score(Y_test_encoded, y_score0)
    accuracy_scores.append(accuracy)

    # Precision
    precision = precision_score(Y_test_encoded, y_score0)
    precision_scores.append(precision)

    # Recall
    recall = recall_score(Y_test_encoded, y_score0)
    recall_scores.append(recall)

    # Matthews correlation coefficient (MCC)
    mcc = matthews_corrcoef(Y_test_encoded, y_score0)
    mcc_scores.append(mcc)

    # F1 Score
    f1 = f1_score(Y_test_encoded, y_score0)
    f1_scores.append(f1)

    print('Fold %d' %(i+1))
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("MCC:", mcc)
    print("F1 Score:", f1)
    print('AUC:', roc_auc)
    print('AUPR:', aupr)
    # 保存结果
    result_dict = {
        "Fold": i,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "MCC": mcc,
        "F1 Score": f1,
        "AUC": roc_auc,
        "AUPR": aupr
    }
    AllResult.append(result_dict)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
mean_aupr = np.mean(auprs)

# Mean evaluation metrics
mean_accuracy = np.mean(accuracy_scores)
mean_precision = np.mean(precision_scores)
mean_recall = np.mean(recall_scores)
mean_mcc = np.mean(mcc_scores)
mean_f1 = np.mean(f1_scores)
print("=============================================")
print("Mean Accuracy:", mean_accuracy)
print("Mean Precision:", mean_precision)
print("Mean Recall:", mean_recall)
print("Mean MCC:", mean_mcc)
print("Mean F1 Score:", mean_f1)
print('Mean AUC:', mean_auc)
print('Mean AUPR:', mean_aupr)
print(f'XGBoost-dimension：{dimension_dr}-{dimension_di}-{dimension_tar}-neg：{neg_flag}')


mean_result_dict = {
    "Fold": "Mean",
    "Accuracy": mean_accuracy,
    "Precision": mean_precision,
    "Recall": mean_recall,
    "MCC": mean_mcc,
    "F1 Score": mean_f1,
    "AUC": mean_auc,
    "AUPR": mean_aupr
}
AllResult.append(mean_result_dict)
# 将结果保存到Excel文件
#result_df = pd.DataFrame(AllResult)
#result_df = result_df.applymap(lambda x: f'{x:.4f}' if isinstance(x, (float, int)) else x)
#result_df.to_excel("classification_results_xgboost.xlsx", index=False)
