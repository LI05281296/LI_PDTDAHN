import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf
import gc
import random
import os
from clac_metric import cv_model_evaluate
from utils import *
from model import GCNModel
from opt import Optimizer

from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, f1_score, roc_curve, auc, \
    precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def PredictScore(train_matrix, sim_matrix_1, sim_matrix_2, seed, epochs, emb_dim, dp, lr, adjdp):
    np.random.seed(seed)
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    adj = constructHNet(train_matrix, sim_matrix_1, sim_matrix_2)
    adj = sp.csr_matrix(adj)
    association_nam = train_matrix.sum()
    X = constructNet(train_matrix)
    features = sparse_to_tuple(sp.csr_matrix(X))
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    adj_orig = train_matrix.copy()
    adj_orig = sparse_to_tuple(sp.csr_matrix(adj_orig))

    adj_norm = preprocess_graph(adj)
    adj_nonzero = adj_norm[1].shape[0]
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'adjdp': tf.placeholder_with_default(0., shape=())
    }
    model = GCNModel(placeholders, num_features, emb_dim,
                     features_nonzero, adj_nonzero, train_matrix.shape[0], name='LAGCN')
    with tf.name_scope('optimizer'):
        opt = Optimizer(
            preds=model.reconstructions,
            labels=tf.reshape(tf.sparse_tensor_to_dense(
                placeholders['adj_orig'], validate_indices=False), [-1]),
            model=model,
            lr=lr, num_u=train_matrix.shape[0], num_v=train_matrix.shape[1], association_nam=association_nam)
    sess = tf.Session(config = config)
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        feed_dict = dict()
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['adj']: adj_norm})
        feed_dict.update({placeholders['adj_orig']: adj_orig})
        feed_dict.update({placeholders['dropout']: dp})
        feed_dict.update({placeholders['adjdp']: adjdp})
        _, avg_cost = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
        if epoch % 100 == 0:
            feed_dict.update({placeholders['dropout']: 0})
            feed_dict.update({placeholders['adjdp']: 0})
            res = sess.run(model.reconstructions, feed_dict=feed_dict)
            print("Epoch:", '%04d' % (epoch + 1),
                  "train_loss=", "{:.5f}".format(avg_cost))
    print('Optimization Finished!')
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['adjdp']: 0})
    res = sess.run(model.reconstructions, feed_dict=feed_dict)
    sess.close()
    return res

def cross_validation_experiment(train_matrix, sim_matrix_1, sim_matrix_2, seed, epochs, emb_dim, dp, lr, adjdp, model_name):
    index_matrix = np.mat(np.where(train_matrix < 2))
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(seed)
    random.shuffle(random_index)
    k_folds = 5
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam %
                                 k_folds]).reshape(k_folds, CV_size,  -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + \
        random_index[association_nam - association_nam % k_folds:]
    random_index = temp
    '''
    metric = np.zeros((1, 7))
    xroc = np.zeros((1, 100))
    yroc = np.zeros((1, 100))
    xpr = np.zeros((1, 100))
    ypr = np.zeros((1, 100))
    print("seed=%d, evaluating %s...." % (seed, model_name))
    '''
    all_predictions = []
    for k in range(k_folds):
        print("------this is %dth cross validation------" % (k+1))
        train_fold_matrix = np.matrix(train_matrix, copy=True)
        test_fold_matrix = np.matrix(train_matrix, copy=True)
        train_fold_matrix[tuple(np.array(random_index[k]).T)] = 0
        test_fold_matrix[tuple(np.array(random_index[k]).T)] = 2
        len_1 = train_matrix.shape[0]
        len_2 = train_matrix.shape[1]
        res = PredictScore(
            train_fold_matrix, sim_matrix_1, sim_matrix_2, seed, epochs, emb_dim, dp, lr,  adjdp)
        res = res.reshape(len_1, len_2)
        all_predictions.append(res)
        np.savetxt(f'../../data/data_ddt/{model_name}_predict_{k}.csv', res, delimiter=",")
    return all_predictions

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
    drug_sim = np.loadtxt('../../data/data/drug_sim_matrix.txt', delimiter=' ')
    disease_sim = np.loadtxt('../../data/data/disease_sim_matrix.txt', delimiter=' ')
    drug_dis_matrix = np.loadtxt('../../data/data/drdi_matrix.txt', delimiter=' ')

    target_sim = np.loadtxt('../../data/data/target_sim_matrix.txt', delimiter=' ')
    drug_target_matrix = np.loadtxt('../../data/data/drtar_matrix.txt', delimiter=' ')
    disease_target_matrix = np.loadtxt('../../data/data/ditar_matrix.txt', delimiter=' ')
    # Parameters
    epoch = 10000
    emb_dim = 128
    lr = 0.01
    adjdp = 0.1
    dp = 0.1
    simw = 6

    # Train and save prediction results for each model
    drug_disease_res = cross_validation_experiment(drug_dis_matrix, drug_sim*simw, disease_sim*simw, 0, epoch, emb_dim, dp, lr, adjdp, "drug_disease")
    drug_target_res = cross_validation_experiment(drug_target_matrix, drug_sim*simw, target_sim*simw, 0, epoch, emb_dim, dp, lr, adjdp, "drug_target")
    disease_target_res = cross_validation_experiment(disease_target_matrix, disease_sim*simw, target_sim*simw, 0, epoch, emb_dim, dp, lr, adjdp, "disease_target")

    # Load positive and negative samples
    positive_samples = pd.read_csv('../../data/data_ddt/pos.csv', names=['drug', 'disease', 'target', 'label'],
                                   header=None)
    '''
    negative_samples0 = pd.read_csv('../../data/data/neg0.csv', names=['drug', 'disease', 'target', 'label'], header=None)
    negative_samples1 = pd.read_csv('../../data/data/neg1.csv', names=['drug', 'disease', 'target', 'label'],
                                   header=None)
    negative_samples2 = pd.read_csv('../../data/data/neg2.csv', names=['drug', 'disease', 'target', 'label'],
                                   header=None)
    negative_samples3 = pd.read_csv('../../data/data/neg3.csv', names=['drug', 'disease', 'target', 'label'],
                                   header=None)
    negative_samples4 = pd.read_csv('../../data/data/neg4.csv', names=['drug', 'disease', 'target', 'label'],
                                   header=None)
    negative_samples5 = pd.read_csv('../../data/data/neg5.csv', names=['drug', 'disease', 'target', 'label'],
                                   header=None)
    negative_samples6 = pd.read_csv('../../data/data/neg6.csv', names=['drug', 'disease', 'target', 'label'],
                                   header=None)
    negative_samples = pd.concat([negative_samples0,negative_samples1,negative_samples2,negative_samples3,negative_samples4,negative_samples5,negative_samples6])

    negative_samples = negative_samples.sample(n=len(positive_samples), replace=False, random_state=42)
    '''
    negative_samples = pd.read_csv('../../data/data_ddt/neg6.csv', names=['drug', 'disease', 'target', 'label'],
                                   header=None)
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
