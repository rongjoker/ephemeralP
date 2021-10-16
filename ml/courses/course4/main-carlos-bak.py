# -*- coding: utf-8 -*-
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as sk
from sklearn.preprocessing._encoders import OneHotEncoder
from sklearn.preprocessing._label import LabelEncoder
from sklearn import preprocessing, metrics
from sklearn.neural_network._multilayer_perceptron import MLPClassifier
from sklearn.tree._classes import DecisionTreeClassifier
from sklearn.ensemble._forest import RandomForestClassifier
from numpy.random.mtrand import np
from sklearn.model_selection._split import train_test_split

def main():
    allData = pd.read_excel('/Users/zhangshipeng/Downloads/fd/machineLearning/Booking1.xlsx')
    allData = allData.dropna()
    encoder(pd.concat([allData], axis=0, ignore_index=True))
    leAgmt = joblib.load('leAgmt.pkl')
    leEB = joblib.load('leEB.pkl')
    leGrp = joblib.load('leGrp.pkl')
    leSH = joblib.load('leSH.pkl')
    leFW = joblib.load('leFW.pkl')
    leCN = joblib.load('leCN.pkl')
    ohe = joblib.load('ohe.pkl')
    A, A2, B, B2 = train_test_split(allData[['AGMT_ID', 'ISEBOOKING', 'OOCL_CMDTY_GRP_CDE', 'SH_COOP_FREQ', 'FW_COOP_FREQ', 'CN_COOP_FREQ']], allData.IS_CONCEAL, random_state=1234567, test_size=0.1)
    X = A.values
    y = B.values
    
    X2 = A2.values
    y2 = B2.values
    # 类别数据,需要进行标签编码
    X[:, 0] = leAgmt.transform(X[:, 0])
    X[:, 1] = leEB.transform(X[:, 1])
    X[:, 2] = leGrp.transform(X[:, 2])
    X[:, 3] = leSH.transform(X[:, 3])
    X[:, 4] = leFW.transform(X[:, 4])
    X[:, 5] = leCN.transform(X[:, 5])
    
    X2[:, 0] = leAgmt.transform(X2[:, 0])
    X2[:, 1] = leEB.transform(X2[:, 1])
    X2[:, 2] = leGrp.transform(X2[:, 2])
    X2[:, 3] = leSH.transform(X2[:, 3])
    X2[:, 4] = leFW.transform(X2[:, 4])
    X2[:, 5] = leCN.transform(X2[:, 5])

    # 将y转为一维
    #y = LabelEncoder().fit_transform(y.ravel()) 
    #y2 = LabelEncoder().fit_transform(y2.ravel()) 

    # 对离散特征进行独热编码，扩维
    X = np.hstack((X[:, [0, 1]], ohe.transform(X[:, [ 2, 3, 4, 5]])))
    X2 = np.hstack((X2[:, [0, 1]], ohe.transform(X2[:, [ 2, 3, 4, 5]])))

    #决策树
    decisionTreeClf = DecisionTreeClassifier()
    decisionTreeClf.fit(X, y)
    predTree = decisionTreeClf.predict(X2)
    # y2_scoreTree = decisionTreeClf.predict_proba(X2)
    #[:, 1]
    
    #随机森林
    oldRandomForestClf = RandomForestClassifier(n_estimators=50)
    oldRandomForestClf.fit(X, y)
    y2_scoreOldRf = oldRandomForestClf.predict_proba(X2)[:, 1] 
    # predOldRf = oldRandomForestClf.predict(X2)

    #改进版随机森林
    randomForestClf = RandomForestClassifier(n_estimators=50)
    randomForestClf.fit(X, y)
    # predNewRlf = randomForestClf.predict(X2)
    y2_scoreNewRf = randomForestClf.predict_proba(X2)[:, 1] 
    fpr, tpr, threshold = metrics.roc_curve(B2.map({False:0, True:1}), y2_scoreNewRf)
    
    #逻辑回归
    lr = sk.LogisticRegressionCV()
    lr.fit(X, y)
    # predLogic = lr.predict(X2)
    y2_scoreLogic = lr.predict_proba(X2)[:, 1] 

    #神经网络
    mlp = MLPClassifier()
    mlp.fit(X, y)
    # predMlp = mlp.predict(X2)
    y2_scoreMlp = mlp.predict_proba(X2)[:, 1]
    
    
    evaluate_model(y2, predTree, y2_scoreOldRf, y2_scoreNewRf, y2_scoreLogic, y2_scoreMlp)
def encoder(data):
    X = data.loc[:, ['AGMT_ID', 'ISEBOOKING', 'OOCL_CMDTY_GRP_CDE', 'SH_COOP_FREQ', 'FW_COOP_FREQ', 'CN_COOP_FREQ']].values
    y = data.loc[:, ['IS_CONCEAL']].values
    
    # 类别数据,需要进行标签编码
    leAgmt = preprocessing.LabelEncoder()
    leAgmt = leAgmt.fit(X[:, 0])
    X[:, 0] = leAgmt.transform(X[:, 0])
    
    leEB = preprocessing.LabelEncoder()
    leEB = leEB.fit(X[:, 1])
    X[:, 1] = leEB.transform(X[:, 1])
    
    leGrp = preprocessing.LabelEncoder()
    leGrp = leGrp.fit(X[:, 2])
    X[:, 2] = leGrp.transform(X[:, 2])
    
    leSH = preprocessing.LabelEncoder()
    leSH = leSH.fit(X[:, 3])
    X[:, 3] = leSH.transform(X[:, 3])
    
    leFW = preprocessing.LabelEncoder()
    leFW = leFW.fit(X[:, 4])
    X[:, 4] = leFW.transform(X[:, 4])
    
    leCN = preprocessing.LabelEncoder()
    leCN = leCN.fit(X[:, 5])
    X[:, 5] = leCN.transform(X[:, 5])
    
    # 将y转为一维
    y = LabelEncoder().fit_transform(y.ravel()) 
    
    ohe = OneHotEncoder(sparse=False, categories='auto')
    ohe = ohe.fit(X[:, [ 2, 3, 4, 5]])
    ohe.transform(X[:, [ 2, 3, 4, 5]])
    
    joblib.dump(leAgmt, 'leAgmt.pkl')
    joblib.dump(leEB, 'leEB.pkl')
    joblib.dump(leGrp, 'leGrp.pkl')
    joblib.dump(leSH, 'leSH.pkl')
    joblib.dump(leFW, 'leFW.pkl')
    joblib.dump(leCN, 'leCN.pkl')
    joblib.dump(ohe, 'ohe.pkl')
    
    return
def evaluate_model(test_labels, predTree, y2_scoreOldRf, y2_scoreNewRf, y2_scoreLogic, y2_scoreMlp):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""
    
    baseline = {}
    baseline['recall'] = metrics.recall_score(test_labels, [1 for _ in range(len(test_labels))])
    baseline['precision'] = metrics.precision_score(test_labels, [1 for _ in range(len(test_labels))])
    baseline['roc'] = 0.5
    
    # results = {}
    # results['recall'] = metrics.recall_score(test_labels, predictions)
    # results['precision'] = metrics.precision_score(test_labels, predictions)
    # results['roc'] = metrics.roc_auc_score(test_labels, probs)
    
    # resultsLogic = {}
    # resultsLogic['recall'] = metrics.recall_score(test_labels, predLogic)
    # resultsLogic['precision'] = metrics.precision_score(test_labels, predLogic)
    # resultsLogic['roc'] = metrics.roc_auc_score(test_labels, y2_scoreLogic)
    
#     for metric in ['recall', 'precision', 'roc']:
#         print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')
    
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = metrics.roc_curve(test_labels, [1 for _ in range(len(test_labels))])
    tree_fpr, tree_tpr, _ = metrics.roc_curve(test_labels, predTree)
    old_rf_fpr, old_rf_tpr, _ = metrics.roc_curve(test_labels, y2_scoreOldRf)
    model_fpr, model_tpr, _ = metrics.roc_curve(test_labels, y2_scoreNewRf)
    logic_fpr, logic_tpr, _ = metrics.roc_curve(test_labels, y2_scoreLogic)
    mlp_fpr, mlp_tpr, _ = metrics.roc_curve(test_labels, y2_scoreMlp)
    # bys_fpr, bys_tpr, _ = metrics.roc_curve(test_labels, y2_scoreBys)

    plt.figure(figsize=(8, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['font.size'] = 12
    
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label='baseline')
    plt.plot(tree_fpr, tree_tpr, 'g', label='决策树')
    plt.plot(model_fpr, model_tpr, 'r', label='改进随机森林')
    plt.plot(old_rf_fpr, old_rf_tpr, 'k', label='传统随机森林')
    plt.plot(logic_fpr, logic_tpr, 'y', label='逻辑回归')
    plt.plot(mlp_fpr, mlp_tpr, 'm', label='神经网络')
    # plt.plot(bys_fpr, bys_tpr, 'c', label='贝叶斯网络')
    plt.legend();
    plt.xlabel('假阳性率'); 
    plt.ylabel('真阳性率'); 
    plt.title('ROC 曲线');
    plt.show()
if __name__ == '__main__':
    main()
