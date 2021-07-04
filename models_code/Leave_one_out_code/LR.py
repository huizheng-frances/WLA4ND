import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
from keras.layers.normalization import BatchNormalization
import warnings
import numpy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

def features_labels_process(data):
    # data = data[0:3000]
    all_lables_list = [col for col in data if col.startswith('label')]
    all_features_list = [col for col in data if col not in all_lables_list]

    features = data[all_features_list]
    labels = data[all_lables_list]

    # For missing sensor-features, replace "nan" with the upper cell(ffill) or below cell(bfill) value
    features = features.fillna(method='ffill')
    # drop the feature column with all "nan"
    # features = features.dropna(axis=1, how='all')
    features = features.fillna(method='bfill')
    features = features.fillna(0) # after forward-filling and back-filling, fill the rest nan cells with 0

    # normalize feature vaules, by subtracting mean, then divide std
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(features)
    normalized_features = pd.DataFrame(np_scaled)

    # fill empty labels with 0
    labels = labels.fillna(0)
    return (normalized_features,labels)

def confu_matrix(y_true, y_pred):
    true_multiclass = []

    for j in range(len(y_true)):
        # print(y_true[j])
        try:
        # for j in range(len(y_true[0])):
            if y_true[j][0] == 1:
                true_multiclass.append('read')
            elif y_true[j][1] == 1:
                true_multiclass.append('writeQA')
            elif y_true[j][2] == 1:
                true_multiclass.append('write')
            elif y_true[j][3] == 1:
                true_multiclass.append('type')
            elif y_true[j][4] == 1:
                true_multiclass.append('rest')
            else:
                true_multiclass.append('off')
        except:
            print(y_true[j])

    pred_multiclass = []
    for k in range(len(y_pred)):
        # for j in range(len(y_true[0])):
        try:
            if y_pred[k][0] == 1:
                pred_multiclass.append('read')
            elif y_pred[k][1] == 1:
                pred_multiclass.append('writeQA')
            elif y_pred[k][2] == 1:
                pred_multiclass.append('write')
            elif y_pred[k][3] == 1:
                pred_multiclass.append('type')
            elif y_pred[k][4] == 1:
                pred_multiclass.append('rest')
            else:
                pred_multiclass.append('off')
        except:
            print(y_pred[k])

    return true_multiclass,pred_multiclass

def evaluation(y_pred,y_true):

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    total = 0
    y_true = y_true.as_matrix()
    for i in range(len(y_pred)):
        for j in range(len(y_pred[i])):
            if y_pred[i][j] == 1 and y_true[i][j] == 1:
                tp += 1
            elif y_pred[i][j] == 1 and y_true[i][j] == 0:
                fp += 1
            elif y_pred[i][j] == 0 and y_true[i][j] == 0:
                tn += 1
            elif y_pred[i][j] == 0 and y_true[i][j] == 1:
                fn += 1
            total += 1

    # true positive rate
    sensitivity = float(tp) / (tp + fn)
    # true negative rate
    specificity = float(tn) / (tn + fp)
    precision = float(tp) / (tp + fp)

    if (precision + sensitivity) ==0:
        F1 = 0
    else:
        F1 = (2 * precision * sensitivity) / (precision + sensitivity)
    # naive accuracy
    accuracy = float(tn + tp) / total

    # # Balanced accuracy is a more fair replacement for the naive accuracy:
    balanced_accuracy = (sensitivity + specificity) / 2.
    true_multiclass,pred_multiclass = confu_matrix(y_true, y_pred)

    #
    print("-" * 10);
    print('Accuracy*:         %.2f' % accuracy);
    print('Sensitivity (TPR): %.2f' % sensitivity);
    print('Specificity (TNR): %.2f' % specificity);
    print('Balanced accuracy: %.2f' % balanced_accuracy);
    print('Precision: %.2f' % precision);
    print('F1: %.2f' % F1);
    print("-" * 10);
    return accuracy, sensitivity, specificity, balanced_accuracy,tp,fn,fp,tn,precision,F1,true_multiclass,pred_multiclass

def confu_matrix_plot(true_multiclass,pred_multiclass):
    labels = ['read', 'writeQA', 'write','type', 'rest', 'off']

    cm = confusion_matrix(true_multiclass, pred_multiclass,labels)
    return cm

if __name__ == '__main__':

    file = open('results/LR.txt', 'w')
    sum_bacc = 0
    sum_re = 0
    sum_pre = 0
    sum_F1 = 0
    sum_acc = 0

    for i in [1, 2, 3, 5, 6, 7, 8, 9]:
        ps = 'p' + str(i)
        train_data = pd.read_csv('data8P/processed/wo' + ps + '.csv')
        test_data = pd.read_csv('data8P/processed/' + ps + '.csv')
        #train_data = pd.read_csv('data_sample/p2.csv')
        #test_data = pd.read_csv('data_sample/p1.csv')



        Num_tp = 0
        Num_fn = 0
        Num_fp = 0
        Num_tn = 0
        train_time = 0

        import time

        start_time = time.time()

        ## Part 2 ************** fixed train & test prediction *****************

        X_train, Y_train = features_labels_process(train_data)
        X_test, Y_test = features_labels_process(test_data)



        ## ------- Fit an independent logistic regression model or MLP for each class using the OneVsRestClassifier wrapper.

        ovr = OneVsRestClassifier(LogisticRegression())

        # ------ evaluation of test data predicion  -----------------------#

        ovr.fit(X_train, Y_train)
        Y_pred = ovr.predict(X_test)



        results = evaluation(Y_pred, Y_test)

        sum_bacc = sum_bacc + results[3]
        sum_F1 = sum_F1 + results[9]
        sum_pre = sum_pre + results[8]
        sum_re = sum_re + results[1]
        sum_acc = sum_acc + results[0]

        file.write('BA:' + ' ' + str(results[3]) + ' ')
        file.write('F1:' + ' ' + str(results[9]) + ' ')
        file.write('precision:' + ' ' + str(results[8]) + '\n')
        file.write('recall:' + ' ' + str(results[1]) + '\n')
        file.write('accuracy:' + ' ' + str(results[0]) + '\n')
        #confu_matrix = confu_matrix_plot(results[10], results[11])
        #print(confu_matrix)
        #numpy.savetxt("confu_matrix/confusion_matrix_LR"+ ps +".csv", X=confu_matrix.astype(int), delimiter=', ', fmt='%.0f')

    file.write('avg_bacc: ' + str(sum_bacc / 8.0) + '\n')
    file.write('avg_TPR: ' + str(sum_re / 8.0) + '\n')
    file.write('avg_precision: ' + str(sum_pre / 8.0) + '\n')
    file.write('avg_F1: ' + str(sum_F1 / 8.0) + '\n')
    file.write('avg_accuracy: ' + str(sum_acc / 8.0) + '\n')






