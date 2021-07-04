"""
Keras LSTM, multi-task & multi-outputs prediction (also can be used in multi-label situation)

"""
from numpy.random import seed
seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)

import pandas as pd
import numpy as np
from numpy import concatenate
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.layers import Input, Embedding, LSTM, Dense
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from keras.models import Model
import helper_funcs
from sklearn.metrics import confusion_matrix
import seaborn as sns

import os
import warnings
import glob
import numpy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings



def build_model(trainX,lstm_layer, drop, r_drop, shared_layer,dense_num, n_labels):

    """
    Keras Function model
    """
    input = Input(shape=(trainX.shape[1], trainX.shape[2]),name='input' )
    lstm_out = LSTM(lstm_layer, activation='relu', dropout=drop, recurrent_dropout=r_drop)(input)

    dense_shared = Dense(shared_layer, activation='relu')(lstm_out)


    sub = Dense(dense_num, activation='relu')(dense_shared)
    out = Dense(n_labels, activation='sigmoid')(sub)

    model = Model(inputs=input, outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    print(model.summary())
    return model


def confu_matrix_plot(true_multiclass,pred_multiclass):
    labels = ['read', 'writeQA', 'write','type', 'rest', 'off']

    cm = confusion_matrix(true_multiclass, pred_multiclass,labels)
    return cm


def main():
    # network parameters
    task_num = 1
    # dense_att = 128
    lstm_layer = 64
    drop = 0.3
    r_drop = 0.3
    l2_value = 0.001
    shared_layer = 576
    dense_num = 64

    look_back = 10  # number of previous timestamp used for training
    n_columns = 71  # total columns
    n_labels = 6  # number of labels

    for i in [1, 2, 3, 5, 6, 7, 8, 9]:
        ps = 'p' + str(i)
        train_data = 'data_sample/processed/wo' + ps + '.csv'
        test_data = 'data_sample/processed/' + ps + '.csv'
        #train_data = 'data_sample/p2.csv'
        #test_data = 'data_sample/p1.csv'

        train, scaled, scaler = helper_funcs.load_dataset(train_data)
        train_X, train_y = helper_funcs.split_dataset(scaled, look_back, n_columns, n_labels)

        test, scaled_test, scaler_test = helper_funcs.load_dataset(test_data)
        test_X, test_y = helper_funcs.split_dataset(scaled, look_back, n_columns, n_labels)

        model = build_model(train_X, lstm_layer, drop, r_drop, shared_layer, dense_num, n_labels)

        import time
        start_time = time.time()

        # fit network
        history = model.fit(train_X, train_y,
                            epochs=100,
                            batch_size=50,
                            validation_split=0.25,
                            # validation_data=(testX_list, testy_list),
                            verbose=2,
                            shuffle=False,
                            callbacks=[
                                keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=2,
                                                              mode='min')]
                            )
        end_time = time.time()
        print('--- %s seconds ---' % (end_time - start_time))

        # make prediction
        pred_time = time.time()

        y_pred = model.predict(test_X)
        pred_end_time = time.time()

        # ===========================================================================================#
        # write parameters & results to file
        file = open('results/M_LSTM' + ps + '.txt', 'w')

        file.write('task_num:' + str(task_num) + '\n')
        file.write('lstm_layer:' + str(lstm_layer) + '\n')
        file.write('drop:' + str(drop) + '\n')
        file.write('r_drop:' + str(r_drop) + '\n')
        file.write('l2_value:' + str(l2_value) + '\n')
        file.write('shared_layer:' + str(shared_layer) + '\n')
        file.write('dense_num:' + str(dense_num) + '\n')

        # balance accuracy
        # for i in range(len(train_data)):
        Bacc = helper_funcs.evaluation(test_X, test_y, y_pred,
                                       look_back, n_columns, n_labels, scaler_test)

        file.write('BA:' + ' ' + str(Bacc[0]) + ' ')
        file.write('F1:' + ' ' + str(Bacc[1]) + ' ')
        file.write('precision:' + ' ' + str(Bacc[2]) + '\n')
        file.write('recall:' + ' ' + str(Bacc[3]) + '\n')
        file.write('accuracy:' + ' ' + str(Bacc[4]) + '\n')
        confu_matrix = confu_matrix_plot(Bacc[5], Bacc[6])
        print(confu_matrix)
        numpy.savetxt("confu_matrix/confusion_matrix_M_LSTM" + ps + ".csv", X=confu_matrix.astype(int), delimiter=', ', fmt='%.0f')

        file.write('training time:' + str(end_time - start_time))
        file.write('prediction time:' + str(pred_end_time - pred_time))
    
if __name__ == '__main__':
    main()


