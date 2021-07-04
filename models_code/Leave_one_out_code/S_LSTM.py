"""
Keras LSTM, single task & multi-label prediction

"""
from numpy.random import seed
seed(1)

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM
from keras.models import Model
from keras.layers import Input, Dense
import helper_funcs
from numpy.random import seed
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
import glob
import numpy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings




def build_model(train):
    """
    Model 1: keras Sequential model
    """
    # model = Sequential()
    # model.add(Convolution1D(nb_filter=128, filter_length=4, activation='relu',input_shape=(train.shape[1], train.shape[2])))
    # model.add(Convolution1D(nb_filter=64, filter_length=2, activation='relu'))
    # # model.add(MaxPooling1D(pool_size=4))
    # model.add(LSTM(64, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(51, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['binary_accuracy'])

    """
    Model 2: keras Function model
    """
    inputs = Input(shape=(train.shape[1], train.shape[2]))
    x = LSTM(64, activation='relu',dropout=0.2)(inputs) # 'return_sequences=True' if connectted to another LSTM or Conv layer.
    x = Dense(64, activation='relu')(x)
    outputs = Dense(6,activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['binary_accuracy'])
    print(model.summary())

    return model

def confu_matrix_plot(true_multiclass,pred_multiclass):
    labels = ['read', 'writeQA', 'write','type', 'rest', 'off']#

    cm = confusion_matrix(true_multiclass, pred_multiclass,labels)
    return cm



def main():

    look_back = 10 # number of previous timestamp used for training
    n_columns = 71 # total columns
    n_labels = 6 # number of labels
    split_ratio = 0.8 # train & test data split ratio

    sum_bacc = 0
    sum_re = 0
    sum_pre = 0
    sum_F1 = 0
    sum_acc = 0

    file = open('results/S_LSTM.txt', 'w')


    import time
    start_time = time.time()

    for i in [1, 2, 3, 5, 6, 7, 8, 9]:
        ps = 'p' + str(i)
        train_data = 'data8P/processed/wo' + ps + '.csv'
        test_data = 'data8P/processed/' + ps + '.csv'
        #train_data = 'data_sample/p2.csv'
        #test_data = 'data_sample/p1.csv'

        #file = open('results/S_LSTM' + ps + '.txt', 'w')

        train, scaled, scaler = helper_funcs.load_dataset(train_data)
        train_X, train_y = helper_funcs.split_dataset(scaled, look_back, n_columns, n_labels)

        test, scaled_test, scaler_test = helper_funcs.load_dataset(test_data)
        test_X, test_y = helper_funcs.split_dataset(scaled, look_back, n_columns, n_labels)

        model = build_model(train_X)

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


        Bacc = helper_funcs.evaluation(test_X, test_y, y_pred,
                                       look_back, n_columns, n_labels, scaler_test)
        sum_bacc = sum_bacc + Bacc[0]
        sum_F1 = sum_F1 + Bacc[1]
        sum_pre = sum_pre + Bacc[2]
        sum_re = sum_re + Bacc[3]
        sum_acc = sum_acc + Bacc[4]

        file.write('BA:' + ' ' + str(Bacc[0]) + ' ')
        file.write('F1:' + ' ' + str(Bacc[1]) + ' ')
        file.write('precision:' + ' ' + str(Bacc[2]) + '\n')
        file.write('recall:' + ' ' + str(Bacc[3]) + '\n')
        file.write('accuracy:' + ' ' + str(Bacc[4]) + '\n')
        confu_matrix = confu_matrix_plot(Bacc[5], Bacc[6])
        print(confu_matrix)
        numpy.savetxt("confu_matrix/confusion_matrix_S_LSTM"+ ps + ".csv", X=confu_matrix.astype(int), delimiter=', ', fmt='%.0f')

        file.write('training time:' + str(end_time - start_time))
        file.write('prediction time:' + str(pred_end_time - pred_time))

    file.write('avg_bacc: ' + str(sum_bacc / 8.0) + '\n')
    file.write('avg_TPR: ' + str(sum_re / 8.0) + '\n')
    file.write('avg_precision: ' + str(sum_pre / 8.0) + '\n')
    file.write('avg_F1: ' + str(sum_F1 / 8.0) + '\n')
    file.write('avg_accuracy: ' + str(sum_acc / 8.0) + '\n')

if __name__ == '__main__':
    main()
