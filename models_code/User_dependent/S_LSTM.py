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

def confu_matrix_plot(true_multiclass,pred_multiclass,i):
    labels = ['read', 'writeQA', 'write','type', 'rest', 'off']
    # labels_rever = ['off', 'rest', 'type','write', 'wr', 'read']

    cm = confusion_matrix(true_multiclass, pred_multiclass,labels)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    sns.heatmap(cm/np.sum(cm), annot=True, fmt=".2%",ax=ax,cmap='Blues')  # annot=True to annotate cells

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.xaxis.set_label_position('top')

    # ax.set_title('Confusion Matrix')
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.xaxis.set_ticks_position('top')

    plt.savefig('results/confusion_matrix_S_LSTM_'+str(i)+'.png', dpi=150)

def confu_matrix_save(true_multiclass,pred_multiclass):
    labels = ['read', 'writeQA', 'write','type', 'rest', 'off']
    cm = confusion_matrix(true_multiclass, pred_multiclass,labels)
    return cm

def main():

    look_back = 10 # number of previous timestamp used for training
    n_columns = 71 # total columns
    n_labels = 6 # number of labels
    split_ratio = 0.8 # train & test data split ratio

    file_list = glob.glob('data8p/processed/*.csv')

    file = open('results/S_LSTM.txt', 'w')
    sum_bacc = 0
    sum_re = 0
    sum_pre = 0
    sum_F1 = 0
    sum_acc = 0

    import time
    start_time = time.time()
    for i in range(len(file_list)):
        locals()['dataset' + str(i)] = file_list[i]

        locals()['dataset' + str(i)], locals()['scaled' + str(i)], locals()['scaler' + str(i)] = helper_funcs.load_dataset(
            locals()['dataset' + str(i)])



        # split into train and test sets
        locals()['train_X' + str(i)], locals()['train_y' + str(i)], locals()['test_X' + str(i)], locals()[
            'test_y' + str(i)] = helper_funcs.split_dataset(locals()['dataset' + str(i)], locals()['scaled' + str(i)], look_back,
                                               n_columns, n_labels, split_ratio)


        model = build_model( locals()['train_X' + str(i)])

        

        # fit network
        history = model.fit( locals()['train_X' + str(i)], locals()['train_y' + str(i)],
                             epochs= 100,
                             batch_size= 50,
                            # validation_data=(locals()['test_X' + str(i)], locals()['test_y' + str(i)]),
                            validation_split=0.25,
                            verbose=2,
                            shuffle=False,
                            callbacks=[
                                keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=25,
                                                              mode='min')]
                            )



        y_predict = model.predict(locals()['test_X' + str(i)])

        results = helper_funcs.evaluation(locals()['test_X' + str(i)], locals()['test_y' + str(i)], y_predict, look_back, n_columns, n_labels, locals()['scaler' + str(i)])

        sum_bacc = sum_bacc + results[0]
        sum_F1 = sum_F1 + results[1]
        sum_pre = sum_pre + results[2]
        sum_re = sum_re + results[3]
        sum_acc = sum_acc + results[4]

        file.write('BA:' + ' ' + str(results[0]) + ' ')
        file.write('F1:' + ' ' + str(results[1]) + ' ')
        file.write('precision:' + ' ' + str(results[2]) + '\n')
        file.write('recall:' + ' ' + str(results[3]) + '\n')
        file.write('accuracy:' + ' ' + str(results[4]) + '\n')
        confu_matrix_plot(results[5],results[6],i)

        confu_matrix = confu_matrix_save(results[5],results[6])
        print(confu_matrix)
        numpy.savetxt("confu_matrix/confusion_matrix_S-LSTM_" + str(i) + ".csv", X=confu_matrix.astype(int),
                      delimiter=', ',
                      fmt='%.0f')
        confu_matrix = confu_matrix / np.sum(confu_matrix) * 1000
        print(confu_matrix)
        numpy.savetxt("confu_matrix/confusion_matrix_S-LSTM-" + str(i) + "-nor.csv", X=confu_matrix.astype(float),
                      delimiter=', ',
                      fmt='%.00f')

    end_time = time.time()
    print('--- %s seconds ---' % (end_time - start_time))

    file.write('avg_bacc: ' + str(sum_bacc / len(file_list)) + '\n')
    file.write('avg_TPR: ' + str(sum_re / len(file_list)) + '\n')
    file.write('avg_precision: ' + str(sum_pre / len(file_list)) + '\n')
    file.write('avg_F1: ' + str(sum_F1 / len(file_list)) + '\n')
    file.write('avg_accuracy: ' + str(sum_acc / len(file_list)) + '\n')
    file.write('training time:' + str(end_time - start_time))



if __name__ == '__main__':
    main()
