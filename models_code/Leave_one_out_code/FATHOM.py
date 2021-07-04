"""
Proposed model: Hierarchical Attention model. First attention layer applied on each task's input_dim, second attention layer applied on joined tasks' TIME_step dimension.

"""
import keras
from keras.layers.core import *
from keras.models import *
from keras.layers import Input, Embedding, Dense,Convolution1D,MaxPooling1D,merge
from keras.layers.recurrent import LSTM
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
import glob
import helper_funcs
import tensorflow as tf
import numpy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings
# tf.enable_eager_execution()

print(tf.executing_eagerly())


TIME_STEPS = 10
SINGLE_ATTENTION_VECTOR = False

# attention applied on TIME_STEP dimension
def attention_time(inputs,i):
    ## inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction'+str(i))(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vect'+str(i))(a)
    output_attention_mul = keras.layers.multiply([inputs, a_probs], name='attention_mul'+str(i))
    return output_attention_mul

def attention_3d_block(shared,inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    activation_weights= Flatten()(shared)
    activation_weights=Dense(TIME_STEPS,activation='tanh')(activation_weights)
    activation_weights=Activation('softmax')(activation_weights)
    activation_weights= RepeatVector(input_dim)(activation_weights)
    activation_weights=Permute([2,1])(activation_weights)
    activation_weighted=keras.layers.multiply([inputs, activation_weights])

    # sum_weighted = Lambda(lambda x: K.sum(x, axis=-2), output_shape=(input_dim,))(activation_weighted)
    return activation_weighted

# attention applied on input_dim dimension
def attention_dim(inputs):
    # hidden_states.shape = (batch_size, time_steps, hidden_size)
    hidden_size = int(inputs.shape[2])
    score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(inputs)
    # h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(inputs)
    # score = dot([score_first_part, h_t], [2, 1], name='attention_score')
    attention_weights = Activation('softmax', name='attention_vecd')(score_first_part)
    context_vector = keras.layers.multiply([inputs, attention_weights], name='context_vector')
    # pre_activation = concatenate([context_vector, h_t], name='attention_output')
    attention_vector = Dense(128, use_bias=False, activation='tanh',name='attention_vector')(context_vector)

    return attention_vector

def build_model(trainX,lstm_layer, drop, r_drop, l2_value, shared_layer,dense_num, n_labels):
    """
    Keras Function model
    """

    # concate_list = []
    # input_list = []
    input = Input(shape=(trainX.shape[1], trainX.shape[2]), name='input')
    attentionD = attention_dim(input)
    lstm_layer1 = LSTM(lstm_layer, activation='relu', dropout=drop,
                                       recurrent_dropout=r_drop,kernel_regularizer=regularizers.l2(l2_value),
                                       return_sequences=True)(attentionD)



    # concate_layer = keras.layers.concatenate(concate_list)


    # output_list = []
    # for i in range(0,task_num):
    concate_layer = attention_3d_block(lstm_layer1,input)
    LSTM_layer2 = LSTM(dense_num,activation='relu',dropout=drop,recurrent_dropout=r_drop)(concate_layer)
    sub = Dense(dense_num,activation='relu')(LSTM_layer2)
    out = Dense(n_labels, activation='sigmoid')(sub)
    # output_list.append(locals()['out'+str(i)])

    model = Model(inputs=input,outputs=out)
    model.compile(loss=helper_funcs.mycrossentropy, optimizer='adam', metrics=[helper_funcs.BA_metric])
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
    split_ratio = 0.8  # train & test data split ratio

    sum_bacc = 0
    sum_re = 0
    sum_pre = 0
    sum_F1 = 0
    sum_acc = 0

    file = open('results/FATHOM.txt', 'w')
    for i in [1,2,3,5,6,7,8,9]: #
        ps = 'p' + str(i)
        train_data = 'data8p/processed/wo'+ ps +'.csv'
        test_data = 'data8p/processed/'+ ps+'.csv'

        train, scaled, scaler = helper_funcs.load_dataset(train_data)
        train_X, train_y = helper_funcs.split_dataset(scaled, look_back, n_columns, n_labels)

        test, scaled_test, scaler_test = helper_funcs.load_dataset(test_data)
        test_X, test_y = helper_funcs.split_dataset(scaled, look_back, n_columns, n_labels)


        model = build_model(train_X,lstm_layer, drop, r_drop, l2_value, shared_layer, dense_num, n_labels)


        import time
        start_time = time.time()

        # fit network
        history = model.fit(train_X, train_y,
                            epochs=200,
                            batch_size=50,
                            validation_split = 0.25,
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


       #===========================================================================================#
        # write parameters & results to file


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

        sum_bacc = sum_bacc + Bacc[0]
        sum_F1 = sum_F1 + Bacc[1]
        sum_pre = sum_pre + Bacc[2]
        sum_re = sum_re + Bacc[3]
        sum_acc = sum_acc + Bacc[4]

        file.write('p' + str(i))
        file.write ('BA:'+' ' + str(Bacc[0])+' ')
        file.write ('F1:'+' '+ str(Bacc[1])+' ')
        file.write ('precision:'+' ' + str(Bacc[2])+ '\n')
        file.write('recall:' + ' ' + str(Bacc[3]) + '\n')
        file.write('accuracy:' + ' ' + str(Bacc[4]) + '\n')
        confu_matrix = confu_matrix_plot(Bacc[5], Bacc[6])
        print(confu_matrix)
        numpy.savetxt("confu_matrix/FATHOM/confusion_matrix_FATHOM-"+ ps +".csv", X=confu_matrix.astype(int), delimiter=', ',fmt ='%.0f')
        confu_matrix = confu_matrix/np.sum(confu_matrix)*1000
        print(confu_matrix)
        numpy.savetxt("confu_matrix/FATHOM/confusion_matrix_FATHOM-" + ps + "-nor.csv", X=confu_matrix.astype(float), delimiter=', ',
                      fmt='%.00f')


        file.write('training time:' + str(end_time - start_time))
        file.write('prediction time:' + str(pred_end_time - pred_time))

    file.write('avg_bacc: ' + str(sum_bacc / 8.0) + '\n')
    file.write('avg_TPR: ' + str(sum_re / 8.0) + '\n')
    file.write('avg_precision: ' + str(sum_pre / 8.0) + '\n')
    file.write('avg_F1: ' + str(sum_F1 / 8.0) + '\n')
    file.write('avg_accuracy: ' + str(sum_acc / 8.0) + '\n')

    file.write('training time:' + str(end_time - start_time))
    file.write('prediction time:' + str(pred_end_time - pred_time))

if __name__ == '__main__':
    main()
