
from numpy import concatenate
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
import keras
import tensorflow as tf
from keras.layers.core import *
from pandas import read_csv
from keras import backend as K
import matplotlib.pyplot as plt


def load_dataset(datasource):

    # load the dataset
    dataframe = read_csv(datasource, index_col=0)
    # dataframe = dataframe.drop('label_source', axis=1)  # drop the last column

    dataframe = dataframe.fillna(method='ffill')
    dataframe = dataframe.fillna(method='bfill')
    dataframe = dataframe.fillna(0)
    # dataframe = dataframe.iloc[0:470]  # first 470 rows of dataframe

    dataset = dataframe.values
    dataset = dataset.astype('float32')

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(dataset)
    return dataset, scaled, scaler


# #take one input dataset and split it into train and test
# def split_dataset(dataset, scaled, look_back, n_columns,n_labels,ratio):
#
#     # frame as supervised learning
#     reframed = series_to_supervised(scaled, look_back, 1)
#
#     # split into train and test sets
#     values = reframed.values
#     n_train_data = int(len(dataset) * ratio)
#     train = values[:n_train_data, :]
#     test = values[n_train_data:, :]
#     # split into input and outputs
#     n_obs = look_back * n_columns
#     train_X, train_y = train[:, :n_obs], train[:, -n_labels:]  # labels are the last 6 columns
#     test_X, test_y = test[:, :n_obs], test[:, -n_labels:]
#
#     print(train_X.shape, len(train_X), train_y.shape)
#
#     # reshape input to be 3D [samples, timesteps, features]
#     train_X = train_X.reshape((train_X.shape[0], look_back, n_columns))
#     test_X = test_X.reshape((test_X.shape[0], look_back, n_columns))
#
#     return train_X, train_y, test_X, test_y

def split_dataset(scaled, look_back, n_columns,n_labels):

    # frame as supervised learning
    reframed = series_to_supervised(scaled, look_back, 1)

    # split into train and test sets
    values = reframed.values

    # split into input and outputs
    n_obs = look_back * n_columns
    data_X, data_y = values[:, :n_obs], values[:, -n_labels:]  # labels are the last 51 columns
    print(data_X.shape, len(data_X), data_y.shape)
    # reshape input to be 3D [samples, timesteps, features]
    data_X = data_X.reshape((data_X.shape[0], look_back, n_columns))

    return data_X, data_y

def tensor_similarity(t1,t2,data1,data2):
    p1 = tf.placeholder(dtype=t1.dtype, shape=t1.shape)
    p2 = tf.placeholder(dtype=t2.dtype, shape=t2.shape)

    # s = tf.losses.cosine_distance(tf.nn.l2_normalize(p1, 0), tf.nn.l2_normalize(p2, 0), axis=1,reduction=Reduction.Mean)
    # ss = keras.layers.dot([p1, p2], axes=2, normalize=True)

    # Method1: using keras dot
    ss = keras.layers.dot([p1, p2], axes=2, normalize=True)

    # Method2: using TF/Keras backend
    square_sum1 = K.sum(K.square(p1), axis=2, keepdims=True)
    norm1 = K.sqrt(K.maximum(square_sum1, K.epsilon()))
    square_sum2 = K.sum(K.square(p2), axis=2, keepdims=True)
    norm2 = K.sqrt(K.maximum(square_sum2, K.epsilon()))

    num = K.batch_dot(p1, K.permute_dimensions(p2, (0, 2, 1)))
    den = (norm1 * K.permute_dimensions(norm2, (0, 2, 1)))
    cos_similarity = num / den

    with tf.Session().as_default() as sess:
        similarity1 = sess.run(ss, feed_dict={p1: data1, p2: data2})
        similarity2 = sess.run(cos_similarity, feed_dict={p1: data1, p2: data2})

    similarity1 = np.average(similarity1)
    similarity2 = np.average(similarity2)

    return similarity1,similarity2

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

    # plt.show()

def evaluation(test_X, test_y, y_pred, timestamps, n_columns, n_labels, scaler):

    test_X = test_X.reshape((test_X.shape[0], timestamps * n_columns))
    # invert scaling for forecast
    y_predict = concatenate((test_X[:, -n_columns:-n_labels], y_pred), axis=1)
    y_predict = scaler.inverse_transform(y_predict)
    y_predict = y_predict[:, -n_labels:]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), n_labels))
    y_true = concatenate((test_X[:, -n_columns:-n_labels], test_y), axis=1)
    y_true = scaler.inverse_transform(y_true)
    y_true = y_true[:, -n_labels:]

    # Round labels of the array to the nearest integer.
    y_predict = np.rint(y_predict)
    y_true = np.rint(y_true)

    y_predict[y_predict <= 0] = 0
    y_true[y_true <= 0] = 0

    # Bacc = BalanceAcc(y_predict,y_true)

    true_multiclass,pred_multiclass = confu_matrix(y_true, y_predict)
    Bacc = []
    BA = tf.keras.backend.get_value(BA_metric(y_true,y_predict))
    F1 = tf.keras.backend.get_value(f1_score(y_true,y_predict))
    pre = tf.keras.backend.get_value(precision(y_true,y_predict))
    re = tf.keras.backend.get_value(recall(y_true, y_pred))
    acc = tf.keras.backend.get_value(accuracy(y_true, y_pred))

    Bacc.append(BA)
    Bacc.append(F1)
    Bacc.append(pre)
    Bacc.append(re)
    Bacc.append(acc)
    Bacc.append(true_multiclass)
    Bacc.append(pred_multiclass)

    return Bacc



# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return K.cast(precision, "float32")

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return K.cast(recall, "float32")

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def accuracy(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    total = possible_positives + possible_negatives + K.epsilon()
    accuracy = (true_negatives + true_positives) / total
    return accuracy

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    f1 = (2 * p * r) / (p + r + K.epsilon())
    f1_score = K.cast(f1, "float32")
    return f1_score

def BA_metric(y_true, y_pred):
    r = recall(y_true, y_pred)
    s = specificity(y_pred, y_true)
    BA = (s + r) / (2+ K.epsilon())
    return BA



def BalanceAcc(y_pred,y_true):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    total = 0
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

    F1 = (2*precision*sensitivity) / (precision+sensitivity)
    # naive accuracy
    accuracy = float(tn + tp) / total

    # # Balanced accuracy is a more fair replacement for the naive accuracy:
    balanced_accuracy = (sensitivity + specificity) / 2.;
    #
    print("-" * 10);
    print('Accuracy*:         %.2f' % accuracy);
    print('Sensitivity (TPR): %.2f' % sensitivity);
    print('Specificity (TNR): %.2f' % specificity);
    print('Balanced accuracy: %.2f' % balanced_accuracy);
    print('Precision: %.2f' % precision);
    print('F1: %.2f' % F1);
    print("-" * 10);

    return accuracy, sensitivity, specificity, balanced_accuracy,tp,fn,fp,tn,precision,F1

def mycrossentropy(y_true, y_pred, e=0.3):
    nb_classes = 51
    loss1 = K.categorical_crossentropy(y_true, y_pred)
    loss2 = K.categorical_crossentropy(K.ones_like(y_pred)/nb_classes, y_pred)
    return (1-e)*loss1 + e*loss2


def get_activations(model, inputs, print_shape_only=False, layer_name=None):
    # Documentation is available online on Github at the address below.
    # From: https://github.com/philipperemy/keras-visualize-activations
    print('----- activations -----')
    activations = []
    inp = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations

def get_activationsD(model, inputs, print_shape_only=False, layer_name=None):
    # Documentation is available online on Github at the address below.
    # From: https://github.com/philipperemy/keras-visualize-activations
    print('----- activations -----')
    activations = []
    inp = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations