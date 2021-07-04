
Platform: Pycharm 

Library: Keras + tensorflow (1.1)
         os; warnings; glob; numpy; pandas; sklearn; matplotlib; 


==================================
FATHOM.py
==================================

Parameters:
----------- 

1. TIME_STEPS: sliding window length 
2. task_num : number of users
3. dense_att: number of nuerons for dense layer in 'attention_dim' function
4. lstm_layer: network parameters in build_model function
   drop: network parameters in build_model function
   r_drop: network parameters in build_model function
   l2_value: network parameters in build_model function
   shared_layer: network parameters in build_model function
   dense_num: network parameters in build_model function

5. look_back = 20  # number of previous timestamp used for training (length of sliding window)
   n_columns = 276  # total columns (including both features and labels)
   n_labels = 51  # number of labels
   split_ratio = 0.8  # 80% for trainning, 20% for testing 


file path:
----------

file_list = glob.glob('data_csv/train/*.csv')  # 'train' is the file folder


model fit:
----------

parameters can be adjusted: epochs, batch_size, 
                            validation_split (no harm if using 0.25, this means 25% of the training data is used for validation), 
                            patience =20 (if 'val_loss (validation loss)' not decrease for 20 epoches, the training will stop)

history = model.fit(trainX_list, trainy_list,
                        epochs=150, 
                        batch_size=60,
                        validation_split = 0.25,
                        verbose=2,
                        shuffle=False,
                        callbacks=[
                            keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=2,
                                                          mode='min')]
                        )


prediction:
-----------
#### predicted results of 8 users #######
y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6,y_pred7,y_pred8 = model.predict(testX_list) 


save results:
-------------
########  this file saves the predicted results of each user and the average values ##########
file = open('FATHOM_8users.txt', 'w')





Plots of attention

==================================
att_plot_input_dim.py    in 'attention_plot' folder
==================================

1. testX_list[0][0:20,:,:] ------- first 20 timestamps of test data of the first user 
2. file_list = glob.glob('../data_csv/train/testtest/user1.csv')  -------- plot the attention weights of one user 
