import pandas as pd
from pandas import read_csv

'''
for i in range(8):
    j = read_csv('data_sample/p'+str(i+1)+'.csv', index_col=0)
    j = j.iloc[0:470]  # first 470 rows of dataframe
    j.to_csv('data_sample/processed/file'+str(i)+'.csv')


frames = []
for i in range(8):
    file = read_csv('data_sample/processed/file' + str(i) + '.csv', index_col=0)
    frames.append(file)

result = pd.concat(frames)
result.to_csv('data_sample/concate.csv')
'''


for k in [1,2,3,5,6,7,8,9]:
    frames = []
    for i in range(1,4): #1,4
        if i == k:
            continue
        file = read_csv('data8p/processed/p' + str(i) + '.csv', index_col=0)
        frames.append(file)
    #or i in #ange(5,10): #5,10
    for i in range(5,10):
        if i == k:
            continue
        file = read_csv('data8p/processed/p' + str(i) + '.csv', index_col=0)
        frames.append(file)

    result = pd.concat(frames)
    result.to_csv('data8p/processed/wop' + str(k) +'.csv')