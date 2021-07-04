import pandas as pd
from pandas import read_csv


for i in range(8):
    if i >=3:
        j = read_csv('data8P/new_p' + str(i + 2) + '.csv', index_col=0)
        j = j.iloc[0:1295]  # first 470 rows of dataframe
        j.to_csv('data8P/processed/file' + str(i) + '.csv')
    else:
        j = read_csv('data8P/new_p'+str(i+1)+'.csv', index_col=0)
        j = j.iloc[0:1295]  # first 470 rows of dataframe
        j.to_csv('data8P/processed/file'+str(i)+'.csv')


frames = []
for i in range(8):
    file = read_csv('data8P/processed/file' + str(i) + '.csv', index_col=0)
    frames.append(file)

result = pd.concat(frames)
result.to_csv('data8P/concate.csv')


frames = []
for i in range(1,3):
    file = read_csv('data8P/new_p' + str(i) + '.csv', index_col=0)
    frames.append(file)

for i in range(5,10):
    file = read_csv('data8P/new_p' + str(i) + '.csv', index_col=0)
    frames.append(file)

result = pd.concat(frames)
result.to_csv('data8P/concate_all.csv')