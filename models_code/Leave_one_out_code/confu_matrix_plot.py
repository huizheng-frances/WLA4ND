import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix


def confu_matrix_plot():
    labels = ['read', 'writeQA', 'write','type', 'rest', 'off']

    matrix = np.loadtxt('confu_matrix/processed/new_per_user.csv',delimiter=',')
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Normalise
    print(matrix)
    print(matrix[0][:])
    print(matrix[0][0])
    print(matrix[0][-1])
    print( np.tile(np.vstack(matrix.astype(np.float).sum(axis=1)),(1, len(matrix[0]))))
    matrix =  matrix / np.vstack(matrix.astype(np.float).sum(axis=1))

    print(matrix[0][0])
    print(matrix[0][-1])
    print(matrix)
    #cmn = cm.astype('float') /cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(matrix, annot=True, fmt=".2%",ax=ax,cmap='Blues')  #/np.sum(matrix) annot=True to annotate cells

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.xaxis.set_label_position('top')

    # ax.set_title('Confusion Matrix')
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.xaxis.set_ticks_position('top')

    plt.savefig('confu_matrix/processed/confusion_matrix_CRNN-per-user.png', dpi=150)

def main():
    confu_matrix_plot()

if __name__ == '__main__':
    main()