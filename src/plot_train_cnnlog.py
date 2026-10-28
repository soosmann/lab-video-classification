"""
Plot a Given a training log file.
"""
import csv

import matplotlib.pyplot as plt


def main(training_log):
    fig_name = training_log.split('/')[-1].replace('.log', '.png') # take last part of path and change filetype
    with open(training_log) as fin:
        reader = csv.reader(fin)
        next(reader, None)  # skip the header
        accuracies_t = []
        accuracies_v = []
        top_5_accuracies = []
        cnn_benchmark = []  # random results
        for epoch, acc, loss, val_acc, val_loss, in reader:
            accuracies_t.append(float(acc))
            accuracies_v.append(float(val_acc))
            cnn_benchmark.append(0.2)  # random

        plt.plot(accuracies_t, label='Training Accuracy')
        plt.plot(accuracies_v, label='Validation Accuracy')
        plt.plot(cnn_benchmark, label='Random Guessing Benchmark (20%)', linestyle='--')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(fig_name)
        plt.show()
        


if __name__ == '__main__':
    
    training_log = 'data/logs/inception-training-1762026182.179461.log'
    main(training_log)
