"""
Given a training log file, plot something.
"""
import csv
import os

import matplotlib.pyplot as plt


def main(training_log):
    fig_name = training_log.split('/')[-1].replace('.log', '.png')  # take last part of path and change filetype
    param_string = fig_name.split('-')[1] 
    class_limit, seq_length = param_string.split('_')
    with open(training_log) as fin:
        reader = csv.reader(fin)
        next(reader, None)  # skip the header

        epochs = []
        accuracies_t = []
        accuracies_v = []
        top_accuracies_t = []
        top_accuracies_v = []
        losses_t = []
        losses_v = []
        #epoch,accuracy,loss,top_k_categorical_accuracy,val_accuracy,val_loss,val_top_k_categorical_accuracy
        if len(next(reader)) == 7:
            for epoch, acc, loss, topk_acc, val_acc, val_loss, val_topk_acc in reader:
                epochs.append(int(epoch))
                accuracies_t.append(float(acc))
                accuracies_v.append(float(val_acc))
                top_accuracies_t.append(float(topk_acc))
                top_accuracies_v.append(float(val_topk_acc))
                losses_t.append(float(loss))
                losses_v.append(float(val_loss))
        else:
            for epoch, acc, loss, val_acc, val_loss in reader:
                epochs.append(int(epoch))
                accuracies_t.append(float(acc))
                accuracies_v.append(float(val_acc))
                losses_t.append(float(loss))
                losses_v.append(float(val_loss))

        fig, (ax_acc, ax_loss) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        ax_acc.plot(epochs, accuracies_t, label='train accuracy')
        ax_acc.plot(epochs, accuracies_v, label='val accuracy')
        if top_accuracies_t and top_accuracies_v:
            ax_acc.plot(epochs, top_accuracies_t, label='train top-5 accuracy', linestyle='--')
            ax_acc.plot(epochs, top_accuracies_v, label='val top-5 accuracy', linestyle='--')
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.set_title(f'Accuracy per epoch (Classes: {class_limit}, Seq Length: {seq_length})')
        ax_acc.grid(True)
        ax_acc.legend()

        ax_loss.plot(epochs, losses_t, label='train loss')
        ax_loss.plot(epochs, losses_v, label='val loss')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_title(f'Loss per epoch (Classes: {class_limit}, Seq Length: {seq_length})')
        ax_loss.grid(True)
        ax_loss.legend()

        plt.tight_layout()
        plt.savefig(fig_name)
        plt.show()


if __name__ == '__main__':
    # for log_file in os.listdir('data/logs/'):
    #     if "lstm" in log_file and log_file.endswith('.log'): # to get only lstm logs and exclude other files
    #         training_log = os.path.join('data/logs/', log_file)
    #         main(training_log)
    training_log = 'data/logs/lstm-5_20-training-1762189387.470415.log'
    main(training_log)
