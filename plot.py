import glob
import os.path

import matplotlib.pyplot as plt
import re

colors = ["blue", 'red', "yellow"]


def parse_log_and_plot(roots, tags, save_path):
    all_epochs = []
    all_losses = []
    all_accuracy = []

    for root in roots:
        log_file_path = os.path.join(root, 'ez_log.txt')
        epochs = []
        losses = []
        accuracy = []
        with open(log_file_path, 'r') as file:
            lines = file.readlines()

        for line in lines[2:]:
            parts = re.findall(r'epoch-(\d+) loss: (\d+\.\d+) top1: (\d+\.\d+)%', line)
            if parts:
                epoch, loss, acc = map(float, parts[0])
                epochs.append(epoch)
                losses.append(loss)
                accuracy.append(acc)
        all_epochs.append(epochs)
        all_losses.append(losses)
        all_accuracy.append(accuracy)

    for epochs, losses, tag, color in zip(all_epochs, all_losses, tags, colors):
        plt.plot(epochs, losses, label=tag, color=color)

    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(save_path.replace('.png', '_loss.png'))
    plt.close()

    for epochs, accuracy, tag, color in zip(all_epochs, all_accuracy, tags, colors):
        plt.plot(epochs, accuracy, label=tag, color=color)

    plt.title('ACC over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()

    plt.savefig(save_path.replace('.png', '_acc.png'))
    plt.close()


def parse_log_and_plot2(roots, tags, save_path):
    all_epochs = []
    all_losses = []
    all_rmse = []

    for root in roots:
        log_file_path = os.path.join(root, 'ez_log.txt')
        epochs = []
        losses = []
        rmse = []
        with open(log_file_path, 'r') as file:
            lines = file.readlines()

        for line in lines[2:]:
            parts = re.findall(r'epoch-(\d+) loss: (\d+\.\d+) rmse: (\d+\.\d+)', line)
            if parts:
                epoch, loss, acc = map(float, parts[0])
                epochs.append(epoch)
                losses.append(loss)
                rmse.append(acc)
        all_epochs.append(epochs)
        all_losses.append(losses)
        all_rmse.append(rmse)

    for epochs, losses, tag, color in zip(all_epochs, all_losses, tags, colors):
        plt.plot(epochs, losses, label=tag, color=color)

    plt.title('loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()

    plt.savefig(save_path.replace('.png', '_loss.png'))
    plt.close()

    for epochs, accuracy, tag, color in zip(all_epochs, all_rmse, tags, colors):
        plt.plot(epochs, accuracy, label=tag, color=color)

    plt.title('RMSE over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()

    plt.savefig(save_path.replace('.png', '_rmse.png'))
    plt.close()


if __name__ == "__main__":

    # roots = []
    # tags = []
    # for i in [4, 5, 6]:
    #     root = f"/workspace/code/classification_1207/results/exp{i}_*"
    #     root = glob.glob(root)[0]
    #     roots.append(root)
    #     tag = os.path.basename(root).split("_")[-1]
    #     tags.append(tag)
    # parse_log_and_plot(roots, tags, "/workspace/code/classification_1207/results/pose.png")

    roots = []
    tags = []
    for i in [9, 10, 11]:
        root = f"/workspace/code/classification_1207/results/exp{i}_*"
        root = glob.glob(root)[0]
        roots.append(root)
        tag = os.path.basename(root).split("_")[-1]
        tags.append(tag)
    parse_log_and_plot2(roots, tags, "/workspace/code/classification_1207/results/depth.png")
