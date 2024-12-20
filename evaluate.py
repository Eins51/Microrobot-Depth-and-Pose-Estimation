import glob
import os.path
import shutil

import torch
import numpy as np
import yaml

from datasets import combo_val, RegDatasetWithPath, BaseDateset
from torch.utils.data import DataLoader


def evaluate(model, val_loader, device, writer=None):
    model.eval()
    total_squared_error = 0
    num_samples = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, labels, paths = data
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.unsqueeze(-1)
            outputs = model(images)
            
            squared_error = (outputs.squeeze() - labels.squeeze()) ** 2
            if writer is not None:
                for path, gt, pred in zip(paths, labels.cpu().numpy()[:, 0], outputs.cpu().numpy()[:, 0]):
                    diff = gt - pred
                    writer.write(f"{path} {gt:.4f} {pred:.4f} {diff:.4f}\n")
            total_squared_error += squared_error.sum().item()
            num_samples += labels.size(0)

    mse = total_squared_error / num_samples

    rmse = np.sqrt(mse)
    writer.close()
    return rmse

def validate(model, val_loader, device, classes):
    model.eval()
    total = 0
    correct = 0
    confusion_label_pred = np.zeros([len(classes), len(classes)], dtype=np.int32)
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # print(outputs.shape, outputs[0])
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            p_np = predicted.cpu().numpy()
            l_np = labels.cpu().numpy()
            for pi, li in zip(p_np, l_np):
                confusion_label_pred[li, pi] += 1

    top1_acc = correct / total

    confusion_acc = confusion_label_pred / np.sum(confusion_label_pred, axis=1, keepdims=True)
    confusion_acc = confusion_acc * 100
    confusion_acc = confusion_acc.round(2)

    return top1_acc, confusion_acc

def work(ckpt_root, gpu=0):
    device = torch.device(f'cuda:{gpu}')
    config_path = glob.glob(os.path.join(ckpt_root, '*.yaml'))[0]
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        f.close()
    from model import get_timm_model_regression
    model = get_timm_model_regression(config, pretrained=False)
    ckpt_path = os.path.join(ckpt_root, 'best.pth')
    assert os.path.exists(ckpt_path)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

    model = model.to(device)
    val_loader = DataLoader(RegDatasetWithPath(config, mode='val', transform=combo_val),
                            batch_size=config['batch_size'],
                            shuffle=False,
                            num_workers=4,
                            drop_last=False)

    writer = open(os.path.join(ckpt_root, 'bad_cases.txt'), 'w+')
    rmse = evaluate(model, val_loader, device, writer)
    print(rmse)


def work2(ckpt_root, gpu=0):
    device = torch.device(f'cuda:{gpu}')
    config_path = glob.glob(os.path.join(ckpt_root, '*.yaml'))[0]
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        f.close()
    model_name = config['model']
    from model import get_timm_model
    model = get_timm_model(config, pretrained=False)
    ckpt_path = os.path.join(ckpt_root, 'best.pth')
    assert os.path.exists(ckpt_path)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

    model = model.to(device)
    val_loader = DataLoader(BaseDateset(config, mode='val', transform=combo_val),
                            batch_size=config['batch_size'],
                            shuffle=False,
                            num_workers=4,
                            drop_last=False)

    classes = [i for i in range(config['num_classes'])]
    top1_acc, confusion_acc = validate(model, val_loader, device, classes)
    print(top1_acc)
    print(confusion_acc)
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rcParams['axes.unicode_minus'] = False

    sns.heatmap(confusion_acc, annot=False, cmap='Blues')

    plt.title(f'Confusion Matrix {model_name}')
    plt.xlabel('prediction')
    plt.ylabel('ground truth')

    plt.show()
    plt.savefig(os.path.join("/workspace/code/classification_1207/export_results7", f"cm_{model_name}.png"))
    plt.close()


def stat_bad_cases(root):
    path = os.path.join(root, 'bad_cases.txt')
    with open(path, 'r') as f:
        lines = f.readlines()
        f.close()

    import matplotlib.pyplot as plt

    all_gt = []
    all_pred = []
    all_diff = []
    for line in lines:
        img_path, gt, pred, abs_diff = line.strip().split(" ")
        all_diff.append(float(abs_diff))
        all_gt.append(float(gt))
        all_pred.append(float(pred))

    all_abs_diff = [abs(x) for x in all_diff]

    plt.title("Depth Regression")
    plt.scatter(np.array(all_gt), np.array(all_pred), s=3)
    plt.plot([min(all_gt), max(all_gt)], [min(all_pred), max(all_pred)], 'r')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.show()
    plt.savefig(os.path.join(root, 'bad_cases_scatter.png'))
    plt.close()

    plt.title("Depth Regression")
    plt.scatter(np.array(all_gt), np.array(all_diff), s=3)
    plt.axhline(y=0, color='r') 
    plt.xlabel('True Values')
    plt.ylabel('Residuals')
    plt.show()
    plt.savefig(os.path.join(root, 'bad_cases_scatter2.png'))
    plt.close()

    plt.title("Depth Regression")
    plt.hist(np.array(all_abs_diff), bins=20)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.show()
    plt.savefig(os.path.join(root, 'bad_cases_hist.png'))
    plt.close()


if __name__ == "__main__":
    # for i in [9, 10, 11]:
    #     a = glob.glob(f"/workspace/code/classification_1207/results/exp{i}_*")[0]
    #     work(a)

    # save_root = "export_results5"
    #
    # for i in [9, 10, 11]:
    #     a = glob.glob(f"/workspace/code/classification_1207/results/exp{i}_*")[0]
    #     save_sub = os.path.join(save_root, os.path.basename(a))
    #     os.makedirs(save_sub, exist_ok=True)
    #     bad_cases = glob.glob(os.path.join(a, "bad_cases_*"))
    #     for p in bad_cases:
    #         shutil.copy(p, save_sub)

    for i in [4, 5, 6]:
        a = glob.glob(f"/workspace/code/classification_1207/results/exp{i}_*")[0]
        work2(a)
