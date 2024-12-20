import os
import shutil
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from model import get_timm_model_regression
import yaml
import argparse
import logging
from datasets import RegDataset, combo_train, combo_val
from loss import *
from tqdm import tqdm


def train_one_epoch(epoch, model, train_loader, optimizer, device, loss_fn, scheduler):
    model.train()
    epoch_loss = []

    for i, data in enumerate(tqdm(train_loader)):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.unsqueeze(-1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step(epoch)
        epoch_loss.append(loss.item())

    return np.array(epoch_loss).mean()


def validate(model, val_loader, device):
    model.eval()
    total_squared_error = 0
    num_samples = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.unsqueeze(-1)
            outputs = model(images)
            squared_error = (outputs.squeeze() - labels.squeeze()) ** 2
            total_squared_error += squared_error.sum().item()
            num_samples += labels.size(0)
    mse = total_squared_error / num_samples
    rmse = np.sqrt(mse)
    return rmse


def run(config_path, device, save):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        f.close()

    logger = logging.getLogger('my_log')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(save, 'ez_log.txt'))
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    model = get_timm_model_regression(config, pretrained=True)
    model.to(device)

    num_epoches = int(config['epoch'])
    criterion = nn.MSELoss()

    base_lr = 0.001
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr)

    wmep = int(config['warmup_epoch'])

    if wmep > 0:
        def lr_lambda(epoch):
            if epoch < wmep:
                return (epoch + 1) / wmep * 0.1
            elif epoch > num_epoches / 2:
                return 0.1
            else:
                return 1.

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = None

    train_loader = DataLoader(RegDataset(config, mode='train', transform=combo_train),
                              batch_size=config['batch_size'], shuffle=True,
                              num_workers=8,
                              pin_memory=True)
    val_loader = DataLoader(RegDataset(config, mode='val', transform=combo_val),
                            batch_size=config['batch_size'],
                            shuffle=False,
                            num_workers=4,
                            drop_last=False)

    best = {
        "epoch": -1,
        'loss': np.inf,
        "rmse": np.inf,
    }

    for epoch in range(1, num_epoches + 1):
        train_loss = train_one_epoch(epoch - 1, model, train_loader, optimizer, device, criterion, scheduler)
        if epoch % config["val_interval"] == 0:
            val_rmse = validate(model, val_loader, device)
            # np.save(os.path.join(confusion_dir, f"epoch_{epoch}.npy"), confusion)
            # logger.info("\nConfusion Matrix  rows=label cols=pred\n" + str(confusion))
            if val_rmse < best['rmse'] or (val_rmse == best['rmse'] and train_loss < best['loss']):
                info = f"epoch-{epoch} loss: {train_loss:.4f} rmse: {val_rmse:.2f} | [last_best] [{best['epoch']}]-{best['loss']:.4f}/{best['rmse']:.2f}"
                best['rmse'] = val_rmse
                best['epoch'] = epoch
                best['loss'] = train_loss
                torch.save(model.state_dict(), os.path.join(save, 'best.pth'))
            else:
                info = f"epoch-{epoch} loss: {train_loss:.4f} rmse: {val_rmse:.2f}"
        else:
            info = f"epoch-{epoch} loss: {train_loss:.4f}"

        logger.info(info)

    # DO test
    # print("Start Testing")
    # model = ClsNet(config)
    # model.load_state_dict(torch.load(os.path.join(save, 'best.pth')), strict=True)
    # model.to(device)
    # model.eval()
    # val_top1_acc, confusion = validate(model, val_loader, device, classes)
    # info = f"final-test[best] top1: {val_top1_acc * 100:.2f}%"
    # logger.info(info)
    # logger.info("\nConfusion Matrix  rows=label cols=pred\n" + str(confusion))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')
    parser.add_argument('-d', "--device", default='cpu', type=str)
    parser.add_argument("--save", default='results', type=str)
    args = parser.parse_args()

    d_i = 1
    save_dir = args.save
    while os.path.exists(save_dir):
        save_dir = args.save + f"_{d_i}"
        d_i += 1

    os.makedirs(save_dir)
    shutil.copy(args.config, save_dir)

    if args.device == 'cpu':
        device = torch.device('cpu')
    else:
        assert torch.cuda.is_available()
        device = torch.device(f"cuda:{args.device}")

    run(args.config, device, save_dir)
