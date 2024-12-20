import timm
import torchvision
import torch.nn as nn


def get_timm_model(config, pretrained=True):
    name = config['model']
    if name == 'alexnet':
        model = torchvision.models.alexnet(pretrained=pretrained)
        model.classifier[6] = nn.Linear(4096, config['num_classes'])
    else:
        model = timm.create_model(name, pretrained=pretrained, num_classes=config['num_classes'])

    return model


def get_timm_model_regression(config, pretrained=True):
    name = config['model']
    if name == 'alexnet':
        model = torchvision.models.alexnet(pretrained=pretrained)
        model.classifier[6] = nn.Linear(4096, 1)
    else:
        model = timm.create_model(name, pretrained=pretrained, num_classes=1)

    return model


if __name__ == "__main__":
    # print(timm.list_models())
    pass
    # model = timm.create_model('fastvit_sa12', checkpoint_path='fastvit_sa12.bin')
