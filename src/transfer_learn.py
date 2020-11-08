from torch import nn
from torchvision import models


def transfer_learn():
    model = models.resnet152(pretrained=True)

    for parms in model.parameters():
        parms.requires_grad = False

    classifier = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Linear(512,64),
    nn.ReLU(),
    nn.Linear(64,10),
    nn.LogSoftmax()
    )
    model.fc = classifier

    # Used for debugging seeing unfreezed model parameters
    # for param in model.parameters():
    #     if param.requires_grad:
    #         print(param.shape)

    return model