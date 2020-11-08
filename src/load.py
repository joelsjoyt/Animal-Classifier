import torch
from torch import nn
from torchvision import models
from os import path

def load_checkpoint(filepath, device):

  if path.exists(filepath):
    if str(device) == "cuda:0":
      print("GPU")
      checkpoint = torch.load(filepath)
    else:
      print("CPU")
      checkpoint = torch.load(filepath, map_location=str(device))

    model = models.resnet152()
    for parms in model.parameters():
      parms.requires_grad=False
    classifier = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Linear(512,64),
    nn.ReLU(),
    nn.Linear(64,10),
    nn.LogSoftmax()
    )
    model.fc = classifier

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    return model, checkpoint['category_index']

