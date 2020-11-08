import torch
from torch.autograd import Variable


def test_model(model, dataloaders, device):
  model.eval()
  accuracy = 0
  
  model.to(device)
    
  for images, labels in dataloaders['test']:
    images = Variable(images)
    labels = Variable(labels)
    images, labels = images.to(device), labels.to(device)
      
    output = model.forward(images)
    ps = torch.exp(output)
    equality = (labels.data == ps.max(1)[1])
    accuracy += equality.type_as(torch.FloatTensor()).mean()

    del images, labels, output  #Freeing GPU memory and cache for preventing VRAM overflow 
    torch.cuda.empty_cache()
      
    print("Testing Accuracy: {:.3f}".format(accuracy/len(dataloaders['val'])))