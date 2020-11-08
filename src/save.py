import torch

def save_model(PATH, model, optimizer_ft):
    num_epochs = 20
    model.epochs = num_epochs
    checkpoint = {
                'epoch': model.epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_ft.state_dict(),
                'category_index' :  model.category_index
                }
    torch.save(checkpoint, PATH)