from os import path
from dataloader import data_loader
from transfer_learn import transfer_learn
from train import train
from test import test_model
from save import save_model
from load import load_checkpoint
from inference import inference
import config



def mode_def(mode, device, PATH, class_names):

    #Loading transformed dataset
    image_datasets, dataloaders, dataset_sizes = data_loader()
    
    if mode == 'train' or mode == 'test':

        if config.loaded_model == None:
            # Loading pretrained Resnet151 model
            model = transfer_learn()

        # If a loaded model exists
        elif config.model_ft_mode == None:
            model = config.loaded_model    

        if mode == 'train':
            config.model_ft_mode, config.optimizer_ft_mode = train(dataloaders, dataset_sizes, model, device)
            print("Training completed.... Do you want to save the model now. Y to save N to skip")
            choice = str(input("Enter your Choice: Y or N\t "))

            if choice == 'Y' or 'y':
                #Mapping dataset classes from our dataset to model
                model.category_index = image_datasets['train'].class_to_idx 
                # Saving Model
                save_model(PATH, config.model_ft_mode, config.optimizer_ft_mode)
                config.flag = 1
                print("Model Saved")


        elif mode == 'test':

            if config.model_ft_mode:
                test_model(config.model_ft_mode, dataloaders, device)
                print("Testing is complete")

            elif config.loaded_model:
                test_model(config.loaded_model, dataloaders, device)
                print("Testing is complete")


    elif mode == 'save':
        
        if config.flag == 1:
            print("The model is already saved... Do you want to save again?")
            choice = str(input("Enter your choice Y to save again or N to skip"))
            if choice == 'Y' or 'y':
                model.category_index = image_datasets['train'].class_to_idx 
                save_model(PATH, config.model_ft_mode, config.optimizer_ft_mode)
                config.flag = 1
                print("Model Saved")

        elif path.exists(PATH):
            print("A saved model already exists") 

        else:            
            model.category_index = image_datasets['train'].class_to_idx
            save_model(PATH, config.model_ft_mode, config.optimizer_ft_mode)
            config.flag = 1  
            print("Model Saved") 

    elif mode == 'load':
        #Loading the saved model
            config.loaded_model, config.category_index = load_checkpoint(PATH, device)
            if config.loaded_model == None and  config.category_index == None:
                print("Model not found, so cannot be loaded")
            else:
                #Fetching classification 
                config.classifications = { values : key for key,values in config.category_index.items()} #swapping key and values for the classes
                print("Loading model is complete")

    elif mode == 'inference':
        errorinf = {"loaded_model":"None","class_names":"None","config.classifications":"None"}
        for error in errorinf.keys():
            if errorinf[error] == None:
                print(error + " Not Declared properly")

        if ((config.loaded_model != None) and (class_names != None) and (config.classifications != None)):
            inference(config.loaded_model, class_names, config.classifications, device)
            

    else:
        print("Wrong Choice")
