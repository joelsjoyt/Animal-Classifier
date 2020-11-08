import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
from decimal import *

def inference(loaded_model, class_names, classifications, device):

    def preprocess_data(image):

        preprocess = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        image = preprocess(image)
        image = torch.unsqueeze(image, 0) 
        image = image.to(device)        
        return image

    def top_five_predections(prob):
       
        topk = torch.topk(prob, 5)
        topk
        values, indeces = torch.topk(prob, 5) 
        np_values = values.squeeze().detach().cpu().numpy()
        np_indeces = indeces.squeeze().cpu().numpy()
        return np_values,np_indeces

    def prediction(img, loaded_model):
        
        loaded_model.eval()
        img = preprocess_data(img)
        probability = loaded_model(img) 
        values, indeces = top_five_predections(probability)
        pred_val, pred_class = torch.max(probability, 1)
        pred_class = pred_class.item()
        pred_val = pred_val.item()
        
        return values, indeces, pred_val, pred_class

    def show_predection(img, filename, classifications): 
        
        #Getting top 5 and the real predicted output
        pred_values, pred_indeces, pred_val, pred_class  = prediction(img, loaded_model)

        pred_values = list(pred_values)
        pred_indeces = list(pred_indeces)
                
        print(" ")
        print("The input image corresponds to : " + str(classifications[pred_class]) +" with a probability of " + str(np.exp(pred_val))) 
        print(" ")
        print("Top 5 prediced classes are : \n")
        for i,x in  zip(pred_indeces, range(len(pred_values))):
            print("The input image corresponds to : " + str(classifications[i]) +" with a probability of " + " "+str(float("{:.4f}".format(np.exp(pred_values[x]))) ))
            

    
    #Getting classifications
    names = list(class_names.keys())
    
   
    for values,elem in zip(classifications.keys(), names):
        classifications[values] = str(elem) 
    

    #Opening an image
    image = Image.open('Inference/test1.jpg')
    filename = str(Path('Inference/test1.jpg').stem) 

    #Predicting classes
    show_predection(image, filename, classifications)



