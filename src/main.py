from setdevice import set_device
import warnings
import json
from mode import mode_def
import sys

warnings.filterwarnings('ignore')

mode = " "

print("Always load the model before inference if you have saved any")

# Set execution device
device = set_device()
print(device)
PATH = "Model_Save/Animal_classifier.pth"

# Open the json file containing the classifier categories
with open('classes.json', 'r') as f:
    class_names = json.load(f)

class_names = {value: key for key, value in class_names.items()}
print("The classes are " + str(list(class_names.keys())) )

# Mode of DL operation to perform
while mode != "exit":
    print(" ")
    print("\n What do you want to do now ? Enter a condition from the following \n 'train' to train model \n 'test' to test model \n 'save' save model \n 'load' to load model\n 'inference'  for classifier inference")
    mode = input("Enter your choice: ")
    if mode == "exit":
        sys.exit(0)
    elif mode == "train" or "test" or "save" or "load" or "inference":
        mode_def(mode, device, PATH, class_names)
    
