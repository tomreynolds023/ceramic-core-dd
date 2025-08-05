from PIL import Image
import torch
from torch import nn

import torchvision.transforms.v2 as transforms
import torchvision.models as models

class Model:
    def __init__(self):
        self.classes = ['Defect', 'Flash', 'OK']

    def load_model(self, model_weights):
        '''
        Loads the model from a saved weights file

        Parameters:
            model_weights: Path to file containing pre-trained model weights (.pth)
        '''
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = models.resnet18()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)
        if self.device == "cuda":
            model.load_state_dict(torch.load(model_weights, weights_only=True))
        else: 
            model.load_state_dict(torch.load(model_weights, 
                                             weights_only=True, 
                                             map_location=torch.device('cpu')
                                             ))
        model = model.to(self.device)
        self.model = model

    def prepare_image(self, image):
        '''
        Prepares the input image with transforms and normalisation used on the test set
        
        Parameters: 
            image: PIL image object

        Returns: 
            image: PIL image with transforms
        '''
        # These are taken from the training data 
        # Hardcoded for performance reasons
        # Should make them settable when model is set
        mean = (0.6014, 0.5435, 0.5036)
        std = (0.1191, 0.1140, 0.1121)
        transform = transforms.Compose([transforms.ToImage(), 
                                        transforms.ToDtype(torch.float32, scale=True),
                                        transforms.Normalize(mean, std)
                                        ])
        prepared_image = transform(image)
        return prepared_image.unsqueeze(0)
    
    def predict(self, image):
        '''
        Runs the prediction model on the input image

        Parameters:
            image: 224x224 PIL image

        Returns: 
            class_label (str): the predicted label
            probabilities (array): probabilities of each class [defect, flash, ok]
        '''
        image = image.to(self.device)

        with torch.no_grad():
            self.model.eval()
            output = self.model(image)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            _, pred = torch.max(output, 1)
            class_label = self.classes[pred]
            return class_label, probabilities
        
    def run(self, image):
        '''
        Runs all required functions to give a class prediction

        Parameters:
            image: 224x224 PIL image

        Returns: 
            prediction (str): the predicted label
            probabilities (array): probabilities of each class [defect, flash, ok]
        '''
        prepped_image = self.prepare_image(image)
        prediction, probabilities = self.predict(prepped_image)
        return prediction, probabilities