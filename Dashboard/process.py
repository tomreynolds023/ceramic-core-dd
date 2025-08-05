from PIL import Image, ImageDraw
import base64
from io import BytesIO
from tqdm.contrib.itertools import product
#from multiprocessing import Manager, Queue
from collections import deque


class Process:
    '''
    Class to contain all image processing before and after model evaluation
    '''
    def __init__(self, model):
        self.output_image = None
        self.model = model
        self.test_image_size = 224

    def evaluate(self, q: deque):
        '''
        Splits the input image into a grid, and sends images one by one for evaluation by the model

        Returns:
            defect_coords (list): List of all coordinates containing defects
            defect_probs (list): List of probabilities for every defect
            And so on for flash and ok
        '''
        w, h = self.output_image.size
        d = self.test_image_size
        defect_coords = []
        defect_probs = []
        flash_coords = []
        flash_probs = []
        ok_coords = []
        ok_probs = []

        # this is a bit of a workaround to make the progress bar work
        h_range = range(0, h-h%d, d)
        w_range = range(0, w-w%d, d)
        grid = product(h_range, w_range)
        grid_it = 0
        grid_length = len(h_range) * len(w_range)

        # loops through, creates each grid box and passes it to the model
        for i, j in grid:
            box = (j, i, j+d, i+d)
            cur_image = self.output_image.crop(box)
            cur_prediction, cur_probabilities  = self.model.run(cur_image)
            if (cur_prediction == 'Defect'):
                defect_coords.append((j, i))
                defect_probs.append(cur_probabilities.cpu().numpy()[0][0])
            elif (cur_prediction == 'Flash'):
                flash_coords.append((j, i))
                flash_probs.append(cur_probabilities.cpu().numpy()[0][1])
            else:
                ok_coords.append((j, i))
                ok_probs.append(cur_probabilities.cpu().numpy()[0][2])

            q.append(grid_it / grid_length)
            grid_it += 1

        return defect_coords, defect_probs, flash_coords, flash_probs, ok_coords, ok_probs

    def resize_image(self, image, max_w):
        '''
        Resizes the input image for quicker display in the UI

        Parameters:
            image: PIL image
            max_w (int): maximum image width after resize
        
        Returns:
            image: resized PIL image
        '''
        w, h = image.size
        self.scale_factor = max_w / w
        self.new_h = int(h * self.scale_factor)
        image = image.resize((max_w, self.new_h), Image.LANCZOS)
        self.new_boxsize = self.test_image_size * self.scale_factor
        return image
        
    def run(self, image, q: deque):
        '''
        Runs the suite of image processing 

        Parameters: 
            image: Base64 string representation of the image from file upload in ui
        '''
        image = base64.b64decode(image)
        image = BytesIO(image)
        self.output_image = Image.open(image)
        self.defect_coords, self.defect_probs, self.flash_coords, self.flash_probs, self.ok_coords, self.ok_probs = self.evaluate(q)
        # resize the image
        self.output_image = self.resize_image(self.output_image, 1440)

        print('Model run!')