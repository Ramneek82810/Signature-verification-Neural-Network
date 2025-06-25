#evaluation before sending to Neural Network
import cv2
import numpy as np
from PIL import Image
import io

#single evaluation 

def preprocess_single_image(image_file):
    image = Image.open(image_file).convert('L') 
    image = image.resize((50, 50))           
    img_array = np.array(image)                  
    img_array = img_array.flatten() / 255.0     

    return img_array.reshape(1, -1)   