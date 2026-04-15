from PIL import Image, ImageOps
import os

def process_image(input_path, output_path):
    #open image
    with Image.open(input_path) as img:        
        #convert img to grayscale
        processed_img = ImageOps.grayscale(img)
        #save processed img
        processed_img.save(output_path)