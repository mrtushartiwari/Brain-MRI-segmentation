import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import random
import cv2

from efficient_Unet import build_effienet_unet


input_shape = (256, 256, 3)
effienet_Unet_model = build_effienet_unet(input_shape)
effienet_Unet_model.load_weights("/home/tushar/Downloads/Case_2_Brain_MRI/MRIstreamlitapp/tf_effienet_Unet_brain_final-20210910T054815Z-001/tf_effienet_Unet_brain_final")
#effienet_Unet_model.summary()

def load_preprocess_image(img):
    im = Image.open(img)
    image = np.array(im)
    image = image / 256.0
    return image

def predict_segmentation_mask(image_path):
    """reads an brain MRI image 
    	Returns the segmentation mask of image
    """
    #print(image_path)
    #img = cv2.imread(image_path)
    img = Image.open(image_path)
    img = np.array(img)
    img = cv2.resize(img,(256,256))
    img = np.array(img, dtype=np.float64)
    img -= img.mean()
    img /= img.std()
    #img = np.reshape(img, (1,256,256,3) # this is the shape our model expects
    X = np.empty((1,256,256,3))
    X[0,] = img
    predict = effienet_Unet_model.predict(X)

    return predict.reshape(256,256,3)

"""
with open("base_models_pickle", "rb") as fp:   #  tf model
    base_models = pickle.load(fp)

with open("meta_models_pickle", "rb") as fp:   # tf lite model
    meta_models = pickle.load(fp)
"""


def plot_MRI_predicted_mask(original_img,predicted_mask):
    """
    Inputs: image and mask 
    Outputs: plot both image and plot side by side
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 5))
    axes[0].imshow(original_img)
    axes[0].get_xaxis().set_visible(False)
    axes[0].get_yaxis().set_visible(False)
    axes[0].set_title('Original MRI')
    axes[1].imshow(predicted_mask)
    axes[1].get_xaxis().set_visible(False)
    axes[1].get_yaxis().set_visible(False)
    axes[1].set_title('Predicted Mask')
    fig.tight_layout()
    filename = 'pair' + str(random.randint(100,1000) ) + str(random.randint(100,1000) ) + '.png'
    plt.savefig(filename)

    print('File saved successully')

    #filename = open(filename)
    print(filename)

    return  filename


def final_fun_1(image_path):
    '''  Input: Image path through the path upload method
        Returns : combined image of original and predicted mask

    '''
    # Preprocessing the inputs
    #mask = predict_segmentation_mask(image_path)
    print(image_path)
    image = load_preprocess_image(image_path)
    mask = predict_segmentation_mask(image_path)
    combined_img  = plot_MRI_predicted_mask(original_img = image,predicted_mask = mask) 
    return combined_img
