import keras
import numpy as np
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D
from pathlib import Path
from keras.models import model_from_json
from keras_preprocessing import  image
from keras.applications import vgg16

model = vgg16.VGG16()

img = image.load_img("13mandhana.jpg",target_size=(224,224))
image_to_test = image.img_to_array(img)
list_of_images = np.expand_dims(image_to_test,axis=0)

list_of_images = vgg16.preprocess_input(list_of_images)

predictions = model.predict(list_of_images)

predicted_classes = vgg16.decode_predictions(predictions,top=9)
for id,name,likeli in predicted_classes[0]:
    print("prediction : {} - {:2f}".format(name,likeli))


