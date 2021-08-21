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
import PIL


class_labels = ["Plane","Car","Bird","Cat","Deer","Dog","Frog","Horse","Boat","Truck"]
f = Path("model_str.json")
model_str = f.read_text()
model = model_from_json(model_str)
model.load_weights("model_weight.h5")
img = image.load_img("airplane-flight.jpg",target_size=(32,32))
image_to_test = image.img_to_array(img)/255
list_of_images = np.expand_dims(image_to_test,axis=0)
results = model.predict(list_of_images)
max=0

for i in range (0,10):
    if (results[0][i]>max):
        max = results[0][i]
        ind = i
print(class_labels[ind])
print(results)







