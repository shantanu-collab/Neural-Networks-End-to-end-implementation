import keras

from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D
from pathlib import Path


(train_x,train_y),(test_x,test_y) = cifar10.load_data()

labels = {"airplane":0,"automobile":1,"bird":2,"cat":3,"deer":4,"dog":5,"frog":6,"horse":7,"ship":8,"truck":9}

print(train_y.shape)

#for i in range (0,1000):
#    plt.imshow(train_x[i])
#    plt.title(train_y[i][0])
#    plt.show()

train_x = train_x.astype("float32")
test_x = test_x.astype("float32")
train_x = train_x/255
test_x = test_x/255

train_y = to_categorical(train_y,10)
test_y =to_categorical(test_y,10)

model = Sequential()
model.add(Conv2D(32,(3,3),padding="same",activation="relu",input_shape = (32,32,3)))
model.add(Conv2D(32,(3,3),activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64,(3,3),padding="same",activation="relu"))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation = "relu",))
model.add(Dropout(0.5))
model.add(Dense(10,activation="softmax"))

model.compile(
    loss = "categorical_crossentropy",
    optimizer= "adam",
    metrics= ["accuracy"]


)

model.fit(train_x,train_y,batch_size=32,epochs=30,validation_data=(test_x,test_y),shuffle=True)

#save the neural network
model_str = model.to_json()
f = Path("model_str.json")
f.write_text(model_str)

#save neural netework's weight
model.save_weights("model_weight.h5")