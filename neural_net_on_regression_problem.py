import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import *
from keras.models import load_model

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

SS = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")
df = df.select_dtypes(exclude="object")
test = test.select_dtypes(exclude="object")
df["LotFrontage"].replace(np.nan,df["LotFrontage"].mean(),inplace=True)
test["LotFrontage"].replace(np.nan,test["LotFrontage"].mean(),inplace=True)

df["MasVnrArea"].replace(np.nan,df["MasVnrArea"].mean(),inplace=True)
df["GarageYrBlt"].replace(np.nan,df["GarageYrBlt"].mean(),inplace=True)
test["MasVnrArea"].replace(np.nan,test["MasVnrArea"].mean(),inplace=True)
test["GarageYrBlt"].replace(np.nan,test["GarageYrBlt"].mean(),inplace=True)
test["GarageArea"].replace(np.nan,test["GarageYrBlt"].mean(),inplace=True)
test["GarageCars"].replace(np.nan,test["GarageYrBlt"].mean(),inplace=True)
test["BsmtHalfBath"].replace(np.nan,test["GarageYrBlt"].mean(),inplace=True)
test["BsmtFullBath"].replace(np.nan,test["GarageYrBlt"].mean(),inplace=True)
test["TotalBsmtSF"].replace(np.nan,test["GarageYrBlt"].mean(),inplace=True)
test["BsmtUnfSF"].replace(np.nan,test["GarageYrBlt"].mean(),inplace=True)
test["BsmtFinSF2"].replace(np.nan,test["GarageYrBlt"].mean(),inplace=True)
test["BsmtFinSF1"].replace(np.nan,test["GarageYrBlt"].mean(),inplace=True)

train_y = df["SalePrice"]
train_x = df.drop(columns ="SalePrice",axis=1)

scale =  MinMaxScaler(feature_range=(0,1))
scaled_train = scale.fit_transform(train_x)
scaled_test = scale.fit_transform(test)

scaled_train = pd.DataFrame(scaled_train,columns=train_x.columns)
scaled_test = pd.DataFrame(scaled_test,columns=test.columns)

model = Sequential()
model.add(Dense(50,input_dim=37,activation="relu"))
model.add(Dense(100,activation="relu"))
model.add(Dense(50,activation="relu"))
model.add(Dense(50,activation="relu"))
model.add(Dense(50,activation="relu"))
model.add(Dense(50,activation="relu"))
model.add(Dense(50,activation="relu"))
model.add(Dense(50,activation="relu"))
model.add(Dense(50,activation="relu"))
model.add(Dense(50,activation="relu"))
model.add(Dense(50,activation="relu"))
model.add(Dense(50,activation="relu"))
model.add(Dense(50,activation="relu"))
model.add(Dense(50,activation="relu"))
model.add(Dense(50,activation="relu"))
model.add(Dense(50,activation="relu"))
model.add(Dense(50,activation="relu"))
model.add(Dense(50,activation="relu"))
model.add(Dense(50,activation="linear"))
model.add(Dense(50,activation="linear"))
model.add(Dense(50,activation="linear"))
model.add(Dense(50,activation="linear"))
model.add(Dense(1,activation ="linear" ))

model.compile(loss = "mean_squared_error",optimizer = "adam")

model.fit(scaled_train,train_y,epochs = 500,shuffle=True,verbose=0)
model.evaluate(scaled_train,train_y)
output = model.predict(scaled_test)
output = pd.DataFrame(output)
Submission = pd.DataFrame({"Id":SS["Id"],"SalePrice":output[0]})

Submission.to_csv("submission.csv",index=False)