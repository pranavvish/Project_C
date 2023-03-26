import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
import Balance_Data as b
import Model as m
import os
from sklearn.model_selection import train_test_split

Data_init=pd.read_csv('./PreprocessedData/Train/User_Data_2.csv')
Data=b.balance(Data_init)
Data=Data.sort_values(by=['Timestamp'])
Data['Activity'].value_counts()

x=Data.drop(['Activity','Timestamp'],axis=1).copy()
y=Data[['Activity']].copy()
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3) 

m.Mod(x_train,y_train,x_test,y_test)