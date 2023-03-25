#Imports
import Data_preprocess as d
import os
import pandas as pd

#Creating lists to store file names

imu=[]
imu_t=[]
ann=[]
ann_t=[]

# Training_path='./OriginalData/TrainingData'
# Testing_path='./OriginalData/TestingData'
# Training_files=os.listdir(Training_path)
# Testing_path=os.listdir(Testing_path)

#Scanning the list of files in training data

for path in os.listdir('./OriginalData/TrainingData'):
    if path.endswith('__x.csv'):
        imu.append(str(path))
    if path.endswith('__x_time.csv'):
        imu_t.append(str(path))
    if path.endswith('__y.csv'):
        ann.append(str(path))
    if path.endswith('__y_time.csv'):
        ann_t.append(str(path))

#Sorting 

imu_t.sort()
imu.sort()
ann.sort()
ann_t.sort()

#Generating output csv file which can be used for training 
for i in range(len(imu)):
    x=imu[i]
    x_t=imu_t[i]
    y=ann[i]
    y_t=ann_t[i]
    user_data=d.Data(x,x_t,y,y_t)
    file_path='./PreprocessedData/User_Data_'+str(i+1)+'.csv'   
    user_data.to_csv(file_path,index=False)
