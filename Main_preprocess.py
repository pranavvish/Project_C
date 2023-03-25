#Imports
import Data_preprocess as d
import os
import pandas as pd

#Creating lists to store file names

#Training files
imu_train=[]
imu_t_train=[]
ann_train=[]
ann_t_train=[]

#Testing files
imu_test=[]
imu_t_test=[]
ann_t_test=[]

# Training_path='./OriginalData/TrainingData'
# Testing_path='./OriginalData/TestingData'
# Training_files=os.listdir(Training_path)
# Testing_path=os.listdir(Testing_path)

#Scanning the list of files in training data

for path in os.listdir('./OriginalData/TrainingData'):
    if path.endswith('__x.csv'):
        imu_train.append(str(path))
    if path.endswith('__x_time.csv'):
        imu_t_train.append(str(path))
    if path.endswith('__y.csv'):
        ann_train.append(str(path))
    if path.endswith('__y_time.csv'):
        ann_t_train.append(str(path))

#Scanning the list of files in testing data

for path in os.listdir('./OriginalData/TestingData'):
    if path.endswith('__x.csv'):
        imu_test.append(str(path))
    if path.endswith('__x_time.csv'):
        imu_t_test.append(str(path))
    if path.endswith('__y_time.csv'):
        ann_t_test.append(str(path))

#Sorting 

imu_t_train.sort()
imu_train.sort()
ann_train.sort()
ann_t_train.sort()
imu_t_test.sort()
imu_test.sort()
ann_t_test.sort()

#Generating output csv file which can be used for training 
for i in range(len(imu_train)):
    x=imu_train[i]
    x_t=imu_t_train[i]
    y=ann_train[i]
    y_t=ann_t_train[i]
    user_data=d.Data_train(x,x_t,y,y_t)
    file_path='./PreprocessedData/Train/User_Data_'+str(i+1)+'.csv'   
    user_data.to_csv(file_path,index=False)

#Generating output csv file which can be used for testing 
for i in range(len(imu_test)):
    x=imu_test[i]
    x_t=imu_t_test[i]
    y_t=ann_t_test[i]
    user_data=d.Data_test(x,x_t,y_t)
    file_path='./PreprocessedData/Test/User_Data_'+str(i+1)+'.csv'   
    user_data.to_csv(file_path,index=False)
