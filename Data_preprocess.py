#Imports
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import re

#Col_names for timeseries data
colnames_imu=['X_a', 'Y_a', 'Z_a','X_g', 'Y_g', 'Z_g','Timestamp']
colnames_ann=['Activity','Timestamp']
colnames_ann_t=['Timestamp']
colnames_imu_t=['Timestamp']

#Function to calculate dataframe for user
def Data(imu,imu_t,ann,ann_t):

    imu  = pd.read_csv('OriginalData/TrainingData/'+imu, names=colnames_imu);
    imu_t = pd.read_csv('OriginalData/TrainingData/'+imu_t,names=colnames_imu_t);
    ann = pd.read_csv('OriginalData/TrainingData/'+ann, names=colnames_ann);
    ann_t = pd.read_csv('OriginalData/TrainingData/'+ann_t,names=colnames_ann_t);

    imu['Timestamp']=imu_t['Timestamp']
    ann['Timestamp']=ann_t['Timestamp']
    
    #df_final=imu.append(ann).sort_values(by=['Timestamp'])
    df_final=pd.concat([imu,ann]).sort_values(by=['Timestamp'])
    df_final=df_final.interpolate()
    df_final=df_final.dropna()
    df_final=df_final.loc[:,['Timestamp','X_a', 'Y_a', 'Z_a','X_g', 'Y_g', 'Z_g','Activity']]
    #df_final.insert(8,'y','0')

    # y_Conditions = [
    # (df_final['Activity'] < 0.75),
    # (df_final['Activity'] >= 0.75) & (df_final['Activity'] < 1.5),
    # (df_final['Activity'] >= 1.5) & (df_final['Activity'] < 2.25),
    # #(df_final['Activity'] >= 1.5) & (df_final['Activity'] < 2)
    # #(df_final['Activity'] >= 2) & (df_final['Activity'] < 2.5)
    # (df_final['Activity'] >= 2.25)]
    
    # y = ['0', '1', '2', '3']
    # df_final['Activity'] = np.select(y_Conditions, y)

    df_final['Activity'] = df_final['Activity'].round().clip(0, 3)
        
    return df_final