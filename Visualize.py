import csv
import matplotlib.pyplot as plt


imu = csv.reader('TrainingData/subject_001_01__x.csv');
imu_t = csv.reader('TrainingData/subject_001_01__x_time.csv');
ann = csv.reader('TrainingData/subject_001_01__y.csv');
ann_t = csv.reader('TrainingData/subject_001_01__y_time.csv');

#Plotting data

fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(imu_t, imu)
axs[1].plot(ann_t, ann)
