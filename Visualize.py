
import pandas as pd
import matplotlib.pyplot as plt
# Simple script illustrating how to visualize some of the data
#changes
# Loading sample data
imu = pd.read_csv('Training/subject_001_01__x.csv');
imu_t = pd.read_csv('Training/subject_001_01__x_time.csv');
ann = pd.read_csv('Training/subject_001_01__y.csv');
ann_t = pd.read_csv('Training/subject_001_01__y_time.csv');

# Plotting data
# figure(2), clf;
# h(1) = subplot(2,1,1); plot(imu_t,imu);
# h(2) = subplot(2,1,2); plot(ann_t,ann);
# linkaxes(h,'x');

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

# Plot the first subplot
ax1.plot(imu_t, imu, color='blue')
ax1.set_title('IMU')

# Plot the second subplot
ax2.plot(ann_t, ann, color='green')
ax2.set_title('ANN')

# Add a shared x-axis label
fig.text(0.5, 0.04, 'X', ha='center')

# Add a shared y-axis label
fig.text(0.04, 0.5, 'Y', va='center', rotation='vertical')

# Show the plot
plt.show()
