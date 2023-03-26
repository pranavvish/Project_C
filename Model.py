import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# generate random input and output data

def Mod(X,y,X_t,y_t):
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X)
    X_train = X_train.reshape(X.shape[0], X.shape[1], 1)
    X_test = scaler.transform(X_t)
    X_test = X_test.reshape(X_t.shape[0], X_t.shape[1], 1)
    y_train = tf.keras.utils.to_categorical(y, num_classes=4)
    y_test = tf.keras.utils.to_categorical(y_t, num_classes=4)

    # print(X_train)
    # print(X_train.shape[0])
    # print(X_train.shape[1])
    # print(X_train.shape[2])
    # print(y_test.shape)
    
    batch_size = 512
    
    model = Sequential()
    model.add(Conv1D(filters=512, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Conv1D(filters=256,kernel_size=3,activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    # compile the model
    opt = Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # fit the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=batch_size, validation_data=(X_test,y_test), verbose=1)

    # plot the training history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # plot actual vs predicted values for test data
    y_pred = np.round(model.predict(X_test)).astype(int)
    y_pred_classes = np.argmax(y_pred, axis=1)
    for i in range(y_test.shape[1]):
        plt.plot(y_test[:, i], label=f'actual {i}')
        plt.plot(y_pred[:, i], label=f'predicted {i}')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    # calculate and print evaluation metrics
    mae_list = []
    mse_list = []
    for i in range(y_test.shape[1]):
        mae = np.mean(np.abs(y_test[:, i] - y_pred_classes))
        mse = np.mean((y_test[:, i] - y_pred_classes)**2)
        mae_list.append(mae)
        mse_list.append(mse)
    mae = np.mean(mae_list)
    mse = np.mean(mse_list)
    print(f'MAE: {mae}')
    print(f'MSE: {mse}')

    print(model.summary())

