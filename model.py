import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten

# Folder containing training images
folder = 'track_1'
main_folder = '../data/{}'.format(folder)
data = pd.read_csv('{}/driving_log_combined.csv'.format(main_folder))

data['steering'].hist(bins=100)
data['target'] = data['steering']

zero_data = data[data['target'] == 0.0]
non_zero_data = data[data['target'] != 0.0]
image_paths = []
measurements = []

# Considering data points where the steering angle is non zero
for index, row in non_zero_data.iterrows():
    img_name = row['center'].split('/')[-1]
    img_path = '{}/{}/IMG/{}'.format(main_folder, row['run'], img_name)
    image_paths.append(img_path)
    steering = row['target']    
    measurements.append(steering)
    
    img_name = row['left'].split('/')[-1]
    img_path = '{}/{}/IMG/{}'.format(main_folder, row['run'], img_name)
    image_paths.append(img_path)
    steering = row['target'] + 0.25
    measurements.append(steering)

    img_name = row['right'].split('/')[-1]
    img_path = '{}/{}/IMG/{}'.format(main_folder, row['run'], img_name)
    image_paths.append(img_path)
    steering = row['target'] - 0.25
    measurements.append(steering)

# Considering 10 % of the data points where the steering angle is zero
for index, row in zero_data.sample(len(non_zero_data)//10).iterrows():
    img_name = row['center'].split('/')[-1]
    img_path = '{}/{}/IMG/{}'.format(main_folder, row['run'], img_name)
    image_paths.append(img_path)
    steering = row['target']    
    measurements.append(steering)
    
    img_name = row['left'].split('/')[-1]
    img_path = '{}/{}/IMG/{}'.format(main_folder, row['run'], img_name)
    image_paths.append(img_path)
    steering = row['target'] + 0.25
    measurements.append(steering)

    img_name = row['right'].split('/')[-1]
    img_path = '{}/{}/IMG/{}'.format(main_folder, row['run'], img_name)
    image_paths.append(img_path)
    steering = row['target'] - 0.25
    measurements.append(steering)

pd.Series(measurements).hist(bins=100)

X_train, X_valid, y_train, y_valid = train_test_split(image_paths, measurements, test_size=0.2, random_state=42)

# Some utility functions
def crop_color_size_change(img):
    cropped_img = img[60:140, :, :]
    color_change = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2YUV)
    return cv2.resize(color_change,(200, 66), interpolation = cv2.INTER_AREA)
def load_image(path):
    img = cv2.imread(path)
    return crop_color_size_change(img)
def augment_image(img, steering):
    if np.random.rand() > 0.5:
        new_img = cv2.flip(img, 1)
        steering = steering * -1.0
        return new_img, steering
    return img, steering
def data_generator(is_train, batch_size, X, y):
    while True:
        images = []
        measurements = []
        for index in np.random.randint(0,len(X), batch_size):
            path = X[index]
            steering = y[index]
            img = load_image(path)
            if is_train:
                img, steering = augment_image(img, steering)
            images.append(img)
            measurements.append(steering)

        images = np.array(images)
        measurements = np.array(measurements)    
        yield (images, measurements)

# Defining the model
model = Sequential()
model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(66,200,3)))
model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))

# Compiling the model
model.compile(loss='mse', optimizer='adam')


# Training for 3 epochs
version = 21
batch_size = 100
model.fit_generator(data_generator(is_train=True, batch_size=batch_size, X=X_train, y=y_train),
                    samples_per_epoch = len(X_train) * 2,
                    nb_epoch = 3,
                    max_q_size=1,
                    validation_data=data_generator(is_train=False, batch_size=50, X=X_valid, y=y_valid),
                    nb_val_samples=len(X_valid))

model_name = '../models/model_v{}.h5'.format(version)
model.save(model_name)
version = version+1
print(model_name)