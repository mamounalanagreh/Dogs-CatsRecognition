import os

# Get the current working directory
current_directory = os.getcwd()

# Construct the full path to the dataset directory
dataset_directory = os.path.join(current_directory, 'dataset')

from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential

# CNN
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3),
                        activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Image Pre-Processing
from keras.preprocessing.image import ImageDataGenerator

tr_datagen = ImageDataGenerator(rescale=1/255, horizontal_flip=True,
                                zoom_range=0.2, shear_range=0.2)

ts_datagen = ImageDataGenerator(rescale=1/255)

tr_dataset = tr_datagen.flow_from_directory('training_set',
                                            target_size=(64,64),
                                            class_mode='binary',
                                            batch_size=32)

ts_dataset = tr_datagen.flow_from_directory('test_set',
                                            target_size=(64,64),
                                            class_mode='binary',
                                            batch_size=32)

# Training
model.fit(tr_dataset, steps_per_epoch=int(4000/32), epochs=100,
          validation_data=ts_dataset, validation_steps=int(1000/32))

print(tr_dataset.class_indices)

# Single Prediction
import numpy as np
from keras.utils import load_img, img_to_array

test_image = load_img('dataset/single_prediction/cat_or_dog_1.jpg',
                      target_size=(64,64))
test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

"""
# predict multiple images
img1 = load_img('sample1.jpg',target_size=(64,64))
img1 = img_to_array(img1)
img2 = load_img('sample2.jpg',target_size=(64,64))
img2 = img_to_array(img2)
img3 = load_img('sample3.jpg',target_size=(64,64))
img3 = img_to_array(img3)
imgs = np.array([img1, img2, img3])
"""

result = model.predict(test_image)
if result[0][0] == 1:
    print("Dog")
else:
    print("Cat")