import numpy as np
import os

# ... rest of your code ...

# Create the preview directory if it doesn't exist


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image  # Only needed for loading the image

# Load the image as a NumPy array
img = np.array(Image.open("C:\\Users\\tanma\\AppData\\Local\\Programs\\Python\\Python37\\dataSet\\dog\\test_dog_3.jpg"))

# Define the data augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Generate batches of augmented images
i = 0
for batch in datagen.flow(img[np.newaxis, ...],  # Add extra dimension for batch
                          batch_size=1,
                          save_to_dir='preview4',
                          save_prefix='val',
                          save_format='jpeg'):
    i += 1
    if i > 20:
        break


