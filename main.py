# Import the necessary libraries
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Define the CNN model
model = tf.keras.Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Create a data generator for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Create a data generator for test data
test_datagen = ImageDataGenerator(rescale=1./255)

# Load the training data and apply the data generator
train_generator = train_datagen.flow(
    X_train, y_train,
    batch_size=32)

# Load the test data and apply the data generator
test_generator = test_datagen.flow(
    X_test, y_test,
    batch_size=32)

# Define a callback for early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Train the model on the training data
history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // 32,
    epochs=100,
    validation_data=test_generator,
    validation_steps=len(X_test) // 32,
    callbacks=[early_stopping])

# Plot the training and validation loss
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
# Clone the TensorFlow source code from GitHub
git clone https://github.com/tensorflow/tensorflow.git

# Change to the TensorFlow directory
cd tensorflow

# Configure the TensorFlow build
./configure

# Build TensorFlow with the specified compiler flags
bazel build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --linkopt="-static-libstdc++" //tensorflow/tools/pip_package:build_pip_package


