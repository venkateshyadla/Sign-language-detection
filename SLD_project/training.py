import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants
IMAGE_SIZE = (300, 300)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 26  # 26 classes
DATASET_PATH = "/suryas/gb_S/final_project/"

# Load dataset
def load_dataset(dataset_path):
    images = []
    labels = []
    label_mapping = {}
    
    class_folders = sorted(os.listdir(dataset_path))
    for label, category in enumerate(class_folders):
        label_mapping[label] = category
        category_path = os.path.join(dataset_path, category)
        
        for image_name in os.listdir(category_path):
            image_path = os.path.join(category_path, image_name)
            image = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
            image = tf.keras.preprocessing.image.img_to_array(image)
            images.append(image)
            labels.append(label)
    
    return np.array(images), np.array(labels), label_mapping

# Split dataset into training and testing sets
def split_dataset(images, labels):
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test

# Data augmentation and normalization
def create_train_generator(x_train):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    train_datagen.fit(x_train)
    return train_datagen

# Define model architecture with dropout

def create_model(input_shape, num_classes):
    model = models.Sequential()
      # Block 1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 4
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 5
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Flatten
    model.add(Flatten())

    # Fully connected layers
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(num_classes, activation='softmax')) 
    return model

# training process
def train_model(model, train_generator, x_train, y_train, x_test, y_test, batch_size, epochs):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    # learning rate is changed
    model.compile(optimizer = optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(train_generator.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(x_train) // batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test))
    
    return model, history

# Save label mapping to text file
def save_label_mapping(label_mapping, file_path):
    with open(file_path, "w") as file:
        for label, category in label_mapping.items():
            file.write(f"{label}: {category}\n")

            
# Load dataset
images, labels, label_mapping = load_dataset(DATASET_PATH)

# Split dataset into training and testing sets
x_train, x_test, y_train, y_test = split_dataset(images, labels)

# Create data generator for training
train_datagen = create_train_generator(x_train)

# Define model
model = create_model(input_shape=(*IMAGE_SIZE, 3), num_classes=NUM_CLASSES)

# Train model
trained_model, history = train_model(model, train_datagen, x_train, y_train, x_test, y_test, BATCH_SIZE, EPOCHS)

# Save trained model
trained_model.save("trained_model.h5")

# Save label mapping to text file
save_label_mapping(label_mapping, "label_mapping.txt")

# test accuracy
test_loss, test_accuracy = trained_model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy}")