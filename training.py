import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras import optimizers

from config import __FONTS_DICT__, __DIRNAME_INPUT__, __DIRNAME_WORKING__, models
from utils import plot_learning_graph

def train_model(model, list_char_images_training, list_chars_fonts_training, list_char_images_validation, list_chars_fonts_validation):
    train_data = np.array(list_char_images_training)
    validation_data = np.array(list_char_images_validation)

    # Reverse the original dictionary for mapping names to numbers
    # {'Flower Rose Brush': 0, 'Skylark': 1, 'Sweet Puppy': 2, 'Ubuntu Mono': 3, 'VertigoFLF': 4, 'Wanted M54': 5, 'always forever': 6}
    reverse_fonts_dict = {v: k for k, v in __FONTS_DICT__.items()}
    # Map font names to numbers on training data
    # Ex. [3, 3, 3, 6, 6, 6, 5, 5, 5, 1, 1, 1, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 6, 6, 6, 6, 6, 4, 4, 4]
    list_chars_fonts_training_id = [
        reverse_fonts_dict[name] for name in list_chars_fonts_training]
    train_labels = to_categorical(list_chars_fonts_training_id, num_classes=len(
        __FONTS_DICT__))  # Convert integer labels to one-hot encoded labels

    # Map font names to numbers on validation data
    # Ex. [3, 3, 3, 6, 6, 6, 5, 5, 5, 1, 1, 1, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 6, 6, 6, 6, 6, 4, 4, 4]
    list_chars_fonts_validation_id = [
        reverse_fonts_dict[name] for name in list_chars_fonts_validation]
    validation_labels = to_categorical(list_chars_fonts_validation_id, num_classes=len(
        __FONTS_DICT__))  # Convert integer labels to one-hot encoded labels

    # start training
    learning = model.fit(train_data, train_labels, epochs=30, verbose=2)

    # evaluate training with validation data
    model.evaluate(validation_data, validation_labels)

    # Save entire model
    model.save(__DIRNAME_WORKING__ + models.LAST_TRAINED_MODEL)

    # plot (loss function x accuracy) graph
    plot_learning_graph(learning)

    return model


def create_model(fonts_num):
    # Assuming input images are grayscale characters with shape (height, width, channels)
    input_shape = (32, 32, 1)  # Adjust the input shape based on your data

    # Create a Sequential model
    model = Sequential([
        # Convolutional layers
        Conv2D(32, kernel_size=(3, 3), activation='relu',
               input_shape=input_shape),
        Conv2D(64, (3, 3), activation='relu',),
        Dropout(0.25),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),

        # Dense (fully connected) layers
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(fonts_num, activation='softmax')
    ])

    # Compile the model
    optimizer = optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()  # Print the model summary

    return model
