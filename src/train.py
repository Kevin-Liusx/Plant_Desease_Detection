import tensorflow as tf

import config
import model as cnn_model
import dataset

def run_training(train_data_dir, validate_data_dir, batch_size, epochs, image_size, image_dimension=None):
    training_dataset = dataset.load_image_dataset(train_data_dir, image_size, batch_size)
    validation_dataset = dataset.load_image_dataset(validate_data_dir, image_size, batch_size)

    model = cnn_model.CNN(num_classes=38)

    if image_dimension is not None:
        model(tf.keras.Input(shape=image_dimension))

    # Compile the model, using the Adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(training_dataset, epochs=epochs, validation_data=validation_dataset)

    return history


if __name__ == '__main__':
    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    training_history = run_training(config.TRAIN_DATA_DIR, config.VALIDATION_DATA_DIR, config.BATCH_SIZE, config.EPOCHS, config.IMAGE_SIZE, image_dimension=(128, 128, 3))