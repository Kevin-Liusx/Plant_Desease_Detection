import tensorflow as tf

import config

def load_image_dataset(image_paths, image_size, batch_size):
    dataset = tf.keras.utils.image_dataset_from_directory(
        image_paths,
        labels="inferred",
        label_mode="categorical",
        class_names=None,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=False
        )
    return dataset
