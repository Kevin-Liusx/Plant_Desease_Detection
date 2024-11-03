import tensorflow as tf

class CNN(tf.keras.Model):
    def __init__(self, num_classes=38, **kwargs):
        super(CNN, self).__init__(**kwargs)
        self.conv1a = tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu') 
        self.conv1b = tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)

        self.conv2a = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu') 
        self.conv2b = tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)

        self.conv3a = tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')
        self.conv3b = tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu')
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)

        self.cov4a = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')
        self.conv4b = tf.keras.layers.Conv2D(256, kernel_size=3, activation='relu')
        self.pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)

        self.conv5a = tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation='relu')
        self.conv5b = tf.keras.layers.Conv2D(512, kernel_size=3, activation='relu')
        self.pool5 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)

        self.dropout1 = tf.keras.layers.Dropout(0.25)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1500, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.4)
        self.output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')

    def get_config(self):
        # Return the config of the model, allowing for re-creation
        config = super(CNN, self).get_config()
        config.update({"num_classes": self.output_layer.units})
        return config

    @classmethod
    def from_config(cls, config):
        # Recreate an instance of the model from the config
        return cls(**config)

    def call(self, x):
        x = self.conv1a(x)
        x = self.conv1b(x)
        x = self.pool1(x)

        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.pool3(x)

        x = self.cov4a(x)
        x = self.conv4b(x)
        x = self.pool4(x)

        x = self.conv5a(x)
        x = self.conv5b(x)
        x = self.pool5(x)

        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.dropout2(x)
        x = self.output_layer(x)

        return x

# model = CNN(num_classes=38)
# model(tf.keras.Input(shape=(128, 128, 3)))
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()