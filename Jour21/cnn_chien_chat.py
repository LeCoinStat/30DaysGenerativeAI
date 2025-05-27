import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds


def load_data(batch_size=32, img_size=(160, 160)):
    (train_ds, val_ds), ds_info = tfds.load(
        'cats_vs_dogs',
        split=['train[:80%]', 'train[80%:]'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def format_example(image, label):
        image = tf.image.resize(image, img_size)
        image = image / 255.0
        return image, label

    train_ds = train_ds.map(format_example).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(format_example).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, ds_info


def build_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    batch_size = 32
    img_size = (160, 160)
    train_ds, val_ds, info = load_data(batch_size, img_size)
    model = build_model((*img_size, 3))
    model.fit(train_ds, validation_data=val_ds, epochs=3)
    model.evaluate(val_ds)


if __name__ == '__main__':
    main()
