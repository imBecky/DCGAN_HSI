import time
from IPython import display
# from param import *
from tensorflow.keras import layers as layers

from utils import *

# from resnet34 import ResNetModel

BANDS = 72
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7 * 7 * 72, use_bias=False, input_shape=(72, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((3528, 72)))

    model.add(layers.Conv1DTranspose(128, 6, strides=1, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv1DTranspose(64, 6, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv1DTranspose(1, 6, strides=2, padding='same', use_bias=False, activation='tanh'))

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv1D(64, 5, strides=2, padding='same',
                            input_shape=(1, 72)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv1D(128, 5, strides=2, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


def make_discriminator_domain_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(FEATURE_dim, 1)))
    model.add(ResBlock(64))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(32))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv1D(filters=3,
                                     strides=3,
                                     kernel_size=7))
    model.add(tf.keras.layers.MaxPool1D(pool_size=4))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dense(1, activation='sigmoid', use_bias=True))
    return model


def make_classifier_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(360, input_shape=(36, 1)))
    model.add(tf.keras.layers.Conv1D(filters=180,
                                     kernel_size=6,
                                     strides=1,
                                     padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool1D(pool_size=3))
    model.add(ResBlock(90))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(ResBlock(90))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(ResBlock(180))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(720))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(360))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(CLASSES_NUM, activation='relu'))
    model.add(tf.keras.layers.LeakyReLU())
    return model


def make_encoder_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(10, input_shape=(72, 1), use_bias=True))
    model.add(tf.keras.layers.Conv1D(filters=3,
                                     kernel_size=3,
                                     strides=2,
                                     padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dense(1))
    # model.add(tf.keras.layers.Conv1D(filters=1,
    #                                  kernel_size=3,
    #                                  strides=2,
    #                                  padding='same'))
    model.add(tf.keras.layers.Activation('tanh'))
    return model


# define losses
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def discriminator_domain_loss(real_target_output, fake_target_output, fake_source_output):
    real_loss = cross_entropy(tf.ones_like(real_target_output), real_target_output)
    real_loss += cross_entropy(tf.ones_like(fake_target_output), fake_target_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_source_output), fake_source_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def encoder_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def classifier_loss(prediction, label):
    return cross_entropy(label, prediction)
