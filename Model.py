import time
from IPython import display
# from param import *
from tensorflow.keras import layers as layers

from utils import *

# from resnet34 import ResNetModel

BANDS = 72
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ResBlock(tf.keras.layers.Layer):
    def __init__(self, out_channel):
        super(ResBlock, self).__init__()
        self.c1 = tf.keras.layers.Conv1D(filters=out_channel,
                                         kernel_size=1,
                                         strides=1,
                                         padding='same',
                                         use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.r1 = tf.keras.layers.Activation('tanh')

    def __call__(self, inputs):
        x = inputs
        x = self.c1(x)
        res = x
        x = self.bn1(x)
        x = self.r1(x)
        x += res
        return x


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7 * 72, use_bias=False, input_shape=(72, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((504, 72)))

    model.add(layers.Conv1DTranspose(144, 6, strides=1, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv1DTranspose(72, 6, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv1DTranspose(36, 6, strides=2, padding='same', use_bias=False, activation='tanh'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(72))
    model.add(layers.Reshape((72, 1)))
    model.add(layers.BatchNormalization())
    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv1D(72, 7, strides=2, padding='same',
                            input_shape=(72, 1)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv1D(128, 5, strides=2, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


def make_classifier_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(360, input_shape=(72, 1)))
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


# define losses
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def encoder_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def classifier_loss(prediction, label):
    return cross_entropy(label, prediction)


generator_optimizer = tf.optimizers.Adagrad(lr)
discriminator_optimizer = tf.optimizers.Adagrad(lr)
classifier_optimizer = tf.optimizers.Adagrad(lr)
