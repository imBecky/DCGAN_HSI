import scipy.io as sio
from Model import *
from utils import *
import os

# adversarial training block
@tf.function
def GAN_train_step(generator, discriminator,
                   source_batch, target_batch):
    X_s, Y_s = get_data_from_batch(source_batch)
    X_t, Y_t = get_data_from_batch(target_batch)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_target = generator(X_s, training=True)
        fake_decision = discriminator(generated_target, training=True)
        real_decision = discriminator(X_t, training=True)

        gen_loss = generator_loss(fake_decision)
        disc_loss = discriminator_loss(real_decision, fake_decision)

        gen_gradient = gen_tape.gradient(gen_loss, generator.trainable_variables)
        disc_gradient = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gen_gradient,
                                                generator.trainable_variables))
        discriminator_optimizer.apply_gradients((zip(disc_gradient,
                                                     discriminator.trainable_variables)))


@tf.function
def classify_train_step(generator, classifier,
                        source_batch, target_batch):
    X_s, Y_s = get_data_from_batch(source_batch)
    X_t, Y_t = get_data_from_batch(target_batch)
    X_s = generator(X_s, training=False)

    with tf.GradientTape() as cls_tape:
        prediction = classifier(X_t, training=True)
        pred_loss = classifier_loss(prediction, Y_t)
        prediction2 = classifier(X_s, training=True)
        pred_loss += classifier_loss(prediction2, Y_s)

        pred_gradient = cls_tape.gradient(pred_loss, classifier.trainable_variables)
        classifier_optimizer.apply_gradients(zip(pred_gradient,
                                                 classifier.trainable_variables))


def train(generator, discriminator, classifier,
          source_ds, target_ds, target_test_ds,
          epochs):
    for epoch in range(epochs):
        start = time.time()
        for source_batch in source_ds.as_numpy_iterator():
            for target_batch in target_ds.as_numpy_iterator():
                GAN_train_step(generator, discriminator,
                               source_batch, target_batch)
        duration = time.time() - start
        print('duration for epoch {} is {}s'.format(epoch+1, duration))
        if epoch % 15 == 0:
            generate_and_save_Images(generator, epoch,
                                     source_ds.as_numpy_iterator().next()['data'])
    for epoch in range(epochs):
        for source_batch in source_ds.as_numpy_iterator():
            for target_batch in target_ds.as_numpy_iterator():
                classify_train_step(generator, classifier,
                                    source_batch, target_batch)
        calculate_acc(target_test_ds, classifier, epoch)
