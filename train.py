import scipy.io as sio
from Model import *
from utils import *
import os

generator_s_optimizer = tf.keras.optimizers.Adagrad(lr)
generator_t_optimizer = tf.keras.optimizers.Adagrad(lr)
discriminator_t_optimizer = tf.keras.optimizers.Adagrad(lr)
discriminator_s_optimizer = tf.keras.optimizers.Adagrad(lr)
discriminator_domain_optimizer = tf.keras.optimizers.Adagrad(lr)
encoder_s_optimizer = tf.keras.optimizers.Adagrad(lr)
encoder_t_optimizer = tf.keras.optimizers.Adagrad(lr)
classifier_optimizer = tf.keras.optimizers.Adagrad(lr)


@tf.function
def train_step_encoder(encoder_s,
                       encoder_t,
                       classifier,
                       source_batch,
                       target_batch,
                       epoch):
    """encode the real samples,
       make predictions of features"""
    # print(epoch)
    source_data, source_label = get_data_from_batch(source_batch)
    target_data, target_label = get_data_from_batch(target_batch)
    with tf.GradientTape(persistent=True) as tape:
        source_feature = encoder_s(source_data, training=True)
        target_feature = encoder_t(target_data, training=True)

        classify_source = classifier(source_feature, training=True)
        # print('source_decision', classify_source[0][:])
        classify_loss = classifier_loss(classify_source, source_label)
        encoder_s_loss = classifier_loss(classify_source, source_label)

        classify_target = classifier(target_feature, training=True)
        classify_loss += classifier_loss(classify_target, target_label)
        encoder_t_loss = classifier_loss(classify_target, target_label)

        gradient_source = tape.gradient(encoder_s_loss,
                                        encoder_s.trainable_variables)
        gradient_target = tape.gradient(encoder_t_loss,
                                        encoder_t.trainable_variables)
        gradient_classifier = tape.gradient(classify_loss,
                                            classifier.trainable_variables)
        encoder_s_optimizer.apply_gradients(zip(gradient_source,
                                                encoder_s.trainable_variables))
        encoder_t_optimizer.apply_gradients(zip(gradient_target,
                                                encoder_t.trainable_variables))
        classifier_optimizer.apply_gradients(zip(gradient_classifier,
                                                 classifier.trainable_variables))


def train_step_domain(encoder_s,
                      encoder_t,
                      classifier,
                      discriminator_domain,
                      source_batch,
                      target_batch):
    source_data, source_label = get_data_from_batch(source_batch)
    target_data, target_label = get_data_from_batch(target_batch)
    with tf.GradientTape(persistent=True) as tape:
        source_feature = encoder_s(source_data, training=True)
        target_feature = encoder_t(target_data, training=True)

        disc_decision_s = discriminator_domain(source_feature, training=True)
        disc_decision_t = discriminator_domain(target_feature, training=True)
        prediction_s = classifier(source_feature, training=True)
        prediction_t = classifier(target_feature, training=True)

        # when input feature source
        encoder_s_loss = encoder_loss(disc_decision_s)
        disc_domain_loss = discriminator_loss(disc_decision_t, disc_decision_s)
        encoder_s_loss += cross_entropy(source_label, prediction_s)
        # when input feature target
        encoder_t_loss = encoder_loss(disc_decision_t)
        disc_domain_loss += discriminator_loss(disc_decision_s, disc_decision_t)
        encoder_t_loss += cross_entropy(target_label, prediction_t)

        encoder_s_gradient = tape.gradient(encoder_s_loss, encoder_s.trainable_variables)
        encoder_t_gradient = tape.gradient(encoder_t_loss, encoder_t.trainable_variables)
        discriminator_domain_gradient = tape.gradient(disc_domain_loss, discriminator_domain.trainable_variables)

        encoder_s_optimizer.apply_gradients(zip(encoder_s_gradient, encoder_s.trainable_variables))
        encoder_t_optimizer.apply_gradients(zip(encoder_t_gradient, encoder_s.trainable_variables))
        discriminator_domain_optimizer.apply_gradients(zip(discriminator_domain_gradient,
                                                           discriminator_domain.trainable_variables))
    del tape


def train_step_double_GAN(generator_s, generator_t,
                          discriminator_t, discriminator_s,
                          batch_source, batch_target):
    data_source, label_source = get_data_from_batch(batch_source)
    data_target, label_target = get_data_from_batch(batch_target)
    with tf.GradientTape(persistent=True) as tape:
        generated_t = generator_s(data_source, training=True)
        generated_s = generator_t(data_target, training=True)
        fake_decision_t = discriminator_t(generated_t)
        real_decision_t = discriminator_t(data_target)
        fake_decision_s = discriminator_s(generated_s)
        real_decision_s = discriminator_s(data_source)

        gen_s_loss = generator_loss(fake_decision_t)
        disc_t_loss = discriminator_loss(real_decision_t, fake_decision_t)
        gen_t_loss = generator_loss(fake_decision_s)
        disc_s_loss = discriminator_loss(real_decision_s, fake_decision_s)

        gradient_gen_s = tape.gradient(gen_s_loss, generator_s.trainable_variables)
        gradient_gen_t = tape.gradient(gen_t_loss, generator_t.trainable_variables)
        gradient_disc_t = tape.gradient(disc_t_loss, discriminator_t.trainable_variables)
        gradient_disc_s = tape.gradient(disc_s_loss, discriminator_s.trainable_variables)

        generator_s_optimizer.apply_gradients(zip(gradient_gen_s, generator_s.trainable_variables))
        generator_t_optimizer.apply_gradients(zip(gradient_gen_t, generator_t.trainable_variables))
        discriminator_t_optimizer.apply_gradients(zip(gradient_disc_t, discriminator_t.trainable_variables))
        discriminator_s_optimizer.apply_gradients(zip(gradient_disc_s, discriminator_s.trainable_variables))
    del tape


def train_step_whole(generator_s, generator_t,
                     discriminator_t, discriminator_s,
                     discriminator_domain,
                     encoder_s, encoder_t,
                     classifier,
                     source_batch, target_batch
                     ):
    data_source, label_source = get_data_from_batch(source_batch)
    data_target, label_target = get_data_from_batch(target_batch)
    with tf.GradientTape(persistent=True) as tape:
        # GAN block 1
        generated_target = generator_s(data_source, training=True)
        real_decision = discriminator_t(data_target, training=True)
        fake_decision = discriminator_t(generated_target, training=True)
        gen_s_loss = generator_loss(fake_decision)
        disc_t_loss = discriminator_loss(real_decision, fake_decision)
        gen_s_gradient = tape.gradient(gen_s_loss, generator_s.trainable_variables)
        disc_t_gradient = tape.gradient(disc_t_loss, discriminator_t.trainable_variables)
        generator_s_optimizer.apply_gradients(zip(gen_s_gradient, generator_s.trainable_variables))
        discriminator_t_optimizer.apply_gradients(zip(disc_t_gradient, discriminator_t.trainable_variables))

        # GAN block2
        generated_source = generator_t(data_target, training=True)
        real_decision2 = discriminator_s(data_source, training=True)
        fake_decision2 = discriminator_s(generated_source, training=True)
        gen_t_loss = generator_loss(fake_decision2)
        disc_s_loss = discriminator_loss(real_decision2, fake_decision2)
        gen_t_gradient = tape.gradient(gen_t_loss, generator_t.trainable_variables)
        disc_s_gradient = tape.gradient(disc_s_loss, discriminator_s.trainable_variables)
        generator_t_optimizer.apply_gradients(zip(gen_t_gradient, generator_t.trainable_variables))
        discriminator_s_optimizer.apply_gradients(zip(disc_s_gradient, discriminator_s.trainable_variables))

        # GAN block3
        feature_s = encoder_s(generated_source, training=True)
        feature_t = encoder_t(generated_target, training=True)
        feature_t_real = encoder_t(data_target, training=True)

        # when import feature_s mainly
        real_decision3 = discriminator_domain(feature_t, training=True)
        fake_decision3 = discriminator_domain(feature_s, training=True)
        disc_domain_loss = discriminator_loss(real_decision3, fake_decision3)
        encoder_s_loss = encoder_loss(fake_decision3)

        # when import feature_t mainly
        real_decision4 = discriminator_domain(feature_s, training=True)
        fake_decision4 = discriminator_domain(feature_t, training=True)
        disc_domain_loss += discriminator_loss(real_decision4, fake_decision4)
        encoder_t_loss = encoder_loss(fake_decision4)

        # calculate the gradient of encoders later, since it need to be combined with classification loss
        disc_domain_gradient = tape.gradient(disc_domain_loss, discriminator_domain.trainable_variables)
        discriminator_domain_optimizer.apply_gradients(zip(disc_domain_gradient,
                                                           discriminator_domain.trainable_variables))
        prediction_s = classifier(feature_s, training=True)
        prediction_t = classifier(feature_t, training=True)
        prediction_t_real = classifier(feature_t_real, training=True)
        pred_loss = classifier_loss(prediction_t, label_source)
        pred_loss += classifier_loss(prediction_t_real, label_target)
        encoder_s_loss += classifier_loss(prediction_s, label_target)
        encoder_t_loss += classifier_loss(prediction_t, label_source)
        encoder_s_gradient = tape.gradient(encoder_s_loss, encoder_s.trainable_variables)
        encoder_t_gradient = tape.gradient(encoder_t_loss, encoder_t.trainable_variables)
        classifier_gradient = tape.gradient(pred_loss, classifier.trainable_variables)
        encoder_s_optimizer.apply_gradients(zip(encoder_s_gradient, encoder_s.trainable_variables))
        encoder_t_optimizer.apply_gradients(zip(encoder_t_gradient, encoder_t.trainable_variables))
        classifier_optimizer.apply_gradients(zip(classifier_gradient, classifier.trainable_variables))
    del tape


def train(generator_s,
          generator_t,
          discriminator_t,
          discriminator_s,
          discriminator_domain,
          encoder_s,
          encoder_t,
          classifier,
          source_train_ds,
          target_train_ds,
          source_test_ds,
          target_test_ds,
          epochs):
    # block1: encode and classify
    for epoch in range(epochs):
        print('=====================- block 1 -========================')
        for source_batch in source_train_ds.as_numpy_iterator():
            for target_batch in target_train_ds.as_numpy_iterator():
                train_step_encoder(encoder_s,
                                   encoder_t,
                                   classifier,
                                   source_batch,
                                   target_batch, epoch)
        calculate_acc(source_test_ds, target_test_ds, encoder_s, encoder_t,
                      classifier, epoch)
    for epoch in range(epochs):
        print('=====================- block 2 -========================')
        for source_batch in source_train_ds.as_numpy_iterator():
            for target_batch in target_train_ds.as_numpy_iterator():
                train_step_domain(encoder_s,
                                  encoder_t,
                                  classifier,
                                  discriminator_domain,
                                  source_batch,
                                  target_batch)
        calculate_acc(source_test_ds, target_test_ds, encoder_s, encoder_t, classifier, epoch)
    for epoch in range(epochs):
        print('=====================- block 3 -========================')
        for source_batch in source_train_ds.as_numpy_iterator():
            for target_batch in target_train_ds.as_numpy_iterator():
                train_step_double_GAN(generator_s,
                                      generator_t,
                                      discriminator_t,
                                      discriminator_s,
                                      source_batch,
                                      target_batch)
        calculate_acc(source_test_ds, target_test_ds, encoder_s, encoder_t, classifier, epoch)
        if epoch % 15 == 0:
            generate_and_save_Images(generator_s, epoch, source_test_ds.as_numpy_iterator().next()['data'])
    for epoch in range(epochs):
        print('=====================- block 4 -========================')
        for source_batch in source_train_ds.as_numpy_iterator():
            for target_batch in target_train_ds.as_numpy_iterator():
                train_step_whole(generator_s, generator_t,
                                 discriminator_t, discriminator_s,
                                 discriminator_domain,
                                 encoder_s, encoder_t,
                                 classifier,
                                 source_batch, target_batch)
        calculate_acc(source_test_ds, target_test_ds, encoder_s, encoder_t, classifier, epoch)
        if epoch % 15 == 0:
            generate_and_save_Images(generator_s, epoch, source_test_ds.as_numpy_iterator().next()['data'])
# for HSI in target_test_ds.as_numpy_iterator():
#     feature = encoder_t(HSI['data'], training=False)
#     classify_output = classifier(feature)
#     print(classifier_loss(classify_output, HSI['label']))
