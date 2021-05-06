import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
import pandas as pd
import datetime
import math
from scipy import stats
import random

import sklearn.decomposition
from sklearn.cluster import KMeans

# TF allow growth

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# tf.compat.v1.enable_eager_execution()
tf.config.run_functions_eagerly(True)
config = ConfigProto()
config.gpu_options.allow_growth = True

session = InteractiveSession(config=config)


# Change to import my data. No labels?
#(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = pd.read_csv('./data/reduced_BX_Book_Ratings.csv', usecols=['User-ID', 'ISBN', 'Book-Rating']).rename({'User-ID': 'user_id', 'ISBN': 'item_id', 'Book-Rating': 'rating'}, axis=1)
train_images.fillna(0, axis=1, inplace=True)
train_images = train_images.pivot(index='user_id', columns='item_id', values='rating')
train_images.fillna(0, inplace=True)
train_images = train_images.to_numpy()

# Do PCA on the training data
n_components = 4096
train_images = (train_images - 10.0) / 10.0  # Normalize the images to [-1, 1]
np.save('./real_data/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + 'real-data-flattened', train_images)

mu = np.mean(train_images, axis=0)
pca = sklearn.decomposition.PCA(n_components=n_components)
train_set = pca.fit_transform(train_images)


# pad examples
# pad_num = int(math.pow(math.ceil(math.sqrt(train_images.shape[1])), 2) - train_images.shape[1])
# train_images = np.pad(train_images, ((0,0), (0, pad_num)))

# Initialize Kmeans shit
n_clusters = 8
kmeans = KMeans(n_clusters=n_clusters, max_iter=1000, n_jobs=-1)

# Split dataset into the K sets
kmeans = kmeans.fit(train_set)
labels = kmeans.predict(train_set)

l_train_set = [np.ndarray((0,) + train_set.shape[1:]) for i in range(n_clusters)]
for i, k in zip(train_set, labels):
  l_train_set[k] = np.append(l_train_set[k], i.reshape((1, train_set.shape[1])), axis=0)

k_fractions = [l_train_set[i].shape[0] / train_set.shape[0] for i in range(len(l_train_set))]

square_len = int(math.sqrt(n_components))

for i in range(len(l_train_set)):
  l_train_set[i] = l_train_set[i].reshape(l_train_set[i].shape[0], square_len, square_len, 1)
train_shape = train_set.shape
# Choosing to use the PCA set in reshaped form to avoid the curse of dimensionality
train_set = train_set.reshape(train_set.shape[0], square_len, square_len, 1)

BUFFER_SIZE = 60000
BATCH_SIZE = 200

for i in range(len(l_train_set)):
    l_train_set[i] = tf.data.Dataset.from_tensor_slices(l_train_set[i]).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

#train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
# train_images = (train_images - 10.0) / 10.0  # Normalize the images to [-1, 1]

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_set).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Model hyperparameters
leak_slope = 0.2
learning_rate = 0.00002
Beta_1 = 0.5

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 256)))
    assert model.output_shape == (None, square_len/8, square_len/8, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, square_len/4, square_len/4, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, square_len/2, square_len/2, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, square_len, square_len, 1)

    return model


# Initialize K GANs w/ same hyperparameters
generators = [make_generator_model() for i in range(len(l_train_set))]

generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

#plt.imshow(generated_image[0, :, :, 0], cmap='gray')

def make_discriminator_model():
    # Change input shape
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[square_len, square_len, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    # model = tf.keras.Sequential()
    # model.add(layers.Dense(train_shape[1], use_bias=False, input_shape=(train_shape[1],)))
    # model.add(layers.Dense(100, activation='relu'))
    # model.add(layers.Dense(100, activation='relu'))
    # model.add(layers.Dense(10, activation='relu'))
    # model.add(layers.Dense(1))

    return model


# Initialize K GANs w/ same hyperparameters
discriminators = [make_discriminator_model() for i in range(len(l_train_set))]
discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision)

# This method returns a helper function to compute cross entropy loss
# Check that this loss works well for me.
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

generator_optimizers = [tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=Beta_1) for i in range(len(l_train_set))]
discriminator_optimizers = [tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=Beta_1) for i in range(len(l_train_set))]

checkpoint_dir = './synthetic_data_training_checkpoints/iter2'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 50
# Change dim to my dim
noise_dim = 100
num_examples_to_generate = 16

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

log_dir = './gan_logs'
summary_writer = tf.summary.create_file_writer(log_dir+'/fit/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(images, epoch):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    with summary_writer.as_default():
        # tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        # tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        # tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('gen_loss', gen_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)


@tf.function
def k_train_step(batch, epoch, writer, discriminator, generator, discriminator_optimizer, generator_optimizer):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(batch, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    with writer.as_default():
        # tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        # tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        # tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('gen_loss', gen_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)


def train(dataset, epochs):
  print('training')
  for epoch in range(epochs):
    print('epoc {0}'.format(epoch))
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch, epoch)

    # Produce images for the GIF as you go
    # display.clear_output(wait=True)
    # generate_and_save_images(generator,
    #                          epoch + 1,
    #                          seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
#   display.clear_output(wait=True)
#   generate_and_save_images(generator,
#                            epochs,
#                            seed)

def k_train(datasets, discriminators, generators, discriminator_optimizers, generator_optimizers, epochs):
    for i, data, discriminator, generator, discriminator_optimizer, generator_optimizer in zip(range(len(datasets)), datasets, discriminators, generators, discriminator_optimizers, generator_optimizers):
        writer = tf.summary.create_file_writer(log_dir+'/fit/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'clusert_'+str(i))

        for epoch in range(epochs):
            print('epoch {0}'.format(epoch))
            start = time.time()
            for batch in data:
                k_train_step(batch, epoch, writer, discriminator, generator, discriminator_optimizer, generator_optimizer)
            print('Time for epoch {} is {} sec'.format(epoch+1, time.time()-start))


def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.ion()
  plt.show()


# Train kth gan on kth set
k_train(l_train_set, discriminators, generators, discriminator_optimizers, generator_optimizers, EPOCHS)
#checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
train(train_dataset, EPOCHS)


def generate_and_save_synthetic_data(model, test_input):
    pred = np.empty((0,64,64,1))
    for i in range(200, test_input.shape[0], 200):
        pred = np.concatenate((pred, model(test_input[i-200:i], training=False)))
    remainder = test_input.shape[0] % 200
    pred = np.concatenate((pred, model(test_input[-remainder:], training=False)))
    # pred = model(test_input, training=False)
    np.save('./synthetic_data/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '4096-comp-synth-square-shape', pred, allow_pickle=False)
    return pred


def k_generate_and_save_synthetic_data(models, fractions, rand_input):
    pred = np.empty((0,64,64,1))
    s = [sum(fractions[0:i]) for i in range(1, len(fractions)+1)]
    for i in range(200, rand_input.shape[0], 200):
        r = random.random()
        k = 0
        for j in range(len(s)):
            if r <= s[j]:
                k = j
                break
        pred = np.concatenate((pred, models[k](rand_input[i-200:i], training=False)))
    remainder = rand_input.shape[0] % 200

    pred = np.concatenate((pred, models[-1](rand_input[-remainder:], training=False)))
    # pred = model(test_input, training=False)
    np.save('./synthetic_data/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '4096-comp-synth-square-shape', pred, allow_pickle=False)
    return pred

print(train_set.shape[0])
random_input = tf.random.normal([train_set.shape[0], noise_dim])
#synth = generate_and_save_synthetic_data(generator, random_input)
synth = k_generate_and_save_synthetic_data(generators, k_fractions, random_input)
print("{} {}".format(train_set.shape, synth.shape))
#print(stats.entropy(train_set, synth))
# Save pca representation of data
np.save('./pca_data/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '4096-comp-pivot-table-square-shape', train_set)

# Reshape the train_set and synth
train_set =  train_set.reshape((train_set.shape[0], n_components))
synth = synth.reshape((synth.shape[0], n_components))
# Reconstruct train_set and synth into full-dim tensors
train_set_hat = np.dot(train_set[:, :n_components], pca.components_[:n_components, :])
synth_hat = np.dot(synth[:, :n_components], pca.components_[:n_components, :])
train_set_hat += mu
synth_hat += mu
# Save reconstructed tensors
np.save('./pca_data/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + 'reconstructed-pivot-table-flattened', train_set_hat)
np.save('./synthetic_data/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + 'reconstructed-synth', synth_hat)
# Try the entropy thing on the full-dim tensors
print('Entropy of reconstructed sets: {}'.format(stats.entropy(train_set_hat, qk=synth_hat)))
