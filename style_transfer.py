import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
CHANNELS = 3

CONTENT_WEIGHT = 0.001
STYLE_WEIGHT = 1.0
TOTAL_VARIATION_WEIGHT = 1.0

VGG_MEAN = [103.939, 116.779, 123.68]

CONTENT_LAYER = "block5_conv2"
STYLE_LAYERS= ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]

EPOCHS = 10
STEPS_PER_EPOCH = 50

def preprocess_image(path):
    image = tf.keras.preprocessing.image.load_img(path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, 0)
    image = tf.keras.applications.vgg19.preprocess_input(image)
    return tf.convert_to_tensor(image)

def postprocess_image(image):
    image = image.reshape(())
    image = image + VGG_MEAN
    image = image[:, :, (2, 1, 0)]
    image = np.clip(image, 0, 255).astype("uint8")
    return image

def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram

def content_loss(content, transfer):
    return CONTENT_WEIGHT * tf.math.reduce_sum(tf.math.square(transfer - content))

def style_loss(style, transfer):
    style_gram = gram_matrix(style)
    transfer_gram = gram_matrix(transfer)
    normalization_factor = 4.0 * (CHANNELS ** 2) * ((IMAGE_HEIGHT * IMAGE_WIDTH) ** 2)
    return STYLE_WEIGHT * (tf.math.reduce_sum(tf.math.square(transfer_gram - style_gram)) / normalization_factor)

def total_variation_loss(transfer):
    return TOTAL_VARIATION_WEIGHT * tf.image.total_variation(transfer)

def vgg19_model(x):
    vgg19 = tf.keras.applications.vgg19.VGG19(input_tensor=x, include_top=False, weights="imagenet")
    vgg19.trainable = False
    vgg19.summary()
    inputs = vgg19.inputs
    outputs = dict([(layer.name, layer.output) for layer in vgg19.layers])
    return tf.keras.Model(inputs, outputs)

def loss_function(content, style, transfer):
    loss = 0
    features = vgg19_model(tf.concat(content, style, transfer), 0)
    content_layer_features = features[CONTENT_LAYER]
    loss = loss + content_loss(content_layer_features[0], content_layer_features[2])
    for layer in STYLE_LAYERS:
        style_layer_features = features[layer]
        loss = loss + style_loss(style_layer_features[1], style_layer_features[2])
    loss = loss + total_variation_loss(transfer)
    return loss

def train_step(content, style, transfer):
    image = tf.Variable(transfer)
    with tf.GradientTape() as tape:
        loss = loss_function(content, style, transfer)
    gradients = tape.gradients(loss, image)
    optimizer = tf.keras.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    optimizer.apply_gradients([gradients, image])
    image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))

def train(content, style, transfer):
    for _ in EPOCHS:
        for _ in STEPS_PER_EPOCH:
            train_step(content, style, transfer)
    image = postprocess_image(transfer)
    plt.imsave("style-transfer.jpg", image)

def main():
    # Call preprocess_image function
    # Call train function
