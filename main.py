import os
import tensorflow as tf
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
     

#import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools


import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature
from skimage.measure import regionprops
import math
from skimage.color import rgb2gray
     

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)
     
##Download images and choose a style image and a content image

content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
style1_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')
style2_path = 'Mondriaan.jpg'
     
##Visualize the input

#Define a function to load an image and limit its maximum dimension to 512 pixels.


def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img
     
#Create a simple function to display an image:


def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)
     

content_image = load_img(content_path)
style1_image = load_img(style1_path)
style2_image = load_img(style2_path)

# plt.subplot(1, 2, 1)
# imshow(content_image, 'Content Image')

# plt.subplot(1, 2, 2)
# imshow(style_image, 'Style Image')
     
## Fast Style Transfer using TF-Hub

# This tutorial demonstrates the original style-transfer algorithm, 
# which optimizes the image content to a particular style. 
# Before getting into the details, let's see how the 
# TensorFlow Hub model does this:


# import tensorflow_hub as hub
# hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
# stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
# tensor_to_image(stylized_image)
     
##Define content and style representations

# Use the intermediate layers of the model to get the content 
# and style representations of the image. Starting from the 
# network's input layer, the first few layer activations represent 
# low-level features like edges and textures. As you step through 
# the network, the final few layers represent higher-level features—object 
# parts like wheels or eyes. In this case, you are using the VGG19 network 
# architecture, a pretrained image classification network. 
# These intermediate layers are necessary to define the representation 
# of content and style from the images. For an input image, try to 
# match the corresponding style and content target representations 
# at these intermediate layers.

#Load a VGG19 and test run it on our image to ensure it's used correctly:


x = tf.keras.applications.vgg19.preprocess_input(content_image*255)
x = tf.image.resize(x, (224, 224))
vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
prediction_probabilities = vgg(x)
prediction_probabilities.shape
     

# predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
# [(class_name, prob) for (number, class_name, prob) in predicted_top_5]
     
#Now load a VGG19 without the classification head, and list the layer names


vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

# print()
# for layer in vgg.layers:
#   print(layer.name)
     
#Choose intermediate layers from the network to represent the style 
# and content of the image:


content_layers = ['block5_conv2'] 

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)
     
##Intermediate layers for style and content

# So why do these intermediate outputs within 
# our pretrained image classification network 
# allow us to define style and content representations?

# At a high level, in order for a network to 
# perform image classification (which this network 
# has been trained to do), it must understand the image. 
# This requires taking the raw image as input pixels 
# and building an internal representation that converts 
# the raw image pixels into a complex understanding of 
# the features present within the image.

# This is also a reason why convolutional neural networks 
# are able to generalize well: they’re able to capture the 
# invariances and defining features within classes 
# (e.g. cats vs. dogs) that are agnostic to background noise 
# and other nuisances. Thus, somewhere between where the raw 
# image is fed into the model and the output classification label, 
# the model serves as a complex feature extractor. 
# By accessing intermediate layers of the model, 
# you're able to describe the content and style of input images.

# Build the model
def vgg_layers(layer_names):
  # Use tf.keras.applications to extract immediate layers
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False

  # Build a model to store those immediate layers  
  outputs = [vgg.get_layer(name).output for name in layer_names]
  model = tf.keras.Model([vgg.input], outputs)
  return model

# Create the model
#style_extractor = vgg_layers(style_layers)
#style1_outputs = style_extractor(style1_image*255)
#style2_outputs = style_extractor(style2_image*255)

#Look at the statistics of each layer's output
# for name, output in zip(style_layers, style_outputs):
#   print(name)
#   print("  shape: ", output.numpy().shape)
#   print("  min: ", output.numpy().min())
#   print("  max: ", output.numpy().max())
#   print("  mean: ", output.numpy().mean())
#   print()

# Calculate Style by using Gram-matrix.
def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)

# Build a model that returns the style and content tensors.

class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    self.vgg = vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    "Expects float input in [0,1]"
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                      outputs[self.num_style_layers:])

    style_outputs = [gram_matrix(style_output)
                     for style_output in style_outputs]

    content_dict = {content_name: value
                    for content_name, value
                    in zip(self.content_layers, content_outputs)}

    style_dict = {style_name: value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}

    return {'content': content_dict, 'style': style_dict}

# When called on an image, this model returns the gram matrix (style) of the style_layers and content of the content_layers:

extractor = StyleContentModel(style_layers, content_layers)

results = extractor(tf.constant(content_image))

# print('Styles:')
# for name, output in sorted(results['style'].items()):
#   print("  ", name)
#   print("    shape: ", output.numpy().shape)
#   print("    min: ", output.numpy().min())
#   print("    max: ", output.numpy().max())
#   print("    mean: ", output.numpy().mean())
#   print()

# print("Contents:")
# for name, output in sorted(results['content'].items()):
#   print("  ", name)
#   print("    shape: ", output.numpy().shape)
#   print("    min: ", output.numpy().min())
#   print("    max: ", output.numpy().max())
#   print("    mean: ", output.numpy().mean())

counter = 0
relative_weight = 0.9


while counter <= 4:

  # Set your style and content target values:
  style1_targets = extractor(style1_image)['style']
  style2_targets = extractor(style2_image)['style']
  content_targets = extractor(content_image)['content']

  # Define a tf.Variable to contain the image to optimize. To make this quick, initialize it with the content image (the tf.Variable must be the same shape as the content image):
  image = tf.Variable(content_image)

  # Since this is a float image, define a function to keep the pixel values between 0 and 1:
  def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

  #Create an optimizer. The paper recommends LBFGS, but Adam works okay, too:
  opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

  #To optimize this, use a weighted combination of the two losses to get the total loss:
  style_weight=1e-2
  content_weight=1e4

  def style_content_loss(outputs):
      style_outputs = outputs['style']
      content_outputs = outputs['content']
      style1_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style1_targets[name])**2) 
                            for name in style_outputs.keys()])
      style1_loss *= style_weight / num_style_layers

      style2_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style2_targets[name])**2) 
                            for name in style_outputs.keys()])
      style2_loss *= style_weight / num_style_layers

      content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                              for name in content_outputs.keys()])
      content_loss *= content_weight / num_content_layers
      loss = (relative_weight * style1_loss + (1 - relative_weight) * style2_loss) + content_loss 
      return loss

  # Use tf.GradientTape to update the image.
  total_variation_weight=30

  @tf.function()
  def train_step(image):
    with tf.GradientTape() as tape:
      outputs = extractor(image)
      loss = style_content_loss(outputs)
      loss += total_variation_weight*tf.image.total_variation(image)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))

  # Now run a few steps to test:
  # train_step(image)
  # train_step(image)
  # train_step(image)
  # tensor_to_image(image)

  # Since it's working, perform a longer optimization:
  import time
  start = time.time()
  
  epochs = 2
  steps_per_epoch = 100

  step = 0
  for n in range(epochs):
    for m in range(steps_per_epoch):
      step += 1
      train_step(image)
      print(".", end='', flush=True)
    #display.clear_output(wait=True)
    #display.display(tensor_to_image(image))
    print("Train step: {}".format(step))

  end = time.time()
  print("Total time: {:.1f}".format(end-start))

  file_name = 'stylized-image' + str(counter) + '.jpg'
  tensor_to_image(image).save(file_name)

  try:
    from google.colab import files
  except ImportError:
    pass
  else:
    files.download(file_name)
  
  relative_weight = relative_weight - 0.2
  counter = counter + 1  

import cv2

content_imageC = np.squeeze(content_image)
imageC = np.squeeze(image)

content_yuv = cv2.cvtColor(np.float32(content_imageC), cv2.COLOR_RGB2YUV)
transfer_yuv = cv2.cvtColor(np.float32(imageC), cv2.COLOR_RGB2YUV)
transfer_yuv[:,:,1:3] = content_yuv[:,:,1:3]
color_preserved_transfer = cv2.cvtColor(transfer_yuv, cv2.COLOR_YUV2RGB)
file_name = 'color-preserved-transfer.png'
tensor_to_image(color_preserved_transfer).save(file_name)    

# Sharp edges detector and eliminator

counter2 = 0

while counter2 <= 4:

  mixed_image_path = 'stylized-image' + str(counter2) + '.jpg'

  mixed_image = load_img(mixed_image_path)

  mixed_image = np.squeeze(mixed_image)
  mixed_image2 = rgb2gray(mixed_image)

  dx = filters.sobel_v(mixed_image2)
  dy = filters.sobel_h(mixed_image2)

  grad_mag = np.sqrt(dx ** 2 + dy ** 2)
  grad_ori = np.arctan2(dy, dx)

  print(grad_mag.shape)

  mixed_image_size = mixed_image.shape
  mixed_image_height = mixed_image_size[0]
  mixed_image_width = mixed_image_size[1]

  print(mixed_image_size)
  print(mixed_image.shape)

  mixed_image = mixed_image.copy()

  for i in range(2,mixed_image_height-1):
    for j in range(2,mixed_image_width-1):
      if grad_mag[i][j] > 0.3:
        for m in range(3):
          a = np.array([
              [mixed_image[i-1][j-1][m],mixed_image[i-1][j][m],mixed_image[i-1][j+1][m]],
              [mixed_image[i][j-1][m],mixed_image[i][j][m],mixed_image[i][j+1][m]],
              [mixed_image[i+1][j-1][m],mixed_image[i+1][j][m],mixed_image[i+1][j+1][m]]
              ])
          x = filters.gaussian(a, sigma=0.4)
          if m == 0:
            mixed_image[i-1][j-1] = [x[0][0],mixed_image[i-1][j-1][1],mixed_image[i-1][j-1][2]]
            mixed_image[i-1][j] = [x[0][1],mixed_image[i-1][j][1],mixed_image[i-1][j][2]]
            mixed_image[i-1][j+1] = [x[0][2],mixed_image[i-1][j+1][1],mixed_image[i-1][j+1][2]]
            mixed_image[i][j-1] = [x[1][0],mixed_image[i][j-1][1],mixed_image[i][j-1][2]]
            mixed_image[i][j] = [x[1][1],mixed_image[i][j][1],mixed_image[i][j][2]]
            mixed_image[i][j+1] = [x[1][2],mixed_image[i][j+1][1],mixed_image[i][j+1][2]]
            mixed_image[i+1][j-1] = [x[2][0],mixed_image[i+1][j-1][1],mixed_image[i+1][j-1][2]]
            mixed_image[i+1][j] = [x[2][1],mixed_image[i+1][j][1],mixed_image[i+1][j][2]]
            mixed_image[i+1][j+1] = [x[2][2],mixed_image[i+1][j+1][1],mixed_image[i+1][j+1][2]]
          if m == 1:
            mixed_image[i-1][j-1] = [mixed_image[i-1][j-1][0],x[0][0],mixed_image[i-1][j-1][2]]
            mixed_image[i-1][j] = [mixed_image[i-1][j][0],x[0][1],mixed_image[i-1][j][2]]
            mixed_image[i-1][j+1] = [mixed_image[i-1][j+1][0],x[0][2],mixed_image[i-1][j+1][2]]
            mixed_image[i][j-1] = [mixed_image[i][j-1][0],x[1][0],mixed_image[i][j-1][2]]
            mixed_image[i][j] = [mixed_image[i][j][0],x[1][1],mixed_image[i][j][2]]
            mixed_image[i][j+1] = [mixed_image[i][j+1][0],x[1][2],mixed_image[i][j+1][2]]
            mixed_image[i+1][j-1] = [mixed_image[i+1][j-1][0],x[2][0],mixed_image[i+1][j-1][2]]
            mixed_image[i+1][j] = [mixed_image[i+1][j][0],x[2][1],mixed_image[i+1][j][2]]
            mixed_image[i+1][j+1] = [mixed_image[i+1][j+1][0],x[2][2],mixed_image[i+1][j+1][2]]
          if m == 2:
            mixed_image[i-1][j-1] = [mixed_image[i-1][j-1][0],mixed_image[i-1][j-1][1],x[0][0]]
            mixed_image[i-1][j] = [mixed_image[i-1][j][0],mixed_image[i-1][j][1],x[0][1]]
            mixed_image[i-1][j+1] = [mixed_image[i-1][j+1][0],mixed_image[i-1][j+1][1],x[0][2]]
            mixed_image[i][j-1] = [mixed_image[i][j-1][0],mixed_image[i][j-1][1],x[1][0]]
            mixed_image[i][j] = [mixed_image[i][j][0],mixed_image[i][j][1],x[1][1]]
            mixed_image[i][j+1] = [mixed_image[i][j+1][0],mixed_image[i][j+1][1],x[1][2]]
            mixed_image[i+1][j-1] = [mixed_image[i+1][j-1][0],mixed_image[i+1][j-1][1],x[2][0]]
            mixed_image[i+1][j] = [mixed_image[i+1][j][0],mixed_image[i+1][j][1],x[2][1]]
            mixed_image[i+1][j+1] = [mixed_image[i+1][j+1][0],mixed_image[i+1][j+1][1],x[2][2]]

  file_name = 'stylized-image_no_sharp_edge' + str(counter2) + '.jpg'
  tensor_to_image(mixed_image).save(file_name)

  try:
    from google.colab import files
  except ImportError:
    pass
  else:
    files.download(file_name)

  counter2 += 1

# Region Brightness Opmitimation

mixed_image_path = 'stylized-image4.jpg'

mixed_image = load_img(mixed_image_path)

mixed_image = np.squeeze(mixed_image)
mixed_image2 = rgb2gray(mixed_image)

mixed_image_size = mixed_image.shape
mixed_image_height = mixed_image_size[0]
mixed_image_width = mixed_image_size[1]

region_height = mixed_image_height//10
region_width = mixed_image_width//10

sum = 0

for i in range(0,region_height):
  for j in range(0,region_width):
    for n in range(3):
      sum += mixed_image[i][j][n]

if sum >= 3*region_height*region_width*0.5:
  relative_weight = 0.9-0.3
  # Set your style and content target values:
  style1_targets = extractor(style1_image)['style']
  style2_targets = extractor(style2_image)['style']
  content_targets = extractor(content_image)['content']

  # Define a tf.Variable to contain the image to optimize. To make this quick, initialize it with the content image (the tf.Variable must be the same shape as the content image):
  image = tf.Variable(content_image)

  #Create an optimizer. The paper recommends LBFGS, but Adam works okay, too:
  opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

  #To optimize this, use a weighted combination of the two losses to get the total loss:
  style_weight=1e-2
  content_weight=1e4

  # Use tf.GradientTape to update the image.
  total_variation_weight=30

  # Since it's working, perform a longer optimization:
  import time
  start = time.time()
  
  epochs = 2
  steps_per_epoch = 100

  step = 0
  for n in range(epochs):
    for m in range(steps_per_epoch):
      step += 1
      train_step(image)
      print(".", end='', flush=True)
    #display.clear_output(wait=True)
    #display.display(tensor_to_image(image))
    print("Train step: {}".format(step))

  end = time.time()
  print("Total time: {:.1f}".format(end-start))

  file_name = 'stylized-image_brightness.jpg'
  tensor_to_image(image).save(file_name)

  try:
    from google.colab import files
  except ImportError:
    pass
  else:
    files.download(file_name)

  new_image_path = 'stylized-image_brightness.jpg'

  new_image = load_img(mixed_image_path)

  new_image = np.squeeze(mixed_image)
  new_image2 = rgb2gray(mixed_image)

  mixed_image = mixed_image.copy()

  for o in range(0,region_height):
    for p in range(0,region_width):
      for q in range(3):
        mixed_image[o][p][q] = new_image[o][p][q]

  file_name = 'stylized-image_brightness_new.jpg'
  tensor_to_image(mixed_image).save(file_name)

  try:
    from google.colab import files
  except ImportError:
    pass
  else:
    files.download(file_name)