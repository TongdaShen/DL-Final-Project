import tensorflow as tf     
import numpy as np
import cv2
import PIL.Image
import time
from skimage import filters
from skimage.color import rgb2gray
     
# Converts a tensor to an image
#   Input: tensor - the tensor to be converted to an image
def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

# Paths for content and style images
content_path = 'cat.png'
style1_path = 'color.jpg'
style2_path = 'messy.jpg'
style3_path = 'mondriaan.jpg'
     
# Load the image from the path and creates tensor
#   Input: path_to_image - the path to the image
#   Return: the tensor representation of the image
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

# Saves file
#   Input: file_name - name of file to be saved/downloaded
def download_file(file_name):
  try:
    from google.colab import files
  except ImportError:
    pass
  else:
    files.download(file_name)
     
# Loads the content and style images
content_image = load_img(content_path)
style1_image = load_img(style1_path)
style2_image = load_img(style2_path)
style3_image = load_img(style3_path)

# Uses a pretrained VGG19 network
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

# Defines the VGG19 layers used for content and style to calculate loss
content_layers = ['block5_conv2'] 

style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

# Builds the model to extract intermediate layers
#   Input: layer_names - the layers of VGG19 to extract
#   Return: the model's outputs at the given layers
def vgg_layers(layer_names):
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False

  outputs = [vgg.get_layer(name).output for name in layer_names]
  model = tf.keras.Model([vgg.input], outputs)
  return model

# Measure style through Gram matrix
#   Input - the tensor to compute the Gram matrix for
def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)

# Build a model that returns the style and content tensors
class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    self.vgg = vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])

    style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

    content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}

    style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}

    return {'content': content_dict, 'style': style_dict}

# Instantiates model that returns the style and content of the given layers
extractor = StyleContentModel(style_layers, content_layers)

counter = 0

# Relative weights for multi-style transfer
relative_weight = [0.5, 0.5]
relative_weight1 = relative_weight[0]
relative_weight2 = relative_weight[1]

# Respective eights used to compute loss
style_weight = 1e-2
content_weight = 1e4
total_variation_weight = 30

# Loss function
#   Input: outputs - result of calling the model
#   Return:   loss - the total loss (weighted sum of style loss and content loss)
def style_content_loss(outputs):
  num_style_layers = len(style_layers)
  num_content_layers = len(content_layers)
  style_outputs = outputs['style']
  for name in style_outputs.keys():
        if name == 'block3_conv1':
          style_outputs[name] = style_outputs[name]*0.2
        if name == 'block4_conv1':
          style_outputs[name] = style_outputs[name]*0.3
        if name == 'block5_conv1':
          style_outputs[name] = style_outputs[name]*0.01
  content_outputs = outputs['content']

  style1_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style1_targets[name])**2) for name in style_outputs.keys()])
  style1_loss *= style_weight / num_style_layers

  style2_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style2_targets[name])**2) for name in style_outputs.keys()])
  style2_loss *= style_weight / num_style_layers

  style3_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style3_targets[name])**2) for name in style_outputs.keys()])
  style3_loss *= style_weight / num_style_layers

  content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                          for name in content_outputs.keys()])
  content_loss *= content_weight / num_content_layers
  loss = (relative_weight1 * style1_loss + relative_weight2 * style2_loss + 
          (1-relative_weight1 - relative_weight2) * style3_loss) + content_loss 
  return loss

while counter <= 4:
  style1_targets = extractor(style1_image)['style']
  style2_targets = extractor(style2_image)['style']
  style3_targets = extractor(style3_image)['style']
  content_targets = extractor(content_image)['content']

  image = tf.Variable(content_image)

  def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

  opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

  @tf.function()
  # Performs a training step
  # Input: image - image being optimized for with respect to loss
  def train_step(image):
    with tf.GradientTape() as tape:
      outputs = extractor(image)
      loss = style_content_loss(outputs)
      loss += total_variation_weight*tf.image.total_variation(image)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))
  
  epochs = 2
  steps_per_epoch = 100

  step = 0
  for n in range(epochs):
    for m in range(steps_per_epoch):
      step += 1
      train_step(image)
      print(".", end='', flush=True)
    print("Train step: {}".format(step))

  file_name = 'stylized-image' + str(counter) + '.jpg'
  tensor_to_image(image).save(file_name)

  download_file(file_name)

  image1 = load_img(file_name)
  image1 = np.squeeze(image1)
  
  image1_size = image1.shape
  image1_height = image1_size[0]
  image1_width = image1_size[1]

  sum = 0
  
  for r in range(0,image1_height):
    for s in range(0,image1_width):
      for t in range(3):
        sum += image1[r][s][t]
  
  relative_weight1 = relative_weight1 - 0.1
  relative_weight2 = relative_weight2 - 0.1
  counter = counter + 1  

# Performs color preserving style transfer by performing a luminance transfer
def color_preserving_style_transfer():
  content_imageC = np.squeeze(content_image)
  imageC = np.squeeze(image)

  content_yuv = cv2.cvtColor(np.float32(content_imageC), cv2.COLOR_RGB2YUV)
  transfer_yuv = cv2.cvtColor(np.float32(imageC), cv2.COLOR_RGB2YUV)
  transfer_yuv[:,:,1:3] = content_yuv[:,:,1:3]
  color_preserved_transfer = cv2.cvtColor(transfer_yuv, cv2.COLOR_YUV2RGB)
  file_name = 'color-preserved-transfer.png'
  tensor_to_image(color_preserved_transfer).save(file_name)    
  download_file(file_name)

color_preserving_style_transfer()








# Performs sharp edges detector and eliminator
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

  mixed_image_size = mixed_image.shape
  mixed_image_height = mixed_image_size[0]
  mixed_image_width = mixed_image_size[1]

  mixed_image = mixed_image.copy()

  for i in range(2,mixed_image_height-1):
    for j in range(2,mixed_image_width-1):
      if grad_mag[i][j] > 0.40:
        for m in range(3):
          a = np.array([
              [mixed_image[i-1][j-1][m],mixed_image[i-1][j][m],mixed_image[i-1][j+1][m]],
              [mixed_image[i][j-1][m],mixed_image[i][j][m],mixed_image[i][j+1][m]],
              [mixed_image[i+1][j-1][m],mixed_image[i+1][j][m],mixed_image[i+1][j+1][m]]
              ])
          x = filters.gaussian(a, sigma=0.5)
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

  download_file(file_name)

  counter2 += 1



     
     
     
     
     
     
# Performs region brightness opmitimation

dict = {'stylized-image0.jpg':[0.5,0.5],'stylized-image1.jpg':[0.4,0.4],
        'stylized-image2.jpg':[0.3,0.3],'stylized-image3.jpg':[0.2,0.2],
        'stylized-image4.jpg':[0.1,0.1],'stylized-image_brightness0.jpg':[0.5,0.5],
        'stylized-image_brightness1.jpg':[0.4,0.4],'stylized-image_brightness2.jpg':[0.3,0.3],
        'stylized-image_brightness3.jpg':[0.2,0.2],'stylized-image_brightness4.jpg':[0.1,0.1]}

for k in range(5):

  chosen_image_path = 'stylized-image' + str(k) + '.jpg'
  relative_weight1 = dict[chosen_image_path][0]
  relative_weight2 = dict[chosen_image_path][1]

  chosen_image = load_img(chosen_image_path)

  chosen_image2 = np.squeeze(chosen_image)

  chosen_image_size = chosen_image2.shape
  chosen_image_height = chosen_image_size[0]
  chosen_image_width = chosen_image_size[1]

  region_height = chosen_image_height//5
  region_width = chosen_image_width//5

  chosen_image2 = chosen_image2.copy()

  for x in range(5):
    for y in range(5):

      relative_weight1 = dict[chosen_image_path][0]
      relative_weight2 = dict[chosen_image_path][1]

      sum = 0

      if sum >= 3*region_height*region_width*0.5 or sum <= 3*region_height*region_width*0.2:

        sum = 0

        for i in range(x*region_height,(x+1)*region_height):
          for j in range(y*region_width,(y+1)*region_width):
            for n in range(3):
              sum += chosen_image2[i][j][n]

        if sum >= 3*region_height*region_width*0.5 or sum <= 3*region_height*region_width*0.3:
          if sum >= 3*region_height*region_width*0.5:
            relative_weight1 = relative_weight1 + 0.90
            relative_weight2 = relative_weight1 - 0.10
          elif sum <= 3*region_height*region_width*0.3:
            relative_weight1 = relative_weight1 - 0.20
            relative_weight2 = relative_weight1 - 0.20

          style1_targets = extractor(style1_image)['style']
          style2_targets = extractor(style2_image)['style']
          style3_targets = extractor(style3_image)['style']
          content_targets = extractor(content_image)['content']

          image = tf.Variable(content_image)

          opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

          style_weight=1e-2
          content_weight=1e4

          total_variation_weight=30
          
          epochs = 1
          steps_per_epoch = 30

          step = 0
          for n in range(epochs):
            for m in range(steps_per_epoch):
              step += 1
              train_step(image)
              print(".", end='', flush=True)
            print("Train step: {}".format(step))

          if x==0 and y == 0:
            file_name = 'stylized-image_brightness' + str(k) + '.jpg'
            tensor_to_image(image).save(file_name)

            new_image_path = 'stylized-image_brightness' + str(k) + '.jpg'
          else:
            file_name = 'stylized-image_brightness_iterate' + str(k) + '.jpg'
            tensor_to_image(image).save(file_name)

            new_image_path = 'stylized-image_brightness_iterate' + str(k) + '.jpg'

          new_image = load_img(new_image_path)

          new_image = np.squeeze(new_image)
          new_image2 = rgb2gray(new_image)
          new_image = new_image.copy()

          for o in range(x*region_height,(x+1)*region_height):
            for p in range(y*region_width,(y+1)*region_width):
              for q in range(3):
                chosen_image2[o][p][q] = new_image[o][p][q]

          file_name = 'stylized-image_brightness' + str(k) + '.jpg'
          tensor_to_image(chosen_image2).save(file_name)

          chosen_image_path = 'stylized-image_brightness' + str(k) + '.jpg'

          chosen_image = load_img(chosen_image_path)

          chosen_image2 = np.squeeze(chosen_image)
          chosen_image2 = chosen_image2.copy()

  download_file(file_name)









# Masked Style Transfer
# 

import os
import numpy as np
# from scipy.misc import imread, imresize, imsave


# function to load masks
def load_mask(mask_path):
    mask = load_img(mask_path)
    mask = np.squeeze(mask)
    mask = rgb2gray(mask)
    print(mask) # Grayscale mask load

    # Perform binarization of mask
    mask[mask <= 0.5] = 0
    mask[mask > 0.5] = 1

    return mask


# function to apply mask to generated image
def mask_content(content, generated, mask):
    width, height, channels = generated.shape
    generated = generated.copy()

    for i in range(width):
        for j in range(height):
            if mask[i, j] == 0:
                generated[i, j, :] = mask[i, j]

    return generated
  
import imageio
from PIL import Image

generated_image = load_img('dog.png')
generated_image = np.squeeze(generated_image)

content_image = load_img('dog.png')
content_image = np.squeeze(content_image)

mask = load_mask('dog.png')

img = mask_content(content_image, generated_image, mask)
file_name = '1.jpg'
tensor_to_image(img).save(file_name)

# Set your style and content target values:
style_image = load_img("color.jpg")
style_targets = extractor(style_image)['style']
content_image = load_img("1.jpg")
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

def style_content_loss1(outputs):
  style_outputs = outputs['style']
  content_outputs = outputs['content']

      # where are those layer weights?

  style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                            for name in style_outputs.keys()])
  style_loss *= style_weight / num_style_layers

  content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                              for name in content_outputs.keys()])
  content_loss *= content_weight / num_content_layers
  loss = style_loss + content_loss 
  return loss

  # Use tf.GradientTape to update the image.
total_variation_weight=30

@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss1(outputs)
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

file_name = '2.jpg'
tensor_to_image(image).save(file_name)

def mask_on_style_transfer(train, pre):
  _, width, height, _ = train.shape
  pre = np.squeeze(pre)
  pre = pre.copy()
  train = np.squeeze(train)
  for i in range(width):
    for j in range(height):
      for c in range(3):
        if pre[i][j][c] == 0:
          pre[i][j][c] = train[i][j][c]

  return pre

img1 = load_img('2.jpg')
img2 = load_img('1.jpg')

result = mask_on_style_transfer(img1,img2)
file_name = '3.jpg'
tensor_to_image(result).save(file_name)

try:
  from google.colab import files
except ImportError:
  pass
else:
  files.download(file_name)
