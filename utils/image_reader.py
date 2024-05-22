import os

import numpy as np
import tensorflow as tf
import random

IGNORE_LABEL = 255
IMG_MEAN = np.array((125.0, 114.4, 107.9), dtype=np.float32)

def image_scaling(img):
    """
    Randomly scales the images between 0.5 to 1.5 times the original size.
    Args:
      img: Training image to scale.
      label: Segmentation mask to scale.
    """
    
    scale = tf.random_uniform([1], minval=0.5, maxval=2.0, dtype=tf.float32, seed=None)
    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[0]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))
    new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
    img = tf.image.resize(img, new_shape)
    return img

def image_mirroring(img):
    """
    Randomly mirrors the images.
    Args:
      img: Training image to mirror.
      label: Segmentation mask to mirror.
    """
    
    distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    img = tf.reverse(img, mirror)
    return img

def random_resize_img_labels(image,resized_h, resized_w):

    scale = tf.random_uniform([1], minval=0.75, maxval=1.25, dtype=tf.float32, seed=None)
    h_new = tf.to_int32(tf.multiply(tf.to_float(resized_h), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(resized_w), scale))

    new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
    img = tf.image.resize_images(image, new_shape)
    return img

def resize_img_labels(image,resized_h, resized_w):

    new_shape = tf.stack([tf.to_int32(resized_h), tf.to_int32(resized_w)])
    img = tf.image.resize_images(image, new_shape)
    return img

def random_crop_and_pad_image_and_labels(image, crop_h, crop_w, ignore_label=255):
    """
    Randomly crop and pads the input images.
    Args:
      image: Training image to crop/ pad.
      label: Segmentation mask to crop/ pad.
      crop_h: Height of cropped segment.
      crop_w: Width of cropped segment.
      ignore_label: Label to ignore during the training.
    """

    image_shape = tf.shape(image)
    last_image_dim = tf.shape(image)[-1]
    img_crop = combined_crop[:, :, :last_image_dim]
    img_crop.set_shape((crop_h, crop_w, 3))

    return img_crop

def read_labeled_image_reverse_list(data_dir, data_list):
    """Reads txt file containing paths to images and ground truth masks.
    
    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
       
    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    f = open(data_list, 'r')
    images = []
    masks = []
    masks_rev = []
    for line in f:
        try:
            image, mask, mask_rev = line.strip("\n").split(' ')
        except ValueError: # Adhoc for test.
            image = mask = mask_rev = line.strip("\n")
        images.append(data_dir + image)
    return images

def read_labeled_image_list(data_dir, data_list):
    """Reads txt file containing paths to images and ground truth masks.
    
    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
       
    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    f = open(data_list, 'r')
    images = []
    for line in f:
        image = line.strip("\n")
        images.append(data_dir + image)
    return images

def read_edge_list(data_dir, data_id_list):
    f = open(data_id_list, 'r')
    edges = []
    for line in f:
        edge = line.strip("\n")
        edges.append(data_dir + '/edges/' + edge + '.png')
    return edges

def read_images_from_disk(input_queue, input_size, random_scale, random_mirror=False): # optional pre-processing arguments
    """Read one image and its corresponding mask with optional pre-processing.
    
    Args:
      input_queue: tf queue with paths to the image and its mask.
      input_size: a tuple with (height, width) values.
                  If not given, return images of original size.
      random_scale: whether to randomly scale the images prior
                    to random crop.
      random_mirror: whether to randomly mirror the images prior
                    to random crop.
      
    Returns:
      Two tensors: the decoded image and its mask.
    """

    img_contents = tf.io.read_file(input_queue[0])

    img = tf.image.decode_jpeg(img_contents, channels=3)
    img_r, img_g, img_b = tf.split(value=img, num_or_size_splits=3, axis=2)
    img = tf.cast(tf.concat([img_b, img_g, img_r], 2), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN

    if input_size is not None:
        h, w = input_size

        # Randomly scale the images
        if random_scale: img= image_scaling(img)

        # Randomly mirror the images
        if random_mirror: img = image_mirroring(img)

        # Randomly crops the images and labels.
        img = random_crop_and_pad_image_and_labels(img, h, w, IGNORE_LABEL)

    return img

class ImageReader(object):
    '''Generic ImageReader which reads images and corresponding segmentation
       masks from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, data_dir, data_id_list, input_size, random_scale,
                 random_mirror, shuffle, coord):
        '''Initialise an ImageReader.
        
        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
          data_id_list: path to the file of image id.
          input_size: a tuple with (height, width) values, to which all the images will be resized.
          random_scale: whether to randomly scale the images prior to random crop.
          random_mirror: whether to randomly mirror the images prior to random crop.
          coord: TensorFlow queue coordinator.
        '''
        self.data_dir = data_dir
        self.data_id_list = data_id_list
        self.input_size = input_size
        self.coord = coord
    
        self.image_list = read_labeled_image_list(self.data_dir, self.data_id_list)
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.queue = tf.compat.v1.train.slice_input_producer([self.images],shuffle=shuffle)
        self.image = read_images_from_disk(self.queue, self.input_size, random_scale, random_mirror)

    def dequeue(self, num_elements):

        image_batch = tf.train.batch([self.image], num_elements)
        return image_batch
