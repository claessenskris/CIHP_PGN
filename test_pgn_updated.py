from __future__ import print_function
import argparse
from datetime import datetime
import os
import sys
import time
import scipy.misc
import scipy.io as sio
import cv2
from glob import glob
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses most TensorFlow logs except for errors (0 = all logs, 1 = filter out INFO, 2 = additionally filter out WARNING, 3 = additionally filter out ERROR)

import tensorflow as tf
import numpy as np
from PIL import Image
from utils import *

start_time = time.perf_counter()

N_CLASSES = 20
DATA_DIR = './CIHP/CIHP_input/images/' #ToDo should be done better
DATA_ID_LIST = './CIHP/CIHP_input/list/val.txt' #ToDo should be done better
with open(DATA_ID_LIST, 'r') as f:
    NUM_STEPS = len(f.readlines()) 
RESTORE_FROM = './checkpoint/CIHP_pgn'

# resizing of images & saving it back on the location (not ideal, but it works)

All_images_files = DATA_DIR + '*.jpg'
files_image = glob(All_images_files)
for file_image in files_image: # loop over all images
    image = cv2.imread(file_image, cv2.IMREAD_UNCHANGED)
    output = cv2.resize(image, (384, 512), interpolation = cv2.INTER_NEAREST)
    cv2.imwrite(file_image, output)

def main():
    """Create the model and start the evaluation process."""

    # Create queue coordinator.
    coord = tf.train.Coordinator()

    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(DATA_DIR, DATA_ID_LIST, None, False, False, False, coord)
        image = reader.image
        image_rev = tf.reverse(image, tf.stack([1]))
        image_list = reader.image_list

    image_batch = tf.stack([image, image_rev])
    h_orig, w_orig = tf.to_float(tf.shape(image_batch)[1]), tf.to_float(tf.shape(image_batch)[2])
    image_batch050 = tf.image.resize(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 0.50)), tf.to_int32(tf.multiply(w_orig, 0.50))]))
    image_batch075 = tf.image.resize(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 0.75)), tf.to_int32(tf.multiply(w_orig, 0.75))]))
    image_batch125 = tf.image.resize(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 1.25)), tf.to_int32(tf.multiply(w_orig, 1.25))]))
    image_batch150 = tf.image.resize(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 1.50)), tf.to_int32(tf.multiply(w_orig, 1.50))]))
    image_batch175 = tf.image.resize(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 1.75)), tf.to_int32(tf.multiply(w_orig, 1.75))]))
         
    # Create network.
    with tf.compat.v1.variable_scope('', reuse=False):
        net_100 = PGNModel({'data': image_batch}, is_training=False, n_classes=N_CLASSES)
    with tf.compat.v1.variable_scope('', reuse=True):
        net_050 = PGNModel({'data': image_batch050}, is_training=False, n_classes=N_CLASSES)
    with tf.compat.v1.variable_scope('', reuse=True):
        net_075 = PGNModel({'data': image_batch075}, is_training=False, n_classes=N_CLASSES)
    with tf.compat.v1.variable_scope('', reuse=True):
        net_125 = PGNModel({'data': image_batch125}, is_training=False, n_classes=N_CLASSES)
    with tf.compat.v1.variable_scope('', reuse=True):
        net_150 = PGNModel({'data': image_batch150}, is_training=False, n_classes=N_CLASSES)
    with tf.compat.v1.variable_scope('', reuse=True):
        net_175 = PGNModel({'data': image_batch175}, is_training=False, n_classes=N_CLASSES)
    # parsing net

    parsing_out1_050 = net_050.layers['parsing_fc']
    parsing_out1_075 = net_075.layers['parsing_fc']
    parsing_out1_100 = net_100.layers['parsing_fc']
    parsing_out1_125 = net_125.layers['parsing_fc']
    parsing_out1_150 = net_150.layers['parsing_fc']
    parsing_out1_175 = net_175.layers['parsing_fc']

    parsing_out2_050 = net_050.layers['parsing_rf_fc']
    parsing_out2_075 = net_075.layers['parsing_rf_fc']
    parsing_out2_100 = net_100.layers['parsing_rf_fc']
    parsing_out2_125 = net_125.layers['parsing_rf_fc']
    parsing_out2_150 = net_150.layers['parsing_rf_fc']
    parsing_out2_175 = net_175.layers['parsing_rf_fc']

    # combine resize
    parsing_out1 = tf.reduce_mean(tf.stack([tf.image.resize(parsing_out1_050, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize(parsing_out1_075, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize(parsing_out1_100, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize(parsing_out1_125, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize(parsing_out1_150, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize(parsing_out1_175, tf.shape(image_batch)[1:3,])]), axis=0)

    parsing_out2 = tf.reduce_mean(tf.stack([tf.image.resize(parsing_out2_050, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize(parsing_out2_075, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize(parsing_out2_100, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize(parsing_out2_125, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize(parsing_out2_150, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize(parsing_out2_175, tf.shape(image_batch)[1:3,])]), axis=0)

    raw_output = tf.reduce_mean(tf.stack([parsing_out1, parsing_out2]), axis=0)
    head_output, tail_output = tf.unstack(raw_output, num=2, axis=0)
    tail_list = tf.unstack(tail_output, num=20, axis=2)
    tail_list_rev = [None] * 20
    for xx in range(14): tail_list_rev[xx] = tail_list[xx]
    tail_list_rev[14] = tail_list[15]
    tail_list_rev[15] = tail_list[14]
    tail_list_rev[16] = tail_list[17]
    tail_list_rev[17] = tail_list[16]
    tail_list_rev[18] = tail_list[19]
    tail_list_rev[19] = tail_list[18]
    tail_output_rev = tf.stack(tail_list_rev, axis=2)
    tail_output_rev = tf.reverse(tail_output_rev, tf.stack([1]))
    
    raw_output_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
    raw_output_all = tf.expand_dims(raw_output_all, dim=0)
    pred_scores = tf.reduce_max(raw_output_all, axis=3)
    raw_output_all = tf.argmax(raw_output_all, axis=3)
    pred_all = tf.expand_dims(raw_output_all, dim=3) # Create 4-d tensor.

    # Which variables to load.
    restore_var = tf.compat.v1.global_variables()

    # Set up tf session and initialize variables. 
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    init = tf.compat.v1.global_variables_initializer()
    
    sess.run(init)
    sess.run(tf.compat.v1.local_variables_initializer())
    
    # Load weights.
    loader = tf.compat.v1.train.Saver(var_list=restore_var)
    if RESTORE_FROM is not None:
        if load(loader, sess, RESTORE_FROM):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    intermediate_time = time.perf_counter()
    diff_time = intermediate_time - start_time
    print('*** Speed Initialisation =>', diff_time)

    # evaluate prosessing
    parsing_dir = './CIHP/CIHP_output/cihp_parsing_maps' #ToDo should be done better
    if not os.path.exists(parsing_dir): os.makedirs(parsing_dir)

    parsing_vis_dir = './CIHP/CIHP_output/cihp_parsing_maps_vis' #ToDo should be done better
    if not os.path.exists(parsing_vis_dir): os.makedirs(parsing_vis_dir)

    # Iterate over training steps.
    for step in range(NUM_STEPS):
        parsing_, scores = sess.run([pred_all, pred_scores])
        if step % 100 == 0:
            print('step {:d}'.format(step))
        img_split = image_list[step].split('/')
        img_id = img_split[-1][:-4]

        msk = decode_labels(parsing_, num_classes=N_CLASSES)
        parsing_im = Image.fromarray(msk[0])

        parsing_im_resized = parsing_im.resize((768, 1024), Image.LANCZOS) # highest quality of upsampling
        parsing_im_resized.save('{}/{}_vis.png'.format(parsing_vis_dir, img_id))

        parsing_new = parsing_[0, :, :, 0]
        parsing_new_image = Image.fromarray((parsing_new).astype(np.uint8))  # Ensure correct format
        parsing_im_resized = parsing_new_image.resize((768, 1024), Image.LANCZOS)
        parsing_im_resized.save('{}/{}.png'.format(parsing_dir, img_id))

    coord.request_stop()
    coord.join(threads)

    stop_time = time.perf_counter()
    diff_time = stop_time - start_time
    print('*** Speed Overall =>', diff_time)

if __name__ == '__main__':
    main()
