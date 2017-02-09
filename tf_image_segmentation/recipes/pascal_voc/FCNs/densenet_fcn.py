
import tensorflow as tf
import numpy as np
import skimage.io as io
import os, sys
from PIL import Image
import set_paths


FLAGS = set_paths.FLAGS
sys.path.append(FLAGS.tf_image_seg_dir)
sys.path.append(FLAGS.slim_path)
sys.path.append(FLAGS.slim_path + '/preprocessing')

from tf_image_segmentation.utils.tf_records import read_tfrecord_and_decode_into_image_annotation_pair_tensors

from tf_image_segmentation.utils.pascal_voc import pascal_segmentation_lut

from tf_image_segmentation.utils.training import get_valid_logits_and_labels

from tf_image_segmentation.utils.augmentation import (distort_randomly_image_color,
                                                      flip_randomly_left_right_image_with_annotation,
                                                      scale_randomly_image_with_annotation_with_fixed_size_output)

from tf_image_segmentation.models.densenet_fcn import layers
from tf_image_segmentation.models.densenet_fcn import densenet_fc

FLAGS = set_paths.FLAGS

checkpoints_dir = FLAGS.checkpoints_dir
log_dir = FLAGS.log_dir + "fcn-8s/"

slim = tf.contrib.slim

if __name__ == '__main__':

    image_train_size = [384, 384, 3]
    number_of_classes = 21
    tfrecord_filename = 'pascal_augmented_train.tfrecords'
    pascal_voc_lut = pascal_segmentation_lut()
    class_labels = pascal_voc_lut.keys()

    densenet_checkpoint = FLAGS.save_dir + 'model_densenet_final.ckpt'

    filename_queue = tf.train.string_input_producer(
        [tfrecord_filename], num_epochs=10)

    image, annotation = read_tfrecord_and_decode_into_image_annotation_pair_tensors(filename_queue)

    model = densenet_fc.create_fc_dense_net(number_of_classes,image_train_size)
