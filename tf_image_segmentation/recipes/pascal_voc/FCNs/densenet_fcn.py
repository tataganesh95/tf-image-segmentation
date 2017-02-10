
import time
import csv
import os
import datetime

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

import keras as K

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint


# http://stackoverflow.com/a/5215012/99379
def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    return datetime.datetime.now().strftime(fmt).format(fname=fname)

dirname = timeStamped('batch_densenet_fcn')
out_dir='/home/ahundt/datasets/tf_image_segmentation_checkpoints/'+dirname+'/'
sess = tf.Session()
K.backend.set_session(sess)

FLAGS = set_paths.FLAGS

checkpoints_dir = FLAGS.checkpoints_dir
log_dir = FLAGS.log_dir + "fcn-8s/"

slim = tf.contrib.slim

if __name__ == '__main__':

    batch_size = 32
    image_train_size = [384, 384, 3]
    number_of_classes = 21
    tfrecord_filename = 'pascal_augmented_train.tfrecords'
    pascal_voc_lut = pascal_segmentation_lut()
    class_labels = pascal_voc_lut.keys()

    densenet_checkpoint = FLAGS.save_dir + 'model_densenet_final.ckpt'

    filename_queue = tf.train.string_input_producer(
        [tfrecord_filename], num_epochs=10)

    image, annotation = read_tfrecord_and_decode_into_image_annotation_pair_tensors(filename_queue)


    tfrecord_val_filename = 'pascal_augmented_val.tfrecords'


    filename_val_queue = tf.train.string_input_producer(
        [tfrecord_filename], num_epochs=1)

    val_image, val_annotation = read_tfrecord_and_decode_into_image_annotation_pair_tensors(filename_val_queue)


    # # Various data augmentation stages
    # image, annotation = flip_randomly_left_right_image_with_annotation(image, annotation)

    # # image = distort_randomly_image_color(image)

    # resized_image, resized_annotation = scale_randomly_image_with_annotation_with_fixed_size_output(image, annotation, image_train_size)


    # resized_annotation = tf.squeeze(resized_annotation)

    # image_batch, annotation_batch = tf.train.shuffle_batch( [resized_image, resized_annotation],
    #                                             batch_size=1,
    #                                             capacity=3000,
    #                                             num_threads=2,
    #                                             min_after_dequeue=1000)


    model = densenet_fc.create_fc_dense_net(number_of_classes,image_train_size)


    model.compile(loss="categorical_crossentropy", optimizer='adam')

    # upsampled_logits_batch, fcn_16s_variables_mapping = FCN_8s(image_batch_tensor=image_batch,
    #                                                         number_of_classes=number_of_classes,
    #                                                         is_training=True)


    # valid_labels_batch_tensor, valid_logits_batch_tensor = get_valid_logits_and_labels(annotation_batch_tensor=annotation_batch,
    #                                                                                     logits_batch_tensor=upsampled_logits_batch,
    #                                                                                     class_labels=class_labels)
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(image)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    tensorboard = TensorBoard(log_dir=out_dir, histogram_freq=10, write_graph=True)
    csv = CSVLogger(out_dir+dirname+'.csv', separator=',', append=True)
    model_checkpoint = ModelCheckpoint(out_dir+'weights.hdf5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')

    start_time = time.time()
    # Fit the model on the batches generated by datagen.flow().
    history = model.fit_generator(datagen.flow(image, annotation,
                        batch_size=batch_size),
                        samples_per_epoch=image.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(val_image, val_annotation),

                        callbacks =[tensorboard,csv,model_checkpoint])

    end_fit_time = time.time()
    average_time_per_epoch = (end_fit_time - start_time) / nb_epoch
    
    model.predict(val_image, batch_size=batch_size, verbose=1)

    end_predict_time = time.time()
    average_time_to_predict = (end_predict_time - end_fit_time) / nb_epoch

    results.append((history, average_time_per_epoch, average_time_to_predict))
    print ('--------------------------------------------------------------------')
    print ('[run_name,batch_size,average_time_per_epoch,average_time_to_predict]')
    print ([dirname,batch_size,average_time_per_epoch,average_time_to_predict])
    print ('--------------------------------------------------------------------')
    
    # Close the Session when we're done.
    sess.close()