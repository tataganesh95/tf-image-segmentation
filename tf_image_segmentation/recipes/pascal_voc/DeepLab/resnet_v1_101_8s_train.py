import tensorflow as tf
import numpy as np
import skimage.io as io
import os, sys
from matplotlib import pyplot as plt

sys.path.append(os.path.join(os.path.join(os.getcwd(), "../../../../../tf-image-segmentation")))
from tf_image_segmentation.utils import set_paths # Sets appropriate paths and provides access to log_dir and checkpoint_path via FLAGS

FLAGS = set_paths.FLAGS

checkpoints_dir = FLAGS.checkpoints_dir
log_dir = os.path.join(FLAGS.log_dir, "deeplab/")

slim = tf.contrib.slim
resnet_101_v1_checkpoint_path = os.path.join(checkpoints_dir, 'resnet_v1_101.ckpt')
# resnet_101_v1_checkpoint_path = os.path.join(checkpoints_dir, 'model_resnet_101_8s_epoch_240822.ckpt')
print(resnet_101_v1_checkpoint_path)
if not os.path.isfile(resnet_101_v1_checkpoint_path) :
    import tf_image_segmentation.utils.download_ckpt as dl_ckpt
    dl_ckpt.download_ckpt('http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz')

from tf_image_segmentation.utils.tf_records import read_tfrecord_and_decode_into_image_annotation_pair_tensors
from tf_image_segmentation.models.resnet_v1_101_8s import resnet_v1_101_8s, extract_resnet_v1_101_mapping_without_logits

from tf_image_segmentation.utils.pascal_voc import pascal_segmentation_lut

from tf_image_segmentation.utils.training import get_valid_logits_and_labels

from tf_image_segmentation.utils.augmentation import (distort_randomly_image_color,
                                                      flip_randomly_left_right_image_with_annotation,
                                                      scale_randomly_image_with_annotation_with_fixed_size_output)

dataset = 'pascal_aug'
image_train_size = [384, 384]
if dataset == 'mscoco':
	number_of_classes = 81
	num_training_images = 82081
	train_dataset = 'mscoco_train2014.tfrecords'
	class_labels = [i for i in range(number_of_classes)] + [255]
elif dataset == 'pascal_aug':
	number_of_classes = 21
	num_training_images = 11127
	train_dataset = 'pascal_augmented_train.tfrecords'
	pascal_voc_lut = pascal_segmentation_lut()
	class_labels = pascal_voc_lut.keys()

num_epochs = 20
tfrecord_filename = os.path.join(FLAGS.data_dir, train_dataset)

print(tfrecord_filename)
#import sys; sys.exit()


filename_queue = tf.train.string_input_producer(
    [tfrecord_filename], num_epochs=num_epochs)

image, annotation = read_tfrecord_and_decode_into_image_annotation_pair_tensors(filename_queue)

# Various data augmentation stages
image, annotation = flip_randomly_left_right_image_with_annotation(image, annotation)

# image = distort_randomly_image_color(image)

resized_image, resized_annotation = scale_randomly_image_with_annotation_with_fixed_size_output(image, annotation, image_train_size)


resized_annotation = tf.squeeze(resized_annotation)

image_batch, annotation_batch = tf.train.shuffle_batch( [resized_image, resized_annotation],
                                             batch_size=1,
                                             capacity=3000,
                                             num_threads=2,
                                             min_after_dequeue=1000)
upsampled_logits_batch, resnet_v1_101_variables_mapping = resnet_v1_101_8s(image_batch_tensor=image_batch,
                                                           number_of_classes=number_of_classes,
                                                           is_training=False)

valid_labels_batch_tensor, valid_logits_batch_tensor = get_valid_logits_and_labels(annotation_batch_tensor=annotation_batch,
                                                                                     logits_batch_tensor=upsampled_logits_batch,
                                                                                    class_labels=class_labels)



cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=valid_logits_batch_tensor,
                                                          labels=valid_labels_batch_tensor)

# Normalize the cross entropy -- the number of elements
# is different during each step due to mask out regions
cross_entropy_sum = tf.reduce_mean(cross_entropies)

pred = tf.argmax(upsampled_logits_batch, dimension=3)

probabilities = tf.nn.softmax(upsampled_logits_batch)


with tf.variable_scope("adam_vars"):
    lr_rate = tf.placeholder(tf.float32, shape=[])
    train_step = tf.train.AdamOptimizer(learning_rate=lr_rate).minimize(cross_entropy_sum)


# Variable's initialization functions
resnet_v1_101_without_logits_variables_mapping = extract_resnet_v1_101_mapping_without_logits(resnet_v1_101_variables_mapping)

# print(resnet_v1_101_without_logits_variables_mapping)

init_fn = slim.assign_from_checkpoint_fn(model_path=resnet_101_v1_checkpoint_path,
                                         var_list=resnet_v1_101_without_logits_variables_mapping)

global_vars_init_op = tf.global_variables_initializer()

tf.summary.scalar('cross_entropy_loss', cross_entropy_sum)

merged_summary_op = tf.summary.merge_all()

summary_string_writer = tf.summary.FileWriter(log_dir)

# Create the log folder if doesn't exist yet
if not os.path.exists(log_dir):
     os.makedirs(log_dir)
if not os.path.exists(FLAGS.save_dir):
     os.makedirs(FLAGS.save_dir)

#The op for initializing the variables.
local_vars_init_op = tf.local_variables_initializer()

combined_op = tf.group(local_vars_init_op, global_vars_init_op)

# We need this to save only model variables and omit
# optimization-related and other variables.
model_variables = slim.get_model_variables()
saver = tf.train.Saver(model_variables)


with tf.Session()  as sess:

    sess.run(combined_op)
    init_fn(sess)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # 10 epochs
    for i in xrange(num_training_images * num_epochs):
        feed_dict = {lr_rate: np.asarray( 0.0001 * ( (1 - i/(num_training_images*num_epochs))**0.9)   )}
        #feed_dict = {lr_rate: np.asarray( 0.000001 )}
        cross_entropy, summary_string, _ = sess.run([ cross_entropy_sum,
                                                      merged_summary_op,
                                                      train_step ], feed_dict=feed_dict)
        print("Iteration: " + str(i) + " Current loss: " + str(cross_entropy))

        summary_string_writer.add_summary(summary_string, i)

        if i % num_training_images == 0:
            save_path = saver.save(sess, os.path.join(FLAGS.save_dir,  "model_resnet_101_8s_epoch_" + str(i) + ".ckpt"))
            print("Model saved in file: %s" % save_path)


    coord.request_stop()
    coord.join(threads)

    save_path = saver.save(sess, os.path.join(FLAGS.save_dir, "model_resnet_101_8s.ckpt"))
    print("Model saved in file: %s" % save_path)

summary_string_writer.close()
