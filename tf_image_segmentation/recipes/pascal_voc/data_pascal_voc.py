#!/usr/bin/env python
# coding=utf-8
"""
This is a script for downloading and converting the pascal voc 2012 dataset
and the berkely extended version.

    # original PASCAL VOC 2012
    # wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar # 2 GB

    # berkeley augmented Pascal VOC
    # wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz # 1.3 GB

This can be run as an independent executable to download
the dataset or be imported by scripts used for larger experiments.

If you aren't sure run this to do a full download + conversion setup of the dataset:
   ./data_pascal_voc.py pascal_voc_setup
"""
from __future__ import division, print_function, unicode_literals
from sacred import Ingredient, Experiment
import sys
import numpy as np
from PIL import Image
from collections import defaultdict
import os
from keras.utils import get_file
# from tf_image_segmentation.recipes import datasets
from tf_image_segmentation.utils.tf_records import write_image_annotation_pairs_to_tfrecord
from tf_image_segmentation.utils import pascal_voc
from os.path import expanduser

# ============== Ingredient 0: settings =================
s = Ingredient("settings")


@s.config
def cfg1():
    verbose = True


# ============== Ingredient 1: dataset.paths =================
data_paths = Ingredient("dataset.paths", ingredients=[s])


@data_paths.config
def cfg2(settings):
    v = not settings['verbose']
    base = expanduser("~") + "/datasets"


# ============== Ingredient 2: dataset =======================
data_pascal_voc = Ingredient("dataset", ingredients=[data_paths, s])


@data_pascal_voc.config
def cfg3(paths):
    dataset_path = paths['base'] + '/VOC2012'
    # sys.path.append("tf-image-segmentation/")
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    # based on https://github.com/martinkersner/train-DeepLab

    # original PASCAL VOC 2012
    # wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar # 2 GB
    pascal_root = dataset_path + '/VOCdevkit/VOC2012'

    # berkeley augmented Pascal VOC
    # wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz # 1.3 GB
    pascal_berkeley_root = dataset_path + '/benchmark_RELEASE'
    urls = [
        'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz'
    ]
    filenames = [
        'VOCtrainval_11-May-2012.tar',
        'benchmark.tgz'
    ]


# @data_pascal_voc.capture
# def pascal_voc_files(dataset_path, filenames, paths, settings, urls):
#     print(dataset_path)
#     print(paths)
#     print(settings)
#     return [dataset_path + filename for filename in filenames]


@data_pascal_voc.command
def pascal_voc_download(dataset_path, filenames, paths, urls):
    zip_paths = pascal_voc_files(dataset_path, filenames, paths, settings, urls)
    for url, filename in zip(urls, filenames):
        get_file(filename, url, untar=True, cache_subdir=dataset_path)


# @data_pascal_voc.command
# def convert_pascal_berkeley_augmented_mat_annotations_to_png(pascal_berkeley_root):
#     convert_pascal_berkeley_augmented_mat_annotations_to_png(pascal_berkeley_root)


# @data_pascal_voc.config
# def cfg_pascal_voc_segmentation_to_tfrecord(dataset_path, filenames, paths, settings):
#     tfrecords_train_filename = dataset_path + '/pascal_augmented_train.tfrecords'
#     tfrecords_val_filename = dataset_path + '/pascal_augmented_val.tfrecords'
#     voc_data_subset_mode = 2


# @data_pascal_voc.command
# def pascal_voc_segmentation_to_tfrecord(dataset_path, pascal_root, pascal_berkeley_root,
#                                         voc_data_subset_mode,
#                                         tfrecords_train_filename, tfrecords_val_filename):
#     # Returns a list of (image, annotation) filename pairs (filename.jpg, filename.png)
#     overall_train_image_annotation_filename_pairs, overall_val_image_annotation_filename_pairs = \
#         pascal_voc.get_augmented_pascal_image_annotation_filename_pairs(pascal_root=pascal_root,
#             pascal_berkeley_root=pascal_berkeley_root,
#             mode=voc_data_subset_mode)

#     # You can create your own tfrecords file by providing
#     # your list with (image, annotation) filename pairs here
#     #
#     # this will create a tfrecord in:
#     # tf_image_segmentation/tf_image_segmentation/recipes/pascal_voc/
#     write_image_annotation_pairs_to_tfrecord(filename_pairs=overall_val_image_annotation_filename_pairs,
#                                              tfrecords_filename=tfrecords_val_filename)

#     write_image_annotation_pairs_to_tfrecord(filename_pairs=overall_train_image_annotation_filename_pairs,
#                                              tfrecords_filename=tfrecords_train_filename)


# @data_pascal_voc.command
# def pascal_voc_setup(filenames, dataset_path, pascal_root,
#                      pascal_berkeley_root, paths, settings,
#                      voc_data_subset_mode,
#                      tfrecords_train_filename,
#                      tfrecords_val_filename):
#     # download the dataset
#     pascal_voc_download(dataset_path, filenames,
#                         paths, settings, urls)
#     # convert the relevant files to a more useful format
#     convert_pascal_berkeley_augmented_mat_annotations_to_png(pascal_berkeley_root)
#     pascal_voc_segmentation_to_tfrecord(dataset_path, pascal_root,
#                                         pascal_berkeley_root,
#                                         voc_data_subset_mode,
#                                         tfrecords_train_filename,
#                                         tfrecords_val_filename)


if __name__ == '__main__':
    # Create pascal voc experiment so dataset interaction
    # can be run as its own executable
    ex = Experiment('pascalvoc', ingredients=[data_pascal_voc, data_paths])
    ex.run_commandline()
