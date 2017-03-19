#!/usr/bin/env python
# coding=utf-8
"""
This is a script for downloading and converting the microsoft coco dataset
from mscoco.org. This can be run as an independent executable to download
the dataset or be imported by scripts used for larger experiments.
"""
from __future__ import division, print_function, unicode_literals
import os
import sys
import zipfile
from collections import defaultdict
from sacred import Experiment, Ingredient
import numpy as np
from PIL import Image
from keras.utils import get_file
from pycocotools.coco import COCO
from tf_image_segmentation.recipes import datasets
from tf_image_segmentation.utils.tf_records import write_image_annotation_pairs_to_tfrecord


# ============== Ingredient 2: dataset =======================
data_coco = Experiment("dataset")


@data_coco.config
def cfg3():
    # TODO(ahundt) add md5 sums for each file
    verbose = True
    dataset_root = os.path.join(os.path.expanduser('~'), '/datasets')
    dataset_path = os.path.join(dataset_root, '/coco')
    urls = [
        'http://msvocds.blob.core.windows.net/coco2014/train2014.zip',
        'http://msvocds.blob.core.windows.net/coco2014/val2014.zip',
        'http://msvocds.blob.core.windows.net/coco2014/test2014.zip',
        'http://msvocds.blob.core.windows.net/coco2015/test2015.zip',
        'http://msvocds.blob.core.windows.net/annotations-1-0-3/instances_train-val2014.zip',
        'http://msvocds.blob.core.windows.net/annotations-1-0-3/person_keypoints_trainval2014.zip',
        'http://msvocds.blob.core.windows.net/annotations-1-0-4/image_info_test2014.zip',
        'http://msvocds.blob.core.windows.net/annotations-1-0-4/image_info_test2015.zip',
        'http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip'
    ]
    data_prefixes = [
        'train2014',
        'val2014',
        'test2014',
        'test2015',
    ]
    image_filenames = [prefix + '.zip' for prefix in data_prefixes]
    annotation_filenames = [
        'instances_train-val2014.zip',
        'image_info_test2014.zip',
        'image_info_test2015.zip',
        'person_keypoints_trainval2014.zip',
        'captions_train-val2014.zip',
    ]
    filenames = []
    filenames.extend(image_filenames)
    filenames.extend(annotation_filenames)
    seg_mask_path = os.path.join(dataset_path, 'seg_mask')
    annotation_paths = [os.path.join(
        dataset_path, '/annotations/instances_%s.json' % prefix) for prefix in data_prefixes]
    seg_mask_paths = [os.path.join(seg_mask_path, prefix) for prefix in data_prefixes]
    tfrecord_filenames = [os.path.join(dataset_path, prefix, '.tfrecords') for prefix in data_prefixes]
    image_dirs = [os.path.join(dataset_path, prefix) for prefix in data_prefixes]


@data_coco.capture
def coco_files(dataset_path, filenames, dataset_root, urls):
    print(dataset_path)
    print(dataset_root)
    print(urls)
    print(filenames)
    return [os.path.join(dataset_path, filename) for filename in filenames]


@data_coco.command
def coco_download(dataset_path, filenames, dataset_root, urls):
    zip_paths = coco_files(dataset_path, filenames, dataset_root, urls)
    for url, filename in zip(urls, filenames):
        path = get_file(filename, url, untar=False, cache_subdir=dataset_path)
        # TODO(ahundt) check if it is already extracted, don't re-extract. see
        # https://github.com/fchollet/keras/issues/5861
        zip_file = zipfile.ZipFile(path, 'r')
        zip_file.extractall(path=dataset_path)
        zip_file.close()


@data_coco.command
def coco_json_to_segmentation(seg_mask_paths, annotation_paths):
    for (seg_mask_path, annFile) in zip(seg_mask_path, annotation_paths):
        coco = COCO(annFile)
        imgToAnns = defaultdict(list)
        for ann in coco.dataset['instances']:
            imgToAnns[ann['image_id']].append(ann)
            # anns[ann['id']] = ann
        for img_num in range(len(imgToAnns.keys())):
            # Both [0]'s are used to extract the element from a list
            img = coco.loadImgs(imgToAnns[imgToAnns.keys()[img_num]][
                0]['image_id'])[0]
            h = img['height']
            w = img['width']
            name = img['file_name']
            root_name = name[:-4]
            MASK = np.zeros((h, w), dtype=np.uint8)
            np.where(MASK > 0)
            for ann in imgToAnns[imgToAnns.keys()[img_num]]:
                mask = coco.annToMask(ann)
                ids = np.where(mask > 0)
                MASK[ids] = ann['category_id']

            im = Image.fromarray(MASK)
            im.save(os.path.join(seg_mask_path, root_name + ".png"))


@data_coco.command
def coco_segmentation_to_tfrecord(tfrecord_filenames, image_dirs,
                                  seg_mask_paths):
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    # Get some image/annotation pairs for example
    for tfrecords_filename, img_dir, mask_dir in zip(tfrecord_filenames, image_dirs, seg_mask_paths):
        img_list = [os.path.join(img_dir, file) for file in os.listdir(img_dir) if file.endswith('.jpg')]
        mask_list = [os.path.join(mask_dir, file) for file in os.listdir(mask_dir) if file.endswith('.png')]
        filename_pairs = zip(img_list, mask_list)
        # You can create your own tfrecords file by providing
        # your list with (image, annotation) filename pairs here
        write_image_annotation_pairs_to_tfrecord(filename_pairs=filename_pairs,
                                                 tfrecords_filename=tfrecords_filename)


@data_coco.command
def coco_setup(dataset_root, dataset_path, data_prefixes,
               filenames, urls, tfrecord_filenames, annotation_paths,
               image_dirs, seg_mask_paths):
    # download the dataset
    coco_download(dataset_path, filenames, dataset_root, urls)
    # convert the relevant files to a more useful format
    coco_json_to_segmentation(seg_mask_paths, annotation_paths)
    coco_segmentation_to_tfrecord(tfrecord_filenames, image_dirs,
                                  seg_mask_paths)


@data_coco.automain
def main(dataset_root, dataset_path, data_prefixes,
         filenames, urls, tfrecord_filenames, annotation_paths,
         image_dirs, seg_mask_paths):
    coco_setup(data_prefixes, dataset_path, filenames, dataset_root, urls,
               tfrecord_filenames, annotation_paths, image_dirs,
               seg_mask_paths)