
#!/usr/bin/env python
# coding=utf-8
"""
This is a very basic example of how to use Sacred.
"""
from __future__ import division, print_function, unicode_literals
from sacred import Experiment, Ingredient
import sys
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import pylab
from PIL import Image
from collections import defaultdict
import os
from keras.utils.datautils import get_file
from tf_image_segmentation.recipes import datasets
from tf_image_segmentation.utils.tf_records import write_image_annotation_pairs_to_tfrecord


# ============== Ingredient 2: dataset =======================
data_coco = Ingredient("data_coco", ingredients=[datasets.data_paths, datasets.s])


@data_coco.config
def cfg3(paths):
    dataset_path = paths['base'] + '/coco'
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
    image_filenames = [
        'train2014.zip',
        'val2014.zip',
        'test2014.zip',
        'test2015.zip',
    ]
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


@data_coco.capture
def coco_files(dataset_path, filenames, paths, settings, urls):
    print(dataset_path)
    print(paths)
    print(settings)
    return [dataset_path + filename for filename in filenames]


@data_coco.command
def coco_download(dataset_path, filenames, paths, settings, urls):
    zip_paths = coco_files(dataset_path, filenames, paths, settings, urls)
    for url, filename in zip(urls, filenames):
        get_file(filename, url, untar=True, cache_subdir=dataset_path)


@data_coco.config
def cfg_json_to_segmentation(paths):
    # Modify the following path to point to mscoco directory
    dataDir='/mnt/disk2/mscoco/'
    dataType='val2014'
    # '/mnt/disk2/mscoco/val2014/seg_mask/'
    save_path = os.path.join(data_dir, dataType, 'seg_mask')
    annFile='%s/annotations/instances_%s.json'%(dataDir,dataType)


@data_coco.command
def coco_json_to_segmentation(dataDir, dataType, save_path, annFile):
    coco=COCO(annFile)
    imgToAnns = defaultdict(list)
    for ann in coco.dataset['instances']:
        imgToAnns[ann['image_id']].append(ann)
        anns[ann['id']] = ann
    for img_num in range(len(imgToAnns.keys())):
        # Both [0]'s are used to extract the element from a list
        img = coco.loadImgs(imgToAnns[imgToAnns.keys()[img_num]][0]['image_id'])[0]
        h = img['height']
        w = img['width']
        name = img['file_name']
        root_name = name[:-4]
        MASK = np.zeros((h,w), dtype=np.uint8)
        np.where( MASK > 0 )
        for ann in imgToAnns[imgToAnns.keys()[img_num]]:
            mask = coco.annToMask(ann)
            ids = np.where( mask > 0 )
            MASK[ids] = ann['category_id']

        im = Image.fromarray(MASK)
        im.save(os.path.join(save_path, root_name + ".png"))


@data_coco.config
def cfg_coco_segmentation_to_tfrecord(dataset_path, paths):
    dataset_path = paths['base'] + '/coco'
    # Update the following four paths
    images_dir = '/mnt/disk2/mscoco/train2014/'
    seg_map_dir = '/mnt/disk2/mscoco/train2014/seg_map/'
    # This file contains the list of file names with segmentation maps
    # Note that there is no extension.
    # e.g.
    # COCO_val2014_000000000042
    # COCO_val2014_000000000073
    # COCO_val2014_000000000074
    # COCO_val2014_000000000133
    # COCO_val2014_000000000136
    # ...

    list_file = '/mnt/disk2/mscoco/train2014/seg_map/list.txt'

    tf_records_filename = '/mnt/disk2/mscoco/mscoco_train2014.tfrecords'


@data_coco.command
def coco_segmentation_to_tfrecord(dataset_path, images_dir, seg_map_dir, list_file, tf_records_filename):
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    # Get some image/annotation pairs for example
    filename_pairs = []
    with open(list_file, 'r') as ff:
        fnames = ff.readlines()
        for fname in fnames:
            fname = fname.rstrip('\n')
            pair = (os.path.join(images_dir, fname + '.jpg'), os.path.join(seg_map_dir, fname + '.png'))
            filename_pairs.append(pair)

    # You can create your own tfrecords file by providing
    # your list with (image, annotation) filename pairs here
    write_image_annotation_pairs_to_tfrecord(filename_pairs=filename_pairs,
                                            tfrecords_filename=tf_records_filename)