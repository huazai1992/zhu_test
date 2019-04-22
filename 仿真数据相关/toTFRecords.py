import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
import argparse
import os
from PIL import Image

parser = argparse.ArgumentParser(description="your script description")
parser.add_argument('--data-source')

args = parser.parse_args()
data_source = args.data_source

def getFiles(repodir, repodirext):
    for subdir, dirs, files in os.walk(repodir):
        for file in files:
            if file.find('.jpg') != -1:
                repodirext.append(os.path.join(subdir, file))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

num = [i for i in range(1, 7)]
#size = [224, 448, 224*4, 224*6, 224*8, 224*10]
size = [448, 224*4, 224*6]
files_per_tfrecord = 1000


for i in num:
    if not os.path.exists('./TFRecords{}'.format(i)):
        os.mkdir('./TFRecords{}'.format(i))
    files_set = []
    getFiles(repodir=data_source + str(size[i-1]), repodirext=files_set)
    count = len(files_set) / files_per_tfrecord
    for j in range(count):
        train_filename = './TFRecords{}/train{}_{}.tfrecords'.format(i, i, j)

        with tf.python_io.TFRecordWriter(train_filename) as tfrecord_writer:
            for file in files_set[j*files_per_tfrecord:(j+1)*files_per_tfrecord]:
                # read in image data by tf
                # img_data = tf.gfile.FastGFile(file, 'rb').read()  # image data type is string
                # get width and height of image
                img = Image.open(file)
                img_data = img.tobytes()
                # create features
                feature = {'raw_img': _bytes_feature(img_data)}
                # create example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                # serialize protocol buffer to string
                tfrecord_writer.write(example.SerializeToString())
# tfrecord_writer.close()
