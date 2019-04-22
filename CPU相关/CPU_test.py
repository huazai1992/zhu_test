import tensorflow as tf
import time
import argparse

size = [224*2, 224*4, 224*6, 224*8, 224*10]

def read_data(file_str):
    file_path_queue = tf.train.string_input_producer(file_str)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_path_queue)

    return serialized_example



def parse_custom_proto(example_serialized):

  feature_map = {
      'raw_img': tf.FixedLenFeature([], tf.string, default_value=''),
  }

  features = tf.parse_single_example(example_serialized, feature_map)


  return features['raw_img']


def decode_raw(image_buffer, scope=None):
    with tf.name_scope(scope or 'decode_raw'):
        image = tf.decode_raw(image_buffer, tf.uint8)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    return image

def preprocess(image_buffer, bbox=None, batch_position=None):
    """Preprocessing image_buffer as a function of its batch position."""

    image = decode_raw(image_buffer)
    # return tf.cast(image, tf.float32)
    return image


def parse_and_preprocess(value, data_dir_index):
    image_buffer = parse_custom_proto(value)
    image = preprocess(image_buffer, None, None)
    # return (label_index, image)
    image = tf.reshape(image, [size[data_dir_index], size[data_dir_index], 3])
    image = tf.image.resize_images(image, [size[data_dir_index], size[data_dir_index]])
    return image

def main(batch_size, loop, data_dir_index , fre, path):

    tf_img = read_data(path)
    img = parse_and_preprocess(tf_img, data_dir_index)

    min_after_dequeue = batch_size + 1
    capacity = min_after_dequeue + 1 * batch_size
    img_batch = tf.train.shuffle_batch([img], batch_size=batch_size, num_threads=1,
                                       capacity=capacity, min_after_dequeue=min_after_dequeue)

    init = tf.initialize_all_variables()

    sess = tf.InteractiveSession()
    sess.run(init)
    threads = tf.train.start_queue_runners(sess=sess)
    with open('./parallel_results/{}-{}-{}-{}.txt'.format(fre, data_dir_index, batch_size, loop),
              'w') as f_out:
        for _ in range(loop):
            start = time.time()
            imgs = sess.run([img_batch])
            end = time.time()
            f_out.write('{}-{} {}'.format(data_dir_index, batch_size, end - start)+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="your script description")
    parser.add_argument('--dir-index')
    parser.add_argument('--batch-size')
    parser.add_argument('--loop')
    parser.add_argument('--fre')

    args = parser.parse_args()
    dir_index = int(args.dir_index)
    batch_size = args.batch_size
    loop = args.loop
    fre = args.fre
    
    path_str = './TFRecords{}/train{}_{}.tfrecords'
    path = [path_str.format(dir_index+1, dir_index+1, i) for i in range(1)]


    main(int(batch_size), int(loop), int(dir_index), int(fre), path)