# encoding: utf8
import argparse
import os
import time
import cProfile
import pstats
import tensorflow as tf

from tensorflow.contrib.data.python.ops import batching
from tensorflow.contrib.data.python.ops import interleave_ops
from tensorflow.contrib.data.python.ops import prefetching_ops
from tensorflow.contrib.data.python.ops import threadpool
from tensorflow.python.framework import function
from tensorflow.python.platform import gfile
from tensorflow.python.layers import utils
from tensorflow.python.client import timeline
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder

def parse_custom_proto(example_serialized):

  feature_map = {
      'raw_img': tf.FixedLenFeature([], tf.string, default_value=''),
  }

  features = tf.parse_single_example(example_serialized, feature_map)


  return features['raw_img']

def parse_example_proto(example_serialized):
  """Parses an Example proto containing a training example of an image.
  The output of the build_image_data.py image preprocessing script is a dataset
  containing serialized Example protocol buffers. Each Example proto contains
  the following fields:
    image/height: 462
    image/width: 581
    image/colorspace: 'RGB'
    image/channels: 3
    image/class/label: 615
    image/class/synset: 'n03623198'
    image/class/text: 'knee pad'
    image/object/bbox/xmin: 0.1
    image/object/bbox/xmax: 0.9
    image/object/bbox/ymin: 0.2
    image/object/bbox/ymax: 0.6
    image/object/bbox/label: 615
    image/format: 'JPEG'
    image/filename: 'ILSVRC2012_val_00041207.JPEG'
    image/encoded: <JPEG encoded string>
  Args:
    example_serialized: scalar Tensor tf.string containing a serialized
      Example protocol buffer.
  Returns:
    image_buffer: Tensor tf.string containing the contents of a JPEG file.
    label: Tensor tf.int32 containing the label.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    text: Tensor tf.string containing the human-readable label.
  """
  # Dense features in Example proto.
  feature_map = {
      'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                              default_value=-1),
      'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
  }
  sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
  # Sparse features in Example proto.
  feature_map.update(
      {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                   'image/object/bbox/ymin',
                                   'image/object/bbox/xmax',
                                   'image/object/bbox/ymax']})

  features = tf.parse_single_example(example_serialized, feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int32)

  xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
  ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
  xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
  ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

  # Note that we impose an ordering of (y, x) just to make life difficult.
  bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

  # Force the variable number of bounding boxes into the shape
  # [1, num_boxes, coords].
  bbox = tf.expand_dims(bbox, 0)
  bbox = tf.transpose(bbox, [0, 2, 1])

  return features['image/encoded'], label, bbox, features['image/class/text']


_RESIZE_METHOD_MAP = {
    'nearest': tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    'bilinear': tf.image.ResizeMethod.BILINEAR,
    'bicubic': tf.image.ResizeMethod.BICUBIC,
    'area': tf.image.ResizeMethod.AREA
}


def get_image_resize_method(resize_method, batch_position=0):
  """Get tensorflow resize method.
  If resize_method is 'round_robin', return different methods based on batch
  position in a round-robin fashion. NOTE: If the batch size is not a multiple
  of the number of methods, then the distribution of methods will not be
  uniform.
  Args:
    resize_method: (string) nearest, bilinear, bicubic, area, or round_robin.
    batch_position: position of the image in a batch. NOTE: this argument can
      be an integer or a tensor
  Returns:
    one of resize type defined in tf.image.ResizeMethod.
  """

  if resize_method != 'round_robin':
    return _RESIZE_METHOD_MAP[resize_method]

  # return a resize method based on batch position in a round-robin fashion.
  resize_methods = list(_RESIZE_METHOD_MAP.values())
  def lookup(index):
    return resize_methods[index]

  def resize_method_0():
    return utils.smart_cond(batch_position % len(resize_methods) == 0,
                            lambda: lookup(0), resize_method_1)

  def resize_method_1():
    return utils.smart_cond(batch_position % len(resize_methods) == 1,
                            lambda: lookup(1), resize_method_2)

  def resize_method_2():
    return utils.smart_cond(batch_position % len(resize_methods) == 2,
                            lambda: lookup(2), lambda: lookup(3))

  # NOTE(jsimsa): Unfortunately, we cannot use a single recursive function here
  # because TF would not be able to construct a finite graph.

  return resize_method_0()


def decode_jpeg(image_buffer, scope=None):  # , dtype=tf.float32):
  """Decode a JPEG string into one 3-D float image Tensor.
  Args:
    image_buffer: scalar string Tensor.
    scope: Optional scope for op_scope.
  Returns:
    3-D float Tensor with values ranging from [0, 1).
  """
  # with tf.op_scope([image_buffer], scope, 'decode_jpeg'):
  # with tf.name_scope(scope, 'decode_jpeg', [image_buffer]):
  with tf.name_scope(scope or 'decode_jpeg'):
    # Decode the string as an RGB JPEG.
    # Note that the resulting image contains an unknown height and width
    # that is set dynamically by decode_jpeg. In other words, the height
    # and width of image is unknown at compile-time.
    image = tf.image.decode_jpeg(image_buffer, channels=3,
                                 fancy_upscaling=False,
                                 dct_method='INTEGER_FAST')

    # image = tf.Print(image, [tf.shape(image)], 'Image shape: ')

    return image

def decode_raw(image_buffer, scope=None):
    with tf.name_scope(scope or 'decode_raw'):
        image = tf.decode_raw(image_buffer, tf.uint8)

    return image

def normalized_image(images):
  # Rescale from [0, 255] to [0, 2]
  images = tf.multiply(images, 1. / 127.5)
  # Rescale to [-1, 1]
  return tf.subtract(images, 1.0)

def eval_image(image,
               height,
               width,
               batch_position,
               resize_method,
               summary_verbosity=0):
  """Get the image for model evaluation.
  We preprocess the image simiarly to Slim, see
  https://github.com/tensorflow/models/blob/master/slim/preprocessing/vgg_preprocessing.py
  Validation images do not have bounding boxes, so to crop the image, we first
  resize the image such that the aspect ratio is maintained and the resized
  height and width are both at least 1.15 times `height` and `width`
  respectively. Then, we do a central crop to size (`height`, `width`).
  Args:
    image: 3-D float Tensor representing the image.
    height: The height of the image that will be returned.
    width: The width of the image that will be returned.
    batch_position: position of the image in a batch, which affects how images
      are distorted and resized. NOTE: this argument can be an integer or a
      tensor
    resize_method: one of the strings 'round_robin', 'nearest', 'bilinear',
      'bicubic', or 'area'.
    summary_verbosity: Verbosity level for summary ops. Pass 0 to disable both
      summaries and checkpoints.
  Returns:
    An image of size (output_height, output_width, 3) that is resized and
    cropped as described above.
  """
  # TODO(reedwm): Currently we resize then crop. Investigate if it's faster to
  # crop then resize.
  with tf.name_scope('eval_image'):
    if summary_verbosity >= 3:
      tf.summary.image(
          'original_image', tf.expand_dims(image, 0))

    shape = tf.shape(image)
    image_height = shape[0]
    image_width = shape[1]
    image_height_float = tf.cast(image_height, tf.float32)
    image_width_float = tf.cast(image_width, tf.float32)

    scale_factor = 1.15

    # Compute resize_height and resize_width to be the minimum values such that
    #   1. The aspect ratio is maintained (i.e. resize_height / resize_width is
    #      image_height / image_width), and
    #   2. resize_height >= height * `scale_factor`, and
    #   3. resize_width >= width * `scale_factor`
    max_ratio = tf.maximum(height / image_height_float,
                           width / image_width_float)
    resize_height = tf.cast(image_height_float * max_ratio * scale_factor,
                            tf.int32)
    resize_width = tf.cast(image_width_float * max_ratio * scale_factor,
                           tf.int32)

    # Resize the image to shape (`resize_height`, `resize_width`)
    image_resize_method = get_image_resize_method(resize_method, batch_position)
    distorted_image = tf.image.resize_images(image,
                                             [resize_height, resize_width],
                                             image_resize_method,
                                             align_corners=False)

    # Do a central crop of the image to size (height, width).
    total_crop_height = (resize_height - height)
    crop_top = total_crop_height // 2
    total_crop_width = (resize_width - width)
    crop_left = total_crop_width // 2
    distorted_image = tf.slice(distorted_image, [crop_top, crop_left, 0],
                               [height, width, 3])

    distorted_image.set_shape([height, width, 3])
    if summary_verbosity >= 3:
      tf.summary.image(
          'cropped_resized_image', tf.expand_dims(distorted_image, 0))
    image = distorted_image
  return image


def train_image(image_buffer,
                height,
                width,
                bbox,
                batch_position,
                resize_method,
                distortions,
                scope=None,
                summary_verbosity=0,
                distort_color_in_yiq=False,
                fuse_decode_and_crop=False):
  """Distort one image for training a network.
  Distorting images provides a useful technique for augmenting the data
  set during training in order to make the network invariant to aspects
  of the image that do not effect the label.
  Args:
    image_buffer: scalar string Tensor representing the raw JPEG image buffer.
    height: integer
    width: integer
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax].
    batch_position: position of the image in a batch, which affects how images
      are distorted and resized. NOTE: this argument can be an integer or a
      tensor
    resize_method: round_robin, nearest, bilinear, bicubic, or area.
    distortions: If true, apply full distortions for image colors.
    scope: Optional scope for op_scope.
    summary_verbosity: Verbosity level for summary ops. Pass 0 to disable both
      summaries and checkpoints.
    distort_color_in_yiq: distort color of input images in YIQ space.
    fuse_decode_and_crop: fuse the decode/crop operation.
  Returns:
    3-D float Tensor of distorted image used for training.
  """
  # with tf.op_scope([image, height, width, bbox], scope, 'distort_image'):
  # with tf.name_scope(scope, 'distort_image', [image, height, width, bbox]):
  with tf.name_scope(scope or 'distort_image'):
    # A large fraction of image datasets contain a human-annotated bounding box
    # delineating the region of the image containing the object of interest.  We
    # choose to create a new bounding box for the object which is a randomly
    # distorted version of the human-annotated bounding box that obeys an
    # allowed range of aspect ratios, sizes and overlap with the human-annotated
    # bounding box. If no box is supplied, then we assume the bounding box is
    # the entire image.
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.image.extract_jpeg_shape(image_buffer),
        bounding_boxes=bbox,
        min_object_covered=0.1,
        aspect_ratio_range=[0.75, 1.33],
        area_range=[0.05, 1.0],
        max_attempts=100,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box
    if summary_verbosity >= 3:
      image = tf.image.decode_jpeg(image_buffer, channels=3,
                                   dct_method='INTEGER_FAST')
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
      image_with_distorted_box = tf.image.draw_bounding_boxes(
          tf.expand_dims(image, 0), distort_bbox)
      tf.summary.image(
          'images_with_distorted_bounding_box',
          image_with_distorted_box)

    # Crop the image to the specified bounding box.
    if fuse_decode_and_crop:
      offset_y, offset_x, _ = tf.unstack(bbox_begin)
      target_height, target_width, _ = tf.unstack(bbox_size)
      crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
      image = tf.image.decode_and_crop_jpeg(
          image_buffer, crop_window, channels=3)
    else:
      image = tf.image.decode_jpeg(image_buffer, channels=3,
                                   dct_method='INTEGER_FAST')
      image = tf.slice(image, bbox_begin, bbox_size)

    distorted_image = tf.image.random_flip_left_right(image)

    # This resizing operation may distort the images because the aspect
    # ratio is not respected.
    image_resize_method = get_image_resize_method(resize_method, batch_position)
    distorted_image = tf.image.resize_images(
        distorted_image, [height, width],
        image_resize_method,
        align_corners=False)
    # Restore the shape since the dynamic slice based upon the bbox_size loses
    # the third dimension.
    distorted_image.set_shape([height, width, 3])
    if summary_verbosity >= 3:
      tf.summary.image('cropped_resized_maybe_flipped_image',
                       tf.expand_dims(distorted_image, 0))

    if distortions:
      distorted_image = tf.cast(distorted_image, dtype=tf.float32)
      # Images values are expected to be in [0,1] for color distortion.
      distorted_image /= 255.
      # Randomly distort the colors.
      distorted_image = distort_color(distorted_image, batch_position,
                                      distort_color_in_yiq=distort_color_in_yiq)

      # Note: This ensures the scaling matches the output of eval_image
      distorted_image *= 255

    if summary_verbosity >= 3:
      tf.summary.image(
          'final_distorted_image',
          tf.expand_dims(distorted_image, 0))
    return distorted_image

def preprocess(image_buffer, bbox=None, batch_position=None):
    """Preprocessing image_buffer as a function of its batch position."""
    # if self.train:
    #   image = train_image(image_buffer, self.height, self.width, bbox,
    #                       batch_position, self.resize_method, self.distortions,
    #                       None, summary_verbosity=self.summary_verbosity,
    #                       distort_color_in_yiq=self.distort_color_in_yiq,
    #                       fuse_decode_and_crop=self.fuse_decode_and_crop)
    # else:
    # image = tf.image.decode_jpeg(
    #       image_buffer, channels=3, dct_method='INTEGER_FAST')
    image = decode_raw(image_buffer)
    if batch_position is not None:
        image = eval_image(image, 224, 224, batch_position,
                           'bilinear',
                             summary_verbosity=0)
    # Note: image is now float32 [height,width,3] with range [0, 255]

    # image = tf.cast(image, tf.uint8) # HACK TESTING

    # normalized = normalized_image(image)
    return tf.cast(image, tf.float32)

def parse_and_preprocess(value, batch_position):
    # image_buffer, label_index, bbox, _ = parse_example_proto(value)
    image_buffer = parse_custom_proto(value)
    image = preprocess(image_buffer, None, None)
    # return (label_index, image)
    return image

def parallel_read_data(data_dir, batch_size, batch_size_per_split, num_splits, cycle_length, num_threads, cache_data=False, train=False):

    file_names = gfile.Glob(os.path.join(data_dir, '%s*_*.tfrecords' % 'train'))
    ds = tf.data.TFRecordDataset.list_files(file_names)

    ds = ds.apply(
        # interleave_ops.parallel_interleave(
        #     tf.data.TFRecordDataset, cycle_length=cycle_length))
        tf.data.experimental.parallel_interleave(
            tf.data.TFRecordDataset,
            cycle_length=cycle_length,
        ))
    if cache_data:
        ds = ds.take(1).cache().repeat()
    counter = tf.data.Dataset.range(batch_size)
    counter = counter.repeat()
    ds = tf.data.Dataset.zip((ds, counter))
    ds = ds.prefetch(buffer_size=batch_size)
    if train:
        # ds = ds.shuffle(buffer_size=10000)
        ds = ds.apply(
            tf.data.experimental.shuffle_and_repeat(buffer_size=10000))
    else:
        ds = ds.repeat()
    # ds = ds.apply(
    #     batching.map_and_batch(
    #         map_func=parse_and_preprocess,
    #         batch_size=batch_size_per_split,
    #         num_parallel_batches=num_splits))
    ds = ds.apply(
        tf.data.experimental.map_and_batch(
            map_func=parse_and_preprocess,
            batch_size=batch_size_per_split,
            num_parallel_batches=num_splits))
    # ds = ds.apply(
    #     batching.map_and_batch(
    #         map_func=parse_and_preprocess,
    #         batch_size=batch_size_per_split,
    #         num_parallel_calls=num_splits))
    ds = ds.prefetch(buffer_size=num_splits)
    if num_threads:
        ds = threadpool.override_threadpool(
            ds,
            threadpool.PrivateThreadPool(
                num_threads, display_name='input_pipeline_thread_pool'))
        ds_iterator = ds.make_initializable_iterator()
        tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS,
                             ds_iterator.initializer)
    else:
        ds_iterator = ds.make_one_shot_iterator()

    return ds_iterator

# tf.image.convert_image_dtype
# tf.image.resize_images
# tf.image.adjust_hue
# tf.image.rgb_to_hsv
# if __name__ == '__main__':
def read_data(cycle_length, num_splits, data_dir_index, batch_size, num_threads):

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    loop = 2
    # num_splits = 1
    # cycle_length = 1
    # data_dir = './benchmarks-cnn_tf_v1.8_compatible/scripts/tf_cnn_benchmarks/test_data/fake_tf_record_data/'
    data_dir = './TFRecords{}/'.format(data_dir_index)
    # batch_size = 10000
    batch_size_per_split = batch_size / num_splits
    # num_threads = 2

    sess = tf.Session()

    test_profiler = model_analyzer.Profiler(graph=sess.graph)
    # time.sleep(10)
    with tf.name_scope('batch_processing'):
        # Build final results per split.
        # images = [[] for _ in range(loop)]
        # labels = [[] for _ in range(loop)]
        ds_iterator = parallel_read_data(data_dir=data_dir,
                                         batch_size=batch_size,
                                         batch_size_per_split=batch_size_per_split,
                                         num_splits=num_splits,
                                         cycle_length=cycle_length,
                                         num_threads=num_threads)


        sess.run(ds_iterator.initializer, options=run_options, run_metadata=run_metadata)
        test_profiler.add_step(step=0, run_meta=run_metadata)
        with open('./parallel_results/{}-{}-{}-{}.txt'.format(cycle_length, num_splits, batch_size, data_dir_index), 'w') as f_out:
            for d in xrange(loop):
                for k in xrange(num_splits):
                    images = ds_iterator.get_next()
                    start = time.time()
                    temp = sess.run(images, options=run_options, run_metadata=run_metadata)
                    end = time.time()
                    # test_profiler.add_step(step=k+1, run_meta=run_metadata)
                    print len(temp)
                    print 'time is: {}'.format(end - start)
                    print 'End {}-{}'.format(d, k)
                    f_out.write('{}-{} {}'.format(d, k, end - start)+'\n')
                    # tl = timeline.Timeline(run_metadata.step_stats)
                    # ctf = tl.generate_chrome_trace_format()
                    # with open('./parallel_timelines/timeline_parallel{}_{}_{}-{}-{}.json'.format(d, k, num_splits, cycle_length, num_threads), 'w') as f:
                    #     f.write(ctf)

    # profile_code_opt_builder = option_builder.ProfileOptionBuilder()
    #
    # # 过滤条件：显示minist.py代码。
    # profile_code_opt_builder.with_max_depth(2000)
    # profile_code_opt_builder.with_node_names(show_name_regexes=['parallel_io_test.py.*'])
    #
    # # 过滤条件：只显示执行时间大于10us的代码
    # profile_code_opt_builder.with_min_execution_time(min_micros=1)
    #
    # # 显示字段：执行时间，且结果按照时间排序
    # profile_code_opt_builder.select(['micros'])
    # profile_code_opt_builder.order_by('micros')
    #
    #
    #
    # # 显示视图为code view
    # test_profiler.profile_python(profile_code_opt_builder.build())
    # time.sleep(10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="your script description")
    parser.add_argument('--dir-index')
    parser.add_argument('--cycle-length')
    parser.add_argument('--num-splits')
    # batch_size is limited by GPU memory
    parser.add_argument('--batch-size')
    parser.add_argument('--num-threads')

    args = parser.parse_args()
    dir_index = args.dir_index
    cycle_length = args.cycle_length
    num_splits = args.num_splits
    batch_size = args.batch_size
    num_threads = args.num_threads

    # cProfile.run('read_data('+ cycle_length +', '+ num_splits +', '+ dir_index +', '+ batch_size +', '+ num_threads +')',
    #              "./parallel_results/result{}-{}-{}-{}-{}.out".format(cycle_length,num_splits, dir_index, batch_size,num_threads))
    # p = pstats.Stats("result")
    # p.strip_dirs().sort_stats('cumtime').print_stats()
    read_data(int(cycle_length), int(num_splits), int(dir_index), int(batch_size), int(num_threads))
