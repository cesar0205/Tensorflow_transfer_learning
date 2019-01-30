import tensorflow as tf
import os
from original_inception_resnet_v2 import inception_preprocessing


def example_decoder(example_proto, target_height, target_width, is_training):
    # Create the keys_to_features dictionary for the decoder
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
        'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        'image/height': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        'image/width': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    parsed_features = tf.parse_single_example(example_proto, keys_to_features)

    raw_image = tf.image.decode_jpeg(parsed_features["image/encoded"])
    original_width = tf.cast(parsed_features["image/width"], tf.int32)
    original_height = tf.cast(parsed_features["image/height"], tf.int32)
    label = tf.cast(parsed_features["image/class/label"], tf.int32)

    # Reshape image data into the original shape
    raw_image = tf.reshape(raw_image, [original_height, original_width, 3])
    image = inception_preprocessing.preprocess_image(raw_image, target_height, target_width, is_training)

    # As for the raw images, we just do a simple reshape to batch it up
    raw_image = tf.expand_dims(raw_image, 0)
    raw_image = tf.image.resize_nearest_neighbor(raw_image, [target_height, target_width])
    raw_image = tf.squeeze(raw_image)

    return image, raw_image, label;

def count_instances_in_dataset(tfrecords_to_count):
    num_samples = 0
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1
    return num_samples;


def get_datasets(dataset_dir, batch_size, height=299, width=299, dataset_name='flowers'):
    with tf.name_scope("datasets"):
        train_file_pattern = dataset_name + '_train'
        validation_file_pattern = dataset_name + '_validation'
        test_file_pattern = dataset_name + '_test'

        train_tfrecords_names = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir)
                                 if file.startswith(train_file_pattern)]
        validation_tfrecords_names = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir)
                                      if file.startswith(validation_file_pattern)]
        test_tfrecords_names = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir)
                                if file.startswith(test_file_pattern)]

        n_train = count_instances_in_dataset(train_tfrecords_names);
        n_validation = count_instances_in_dataset(validation_tfrecords_names);
        n_test = count_instances_in_dataset(test_tfrecords_names);

        dataset = tf.data.TFRecordDataset(train_tfrecords_names)
        dataset = dataset.map(lambda x: example_decoder(x, height, width, True))
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.repeat()
        train_dataset = dataset.batch(batch_size)

        dataset = tf.data.TFRecordDataset(validation_tfrecords_names)
        dataset = dataset.map(
            lambda x: example_decoder(x, height, width, False))  # No need to shuffle test and val data
        dataset = dataset.repeat()  # It's important to repeat, otherwise it will send a sign to the monitoredtrainingsession to stop once the dataset is empty
        validation_dataset = dataset.batch(batch_size)

        dataset = tf.data.TFRecordDataset(test_tfrecords_names)
        dataset = dataset.map(lambda x: example_decoder(x, height, width, False))
        dataset = dataset.repeat()
        test_dataset = dataset.batch(batch_size)

        iter_train_handle = train_dataset.make_one_shot_iterator().string_handle()
        iter_val_handle = validation_dataset.make_one_shot_iterator().string_handle()
        iter_test_handle = test_dataset.make_one_shot_iterator().string_handle()

        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
        images, raw_images, labels = iterator.get_next()

        final_datasets = {'images': images,
                          'raw_images': raw_images,
                          'labels': labels,
                          "iter_train_handle": iter_train_handle,
                          "iter_val_handle": iter_val_handle,
                          "iter_test_handle": iter_test_handle,
                          'n_train': n_train,
                          'n_test': n_test,
                          'n_validation': n_validation,
                          'handle': handle}

        return final_datasets;