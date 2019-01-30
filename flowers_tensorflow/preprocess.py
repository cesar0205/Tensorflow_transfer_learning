from create_tf_records.dataset_utils import _dataset_exists, _get_filenames_and_classes, write_label_file, _convert_dataset
import random


dataset_dir = './flowers' #Dataset directory

# Proportion of dataset to be used for evaluation
validation_size = 0.2
test_size = 0.2

# The number of shards to split the dataset into.
num_shards = 2

# Seed for repeatability.
random_seed = 0

#Output filename for the naming the TFRecord file
tfrecord_filename = "flowers"

#Posprocessing file height and width.
image_size = 299

def split_data():

    photo_filenames, class_names = _get_filenames_and_classes(dataset_dir)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    write_label_file(labels_to_class_names, dataset_dir)

    #Find the number of validation examples we need
    num_validation = int(validation_size * len(photo_filenames))
    num_test = int(test_size * len(photo_filenames))
    num_training = len(photo_filenames) - num_test - num_validation;

    # Divide the training datasets into train and test:
    random.seed(random_seed)
    random.shuffle(photo_filenames)
    training_filenames = photo_filenames[:num_training]
    validation_filenames = photo_filenames[num_training:num_training + num_validation]
    test_filenames = photo_filenames[-num_test:]


    # First, convert the training and validation sets.

    _convert_dataset('train', training_filenames, class_names_to_ids,
                     dataset_dir = dataset_dir,
                     tfrecord_filename = tfrecord_filename,
                     _NUM_SHARDS = num_shards)
    _convert_dataset('validation', validation_filenames, class_names_to_ids,
                     dataset_dir = dataset_dir,
                     tfrecord_filename = tfrecord_filename,
                     _NUM_SHARDS = num_shards)
    _convert_dataset('test', test_filenames, class_names_to_ids,
                     dataset_dir = dataset_dir,
                     tfrecord_filename = tfrecord_filename,
                     _NUM_SHARDS = num_shards)
