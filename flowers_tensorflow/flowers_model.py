import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from read_tf_records.create_dataset import get_datasets
from tensorflow.python.platform import tf_logging as logging
import tensorflow.contrib.slim as slim
from early_stopping import EarlyStoppingHook
import os
import time
from original_inception_resnet_v2.inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
from preprocess import split_data

def stream_loss_op(loss, name="stream_loss"):
    with tf.variable_scope(name):
        accum_loss = tf.get_variable(name="accum_loss", shape=[], initializer=tf.zeros_initializer(),
                                     collections=[tf.GraphKeys.LOCAL_VARIABLES])
        count = tf.get_variable(name="count", shape=[], initializer=tf.zeros_initializer(),
                                collections=[tf.GraphKeys.LOCAL_VARIABLES])

        update_total_op = tf.assign_add(accum_loss, loss);
        update_count_op = tf.assign_add(count, 1);

        update_op = update_total_op / update_count_op
        stream_loss = accum_loss / count
        return stream_loss, update_op


class InceptionV2():
    def __init__(self, n_classes, dataset_dir, log_dir, checkpoint_dir, init_checkpoint_file, save_interval=None,
                 max_checks_without_progress=10):
        self.n_classes = n_classes;
        self.dataset_dir = dataset_dir;
        self.log_dir = log_dir;
        self.checkpoint_dir = checkpoint_dir;
        self.init_checkpoint_file = init_checkpoint_file;
        self.final_checkpoint_file = os.path.join(checkpoint_dir, 'final_model', 'final_model.ckpt');
        self.save_interval = save_interval;
        self.max_checks_without_progress = max_checks_without_progress;

    def build_model(self):
        tf.reset_default_graph();
        logging.info("Building model...")
        # Now we start to construct the graph and build our model
        tf.logging.set_verbosity(tf.logging.INFO)  # Set the verbosity to INFO level

        with tf.name_scope("batch"):
            datasets = get_datasets(self.dataset_dir, self.batch_size)
            images, raw_images, labels = datasets['images'], datasets['raw_images'], datasets['labels']

        n_train_batches = int(datasets['n_train'] // self.batch_size) + 1
        n_validation_batches = int(datasets['n_validation'] // self.batch_size) + 1
        n_test_batches = int(datasets['n_test'] // self.batch_size) + 1

        if (self.save_interval is None):
            self.save_interval = n_train_batches;

        logging.info("Evaluating each %d steps", self.save_interval)

        self.is_training = tf.placeholder_with_default(False, shape=(), name="is_training")
        # Create the model inference
        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            logits, end_points = inception_resnet_v2(images, num_classes=self.n_classes, is_training=self.is_training)

        # For the final model I need all the original nodes to be included in the network structure.
        final_inception_saver = tf.train.Saver(max_to_keep=1);

        # Define the scopes that you want to exclude for restoration.
        exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']
        # Gets a list of variables
        variables_to_restore = slim.get_variables_to_restore(exclude=exclude)

        self.original_inception_saver = tf.train.Saver(variables_to_restore)

        # Create the global step for monitoring the learning_rate and training.
        global_step = tf.train.get_or_create_global_step()
        self.global_step = global_step;

        # Performs the equivalent to tf.nn.sparse_softmax_cross_entropy_with_logits but enhanced with checks
        with tf.name_scope("new_loss"):
            # Perform one-hot-encoding of the labels (Try one-hot-encoding within the load_batch function!)
            xentropy = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
            total_loss = tf.losses.get_total_loss()  # obtain the regularization losses as well
            train_loss, train_loss_update = stream_loss_op(total_loss, "train_stream_loss")

            val_loss = tf.reduce_mean(xentropy, name="val_loss")
            val_loss, val_loss_update = stream_loss_op(val_loss, "val_stream_loss")

            self.save_flag = tf.Variable(False, dtype=tf.bool);

        # Define your exponentially decaying learning rate
        with tf.name_scope("new_opt_step"):
            lr = tf.train.exponential_decay(
                learning_rate=self.initial_learning_rate,
                global_step=global_step,
                decay_steps=int(self.num_epochs_before_decay * n_train_batches),
                decay_rate=self.learning_rate_decay_factor,
                staircase=True)
            self.lr = lr;

            # Now we can define the optimizer that takes on the learning rate
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)

            # Create the train_op.
            # When evaluated, computes the gradients, runs the UPDATE_OPS and returns the total loss value.
            train_op = slim.learning.create_train_op(total_loss, optimizer, global_step=global_step)
            self.train_op = train_op;

        with tf.name_scope("metrics"):
            # State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
            predictions = tf.argmax(end_points['Predictions'], 1)

            train_accuracy, train_accuracy_update = tf.metrics.accuracy(labels, predictions, name="train_stream_acc")
            val_accuracy, val_accuracy_update = tf.metrics.accuracy(labels, predictions, name="val_stream_acc")

        with tf.name_scope("summaries"):
            # Now finally create all the summaries you need to monitor and group them into one summary op.
            train_loss_summary = tf.summary.scalar('losses/total_loss', train_loss)
            train_acc_summary = tf.summary.scalar('train_accuracy', train_accuracy)
            lr_summary = tf.summary.scalar('learning_rate', lr)
            train_summary_op = tf.summary.merge([train_loss_summary, train_acc_summary, lr_summary])

            val_loss_summary = tf.summary.scalar('losses/val_loss', val_loss)
            val_acc_summary = tf.summary.scalar('val_accuracy', val_accuracy)
            val_summary_op = tf.summary.merge([val_loss_summary, val_acc_summary])

            test_summary_op = tf.summary.scalar('test_accuracy', val_accuracy)

        # We don't need an explicit writer for training data. It is handled by its hook.
        self.validation_writer = tf.summary.FileWriter(os.path.join(self.log_dir, "validation/"),
                                                       graph=tf.get_default_graph())
        self.train_writer = tf.summary.FileWriter(os.path.join(self.log_dir, "train/"), graph=tf.get_default_graph())

        with tf.name_scope("hooks"):
            self.save_checkpoint_hook = tf.train.CheckpointSaverHook(
                checkpoint_dir=self.checkpoint_dir,
                checkpoint_basename='flowers_model.ckpt',
                save_steps=self.save_interval)

            logging.info("Saving checkpoints each %d steps", self.save_interval)

            self.save_summary_hook = tf.train.SummarySaverHook(
                summary_writer=self.train_writer,
                save_steps=self.save_interval,
                summary_op=train_summary_op);

            stop_hook = tf.train.StopAtStepHook(last_step=n_train_batches * self.n_epochs + 1);
            logging.info("Last step should be at %d", n_train_batches * self.n_epochs + 1)
            # Although the save_checkpoint_hook can save the final model at the final step we need another hook to save only
            # the variables that we need(the short version)

            early_stopping_hook = EarlyStoppingHook(max_checks_without_progress=self.max_checks_without_progress,
                                                    evaluation_interval=self.save_interval,
                                                    loss=val_loss, saver=final_inception_saver,
                                                    save_flag_tensor=self.save_flag,
                                                    final_checkpoint_file=self.final_checkpoint_file)

        train_stream_acc_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics/train_stream_acc")
        val_stream_acc_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics/val_stream_acc")
        train_stream_loss_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="train_stream_loss")
        val_stream_loss_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="val_stream_loss")

        # Init running ops
        self.train_stream_acc_init = tf.variables_initializer(train_stream_acc_vars)
        self.val_stream_acc_init = tf.variables_initializer(val_stream_acc_vars)
        self.train_stream_loss_init = tf.variables_initializer(train_stream_loss_vars)
        self.val_stream_loss_init = tf.variables_initializer(val_stream_loss_vars)

        with tf.name_scope("saver"):
            # Now we create a saver function that actually restores the variables from a checkpoint file in a sess
            # This saver grabs the graph structure constructed up to now (Long version) and we ask it to only restore into
            # memory all but the last layers.
            extended_model_saver = tf.train.Saver(variables_to_restore)

        # Handlers for datasets
        self.iter_train_handle = datasets["iter_train_handle"]
        self.iter_val_handle = datasets["iter_val_handle"]
        self.iter_test_handle = datasets["iter_test_handle"]
        self.data_handle = datasets["handle"]

        # Metrics
        self.train_accuracy_update = train_accuracy_update
        self.train_accuracy = train_accuracy;

        self.val_accuracy_update = val_accuracy_update
        self.val_accuracy = val_accuracy;

        self.train_loss_update = train_loss_update
        self.train_loss = train_loss;

        self.val_loss_update = val_loss_update;
        self.val_loss = val_loss

        # Summaries
        self.train_summary_op = train_summary_op
        self.val_summary_op = val_summary_op
        self.test_summary_op = test_summary_op

        self.n_train_batches = n_train_batches;
        self.n_validation_batches = n_validation_batches;
        self.n_test_batches = n_test_batches;

        self.stop_hook = stop_hook;
        self.early_stopping_hook = early_stopping_hook;

        logging.info("Finished building model...")

    # Print training statistics at the beggining of each epoch
    def print_training_stats(self, sess, epoch):
        logging.info('Epoch %s/%s', epoch, self.n_epochs)
        lr, acc, loss = sess.run([self.lr, self.train_accuracy, self.train_loss], feed_dict={self.is_training: True})
        logging.info('Current Learning Rate: %s', lr)
        logging.info('Streaming Accuracy: %s', acc)
        logging.info('Streaming Loss: %s', loss)

    def run_validation(self, sess, n_batches, handle_val, step, operation):
        logging.info("Computing %s with %d batches at step %d", operation, n_batches, step)

        for batch in range(n_batches):
            acc, loss = self.val_step(sess, handle_val, batch, operation)

        return acc, loss;

    # Create a evaluation step function
    def val_step(self, sess, handle_val, step, operation):
        start_time = time.time()
        val_acc, val_loss = sess.run([self.val_accuracy_update, self.val_loss_update],
                                     feed_dict={self.data_handle: handle_val})
        time_elapsed = time.time() - start_time

        # Log some information
        logging.info('%s Step %s: %s Acc: %.4f %s Loss: %.4f  (%.2f sec/step)',
                     operation, step, operation, val_acc, operation, val_loss, time_elapsed)

        return val_acc, val_loss;

    def train(self, n_epochs=4, batch_size=10, initial_learning_rate=0.0002, learning_rate_decay_factor=0.7,
              num_epochs_before_decay=4):
        print("Training model....")
        self.n_epochs = n_epochs;
        self.batch_size = batch_size;
        self.initial_learning_rate = initial_learning_rate;
        self.learning_rate_decay_factor = learning_rate_decay_factor;
        self.num_epochs_before_decay = num_epochs_before_decay;

        self.build_model();

        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        def restore_fn(scaffold, sess):
            logging.info("Restoring initial model")
            return self.original_inception_saver.restore(sess, self.init_checkpoint_file)

        scaffold = tf.train.Scaffold(init_fn=restore_fn)
        # At this point we have a new structure graph built in build_model(). In case the system crashes we need to be sure
        # that the session stored in checkpoint_dir matches the current model in memory.
        with tf.train.MonitoredTrainingSession(  # checkpoint_dir = self.checkpoint_dir,
                scaffold=scaffold,
                # In case training chrashed it will look automatically in this directory for the last checkpoint file
                # If the checkpoint_dir is empty then it will call restore_fn for the initial checkpoint
                hooks=[self.stop_hook,
                       self.save_checkpoint_hook,
                       self.save_summary_hook,
                       self.early_stopping_hook
                       ]
        ) as sess:

            # scaffold=scaffold) as sess:

            handle_train, handle_val = sess.run([self.iter_train_handle, self.iter_val_handle])

            while not sess.should_stop():
                step = tf.train.global_step(sess, self.global_step)

                if step % self.save_interval == 0 and step > 0:
                    self.print_training_stats(sess, step // self.n_train_batches + 1);
                    sess.run([self.train_stream_acc_init, self.train_stream_loss_init])

                new_step, train_acc, train_loss = self.train_step(sess, handle_train)

                if new_step % self.save_interval == 0:
                    sess.run([self.val_stream_acc_init, self.val_stream_loss_init, self.save_flag.initializer]);
                    val_acc, val_loss = self.run_validation(sess, self.n_validation_batches, handle_val, new_step,
                                                            "validation");
                    summaries = sess.run(self.val_summary_op)
                    self.validation_writer.add_summary(summaries)

                    logging.info("Setting flag to true....")
                    sess.run(self.save_flag, feed_dict={self.save_flag: True});
                    logging.info("=====Final val loss: ", val_loss)

            # We log the final training loss and accuracy
            logging.info('Final training loss: %s', train_loss)
            logging.info('Final training accuracy: %s', train_acc)

    # Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
    def train_step(self, sess, handle_train):
        '''
            Runs a session and gives a logging on the time elapsed for each global step
        '''
        start_time = time.time()
        __, train_acc, train_loss, new_step = sess.run(
            [self.train_op, self.train_accuracy_update, self.train_loss_update,
             self.global_step], feed_dict={self.is_training: True, self.data_handle: handle_train})
        time_elapsed = time.time() - start_time

        # Run the logging to print some results
        logging.info('Global step %s: Train loss: %.4f (%.2f sec/step)', new_step, train_loss, time_elapsed)

        return new_step, train_acc, train_loss

    def test(self, batch_size, checkpoint_file=None):
        tf.reset_default_graph();
        n_samples = 4;

        labels_to_class_names = {}
        with open("./flowers/labels.txt", "r") as f:
            for line in f:
                elements = line.split(":")
                labels_to_class_names[int(elements[0])] = elements[1].strip()

        if checkpoint_file is None:
            checkpoint_file = self.final_checkpoint_file

        with tf.name_scope("batch"):
            datasets = get_datasets(self.dataset_dir, batch_size)
            images, raw_images, labels = datasets['images'], datasets['raw_images'], datasets['labels']

        self.n_test_batches = int(datasets['n_test'] / batch_size)
        is_training = tf.placeholder_with_default(False, shape=(), name="is_training")

        # Create the model inference
        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            val_logits, val_end_points = inception_resnet_v2(images, num_classes=self.n_classes,
                                                             is_training=is_training)

        # Define final model saver here to match the final model saver in the training part.
        final_inception_saver = tf.train.Saver();

        with tf.name_scope("new_loss"):
            val_xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=val_logits)
            val_loss = tf.reduce_mean(val_xentropy, name="val_loss")
            val_loss, val_loss_update = stream_loss_op(val_loss, "val_stream_loss")

        # The next nodes won't be taken into account when restoring the model.
        with tf.name_scope("metrics"):
            val_predictions = tf.argmax(val_end_points['Predictions'], 1)
            val_accuracy, val_accuracy_update = tf.metrics.accuracy(labels, val_predictions, name="val_stream_acc")

        val_stream_acc_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics/val_stream_acc")
        val_stream_loss_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="val_stream_loss")

        # Init running ops
        self.val_stream_acc_init = tf.variables_initializer(val_stream_acc_vars)
        self.val_stream_loss_init = tf.variables_initializer(val_stream_loss_vars)

        self.val_accuracy_update = val_accuracy_update;
        self.val_loss_update = val_loss_update;
        self.iter_test_handle = datasets["iter_test_handle"]
        self.data_handle = datasets["handle"]

        local_init = tf.local_variables_initializer();

        # Caculate test accuracy
        with tf.Session() as sess:

            handle_test, __ = sess.run([self.iter_test_handle, local_init])
            final_inception_saver.restore(sess, checkpoint_file)
            test_acc, test_loss = self.run_validation(sess, self.n_test_batches, handle_test, 0, "test");

        logging.info('Final test accuracy: %s', test_acc)

        # Print correct and incorrect samples
        with tf.Session() as sess:
            handle_test, __ = sess.run([self.iter_test_handle, local_init])

            final_inception_saver.restore(sess, checkpoint_file)
            raw_images_, labels_, predictions_ = sess.run([raw_images, labels, val_predictions],
                                                          feed_dict={self.data_handle: handle_test});
            pos_indexes = np.where(labels_ == predictions_)[:n_samples]
            pos_images, pos_labels, pos_predictions = raw_images_[pos_indexes], labels_[pos_indexes], predictions_[
                pos_indexes]

            neg_images = np.empty([0, 299, 299, 3], dtype=np.int64)
            neg_labels = np.array([], dtype=np.int64)
            neg_predictions = np.array([], dtype=np.int64)
            while (True):
                raw_images_, labels_, predictions_ = sess.run([raw_images, labels, val_predictions],
                                                              feed_dict={self.data_handle: handle_test});
                neg_indexes_part = np.where(labels_ != predictions_)

                neg_images = np.append(neg_images, raw_images_[neg_indexes_part], axis=0)
                neg_labels = np.append(neg_labels, labels_[neg_indexes_part])
                neg_predictions = np.append(neg_predictions, predictions_[neg_indexes_part])

                if (len(neg_labels) > n_samples):
                    break;

            neg_images = neg_images[:n_samples]
            neg_labels = neg_labels[:n_samples]
            neg_predictions = neg_predictions[:n_samples]

            plt.figure(figsize=(15, 8))
            print("Correct predictions")

            for i in range(n_samples):
                plt.subplot(140 + i + 1)
                plt.title("Label:{}\nPrediction:{}".format(labels_to_class_names[pos_labels[i]],
                                                           labels_to_class_names[pos_predictions[i]]))
                plt.imshow(pos_images[i])
                plt.axis("off")
                plt.show()

            print("Incorrect predictions")

            plt.figure(figsize=(15, 8))

            for i in range(n_samples):
                plt.subplot(140 + i + 1)
                plt.title("Label:{}\nPrediction:{}".format(labels_to_class_names[neg_labels[i]],
                                                           labels_to_class_names[neg_predictions[i]]))
                plt.imshow(neg_images[i])
                plt.axis("off")
                plt.show()


def main():

    #split_data();

    model = InceptionV2(n_classes=5,
                        log_dir="./logs",
                        dataset_dir="./flowers",
                        checkpoint_dir='./checkpoints',
                        init_checkpoint_file='./original_inception_resnet_v2/inception_resnet_v2_2016_08_30.ckpt',
                        max_checks_without_progress=10);
    model.train(n_epochs=5, batch_size=8)

    #model.test(batch_size = 10)

if __name__ == "__main__":
    main();
