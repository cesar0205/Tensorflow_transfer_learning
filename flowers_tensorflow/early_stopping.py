import tensorflow as tf
import numpy as np
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.platform import tf_logging as logging

class EarlyStoppingCounter:

    def __init__(self, curr_loss):
        self.curr_loss = curr_loss;
        self.update_ops = self._build_early_stopping();

    def update_best_loss_fn(self):
        # This is because I want the update assign, and the init assign be executed. However as the inc_checks_without_progress_fn
        # returns one op element of type int, I also have to return a list of one op of type int here.
        # The only way I have to achieve this is to use a control dependencies where I put all the ops I want to execute besides
        # the op I want to return.
        with tf.control_dependencies([tf.assign(self.best_loss, self.curr_loss)]):
            checks_without_progress_init = tf.assign(self.checks_without_progress, 0)
        return [self.curr_loss, checks_without_progress_init];

    def inc_checks_without_progress_fn(self):
        inc_checks_without_progress_op = tf.assign_add(self.checks_without_progress, 1);
        return [self.curr_loss, inc_checks_without_progress_op];

    def _build_early_stopping(self):
        self.best_loss = tf.Variable(np.inf, dtype=tf.float32);
        self.checks_without_progress = tf.Variable(0, dtype=tf.int32)

        update_ops = tf.cond(self.curr_loss < self.best_loss, self.update_best_loss_fn,
                             self.inc_checks_without_progress_fn)
        return update_ops;


class EarlyStoppingHook(session_run_hook.SessionRunHook):
    def __init__(self, max_checks_without_progress, evaluation_interval, loss, saver, save_flag_tensor,
                 final_checkpoint_file):
        self.max_checks_without_progress = max_checks_without_progress;
        self.evaluation_interval = evaluation_interval;
        self.loss = loss;
        self.saver = saver;
        self.save_flag_tensor = save_flag_tensor;
        self.counter = EarlyStoppingCounter(loss);
        self.best_loss_value = np.inf;
        self.final_checkpoint_file = final_checkpoint_file;

    def begin(self):
        self._global_step_tensor = training_util._get_or_create_global_step_read()
        self._previous_step = 0
        self._step = 0

    def before_run(self, run_context):
        # print("Before run....here", self._step, self._previous_step)

        if (self._step % self.evaluation_interval == 0) and (self._step != self._previous_step):
            # print("This is step before run... lets ask for the flag", self._step)

            return session_run_hook.SessionRunArgs({'step': self._global_step_tensor,
                                                    'flag': self.save_flag_tensor})
        else:
            return session_run_hook.SessionRunArgs({'step': self._global_step_tensor})

    def after_run(self, run_context, run_values):

        if (self._step % self.evaluation_interval == 0) and (self._step != self._previous_step):

            global_step = run_values.results['step'];
            flag = run_values.results['flag'];
            if (flag):
                new_loss, checks_without_progress = run_context.session.run(self.counter.update_ops)

                logging.info("=======Fetching results after flag is issued. New loss: {}, Checks without progress: {}, "
                             "Current step: {}", new_loss, checks_without_progress, self._step)

                if (checks_without_progress == 0):
                    logging.info("Saving new best model....")
                    self.saver.save(run_context.session, self.final_checkpoint_file, global_step)

                if (checks_without_progress > self.max_checks_without_progress):
                    logging.info("Max checks reached...finishing...")
                    run_context.request_stop();

                self._previous_step = global_step;
            self._step = global_step;
        else:
            self._step = run_values.results['step'];
