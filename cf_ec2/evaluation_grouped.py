import numpy as np
import pandas as pd
from tqdm import tqdm
import time
# from absl import logging
# import logging
import tensorflow as tf
# import keras
from . import evaluation

# logger = logging.getLogger()

class BatchTimestamp(object):
  """A structure to store batch time stamp."""

  def __init__(self, batch_index, timestamp):
    self.batch_index = batch_index
    self.timestamp = timestamp

  def __repr__(self):
    return "'BatchTimestamp<batch_index: {}, timestamp: {}>'".format(
        self.batch_index, self.timestamp)

class metricsCallback(tf.keras.callbacks.Callback):
    """Callback for Keras models."""

    def __init__(self, batch_size, log_steps, logdir=None):
        """Callback for logging performance.
        Args:
        batch_size: Total batch size.
        log_steps: Interval of steps between logging of batch level stats.
        logdir: Optional directory to write TensorBoard summaries.
        """
        # TODO(wcromar): remove this parameter and rely on `logs` parameter of
        # on_train_batch_end()
        self.batch_size = batch_size
        super(metricsCallback, self).__init__()
        self.log_steps = log_steps
        self.last_log_step = 0
        self.steps_before_epoch = 0
        self.steps_in_epoch = 0
        self.start_time = None

        if logdir:
            self.summary_writer = tf.summary.create_file_writer(logdir)
        else:
            self.summary_writer = None

        # Logs start of step 1 then end of each step based on log_steps interval.
        self.timestamp_log = []

        # Records the time each epoch takes to run from start to finish of epoch.
        self.epoch_runtime_log = []

    @property
    def global_steps(self):
        """The current 1-indexed global step."""
        return self.steps_before_epoch + self.steps_in_epoch

    @property
    def average_steps_per_second(self):
        """The average training steps per second across all epochs."""
        return self.global_steps / sum(self.epoch_runtime_log)

    @property
    def average_examples_per_second(self):
        """The average number of training examples per second across all epochs."""
        return self.average_steps_per_second * self.batch_size

    def on_train_end(self, logs=None):
        self.train_finish_time = time.time()

        if self.summary_writer:
            self.summary_writer.flush()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_train_batch_begin(self, batch, logs=None):
        if not self.start_time:
            self.start_time = time.time()

        # Record the timestamp of the first global step
        if not self.timestamp_log:
            self.timestamp_log.append(BatchTimestamp(self.global_steps,
                                                self.start_time))

    def on_train_batch_end(self, batch, logs=None):
        """Records elapse time of the batch and calculates examples per second."""
        self.steps_in_epoch = batch + 1
        steps_since_last_log = self.global_steps - self.last_log_step
        if steps_since_last_log >= self.log_steps:
            now = time.time()
            elapsed_time = now - self.start_time
            steps_per_second = steps_since_last_log / elapsed_time
            examples_per_second = steps_per_second * self.batch_size

            self.timestamp_log.append(BatchTimestamp(self.global_steps, now))
            # logger.info(
            #     'TimeHistory: %.2f seconds, %.2f examples/second between steps %d '
            #     'and %d', elapsed_time, examples_per_second, self.last_log_step,
            #     self.global_steps)
            print(
                'TimeHistory: {:.2f} seconds, {:.2f} examples/second between steps {} and {}'.format(
                    elapsed_time, 
                    examples_per_second, 
                    self.last_log_step,
                    self.global_steps
                ))            

            if self.summary_writer:
                with self.summary_writer.as_default():
                    tf.summary.scalar('global_step/sec', steps_per_second,
                                        self.global_steps)
                    tf.summary.scalar('examples/sec', examples_per_second,
                                        self.global_steps)

            self.last_log_step = self.global_steps
            self.start_time = None

    def on_epoch_end(self, epoch, logs=None):
        epoch_run_time = time.time() - self.epoch_start
        self.epoch_runtime_log.append(epoch_run_time)

        self.steps_before_epoch += self.steps_in_epoch
        self.steps_in_epoch = 0
        print("global_steps: {}, last_log_step: {}".format(self.global_steps, self.last_log_step))
        # ## addtional evaluation metrics (self.validation_data deprecated in tf2)
        # X_val = [self.validation_data[0],self.validation_data[1]]
        # y_val = self.validation_data[2]
        # y_predict = self.model.predict(x = X_val)
        # logs['val_auc'] = evaluation.auc(y_val, y_predict)
        # logs['val_rmse'] = evaluation.rmse(y_val, y_predict)
        # logs['val_logloss'] = evaluation.logloss(y_val, y_predict)

        ## workaround for tf2
        # class ValMetrics(Callback):

        #     def __init__(self, validation_data):
        #         super(Callback, self).__init__()
        #         self.X_val, self.y_val = validation_data        



class metricsEval():
    def __init__(self, model, users, items):
        self.model = model
        self.users = list(set(users))
        self.items = list(set(items))

    def getRecs(self):
        ## create placeholders for user, item, pred
        list_users, list_items, list_preds = [], [], []
        ## get predictions for each user-item pair
        for user in tqdm(self.users):
            user = [user] * len(self.items)
            list_users.extend(user)
            list_items.extend(self.items)
            list_preds.extend(
                self.model.predict(
                    x=[
                        np.array(user),
                        np.array(self.items)
                    ]
                ).flatten()
            )
        ## create a pandas dataframe
        self.all_predictions = pd.DataFrame(data={
            'userID': list_users,
            'itemID': list_items,
            'prediction': list_preds
        })


    def getRankBasedMetrics(self,users_test,items_test,ratings_test,K=10):
        assert K>0, 'Invalid K value for evaluation!!'
        ## format the true test set
        all_test = self._formatTestDF(users_test,items_test,ratings_test)
        ## calculate the metrics
        recall = evaluation.recall_at_k(
            rating_true=all_test,
            rating_pred=self.all_predictions,
            col_user='userID',
            col_item='itemID',
            col_rating='rating',
            col_pred='prediction',
            k=K
        )
        precision = evaluation.precision_at_k(
            rating_true=all_test,
            rating_pred=self.all_predictions,
            col_user='userID',
            col_item='itemID',
            col_rating='rating',
            col_pred='prediction',
            k=K
        )     
        ndcg = evaluation.ndcg_at_k(
            rating_true=all_test,
            rating_pred=self.all_predictions,
            col_user='userID',
            col_item='itemID',
            col_rating='rating',
            col_pred='prediction',
            k=K
        )
        map = evaluation.map_at_k(
            rating_true=all_test,
            rating_pred=self.all_predictions,
            col_user='userID',
            col_item='itemID',
            col_rating='rating',
            col_pred='prediction',
            k=K
        )
        return recall,precision,ndcg,map           

    def getOverlapBasedMetrics(self,users_test,items_test,ratings_test):
        y_true = ratings_test
        y_pred = self.model.predict(
            x=[
                np.array(users_test),
                np.array(items_test)
            ]
        ).flatten()
        ## get the metrics
        rmse = evaluation.rmse(y_true, y_pred)
        auc = evaluation.auc(y_true,y_pred)
        logloss = evaluation.logloss(y_true,y_pred)
        return rmse,auc,logloss

    def _formatTestDF(self,users_test,items_test,ratings_test):
        all_test = pd.DataFrame(data={
            'userID':users_test,
            'itemID':items_test,
            'rating':ratings_test
        })
        all_test = all_test[all_test.rating>0].copy().reset_index(drop=True)
        return all_test

