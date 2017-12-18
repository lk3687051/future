import tensorflow as tf
import sys
from future.dataset.history_feature_ratio import history_feature_ratio
from future.model import StockModel
import numpy as np
class OneRadioAvgStockModel(StockModel):
    def __init__(self):
        self.name = "cnn_one_mean_radio_avg"
        self.target_label = 'target_price_change1'
        StockModel.__init__(self)

    def _get_daily_feature(self, date = None):
        feature = history_feature_ratio(date=date)
        self.daily_dataset = feature.daily_feature()
        self.features = self.daily_dataset.iloc[:, 0: 300]

    def process_labels(self, labels):
        labels[ (-2 < labels) & (labels < 2)] = 0
        labels[labels <= -2] = -1
        labels[labels >=  2] = 1
        labels = labels + 1
        return labels

    def model_fn(self, features, labels, mode):
        """Model function for CNN."""
        input_layer = tf.reshape(features["x"], [-1, 5, 60, 1])
        conv1 = tf.layers.conv2d(inputs=input_layer, filters=32,
                    kernel_size=[1, 5],padding="same",activation=tf.nn.relu)
        pool1 = tf.layers.average_pooling2d(inputs=conv1, pool_size=[1, 2], strides=[1,2])

        conv2 = tf.layers.conv2d(inputs=pool1,filters=64,kernel_size=[1, 5],
                    padding="same",activation=tf.nn.relu)
        pool2 = tf.layers.average_pooling2d(inputs=conv2, pool_size=[1, 2], strides=[1,2])

        pool2_flat = tf.reshape(pool2, [-1, 5 * 15 * 64])

        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(
                    inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

        logits = tf.layers.dense(inputs=dropout, units=3)
        predictions = {
                    # Generate predictions (for PREDICT and EVAL mode)
                    "classes": tf.argmax(input=logits, axis=1),
                    # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
                    # `logging_hook`.
                    "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
                    }
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=3)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.003)
            train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
                    "accuracy": tf.metrics.accuracy(
                        labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(
                    mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    def train(self):
        feature = history_feature_ratio()
        train_dataset, x_ = feature.get_dataset()
        self.features = train_dataset.iloc[:, 0: 300]

        self.labels = self.process_labels(train_dataset[self.target_label])
        print(self.features)
        print(self.labels)
        # Create the Estimator
        tf.logging.set_verbosity(tf.logging.INFO)

        ## Set up logging for predictions
        # Log the values in the "Softmax" tensor with label "probabilities"
        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(
                    tensors=tensors_to_log, every_n_iter=1000)

        # Train the model
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": self.features.values.astype(np.float32)},
                    y=self.labels.values.astype(np.float32),
                    batch_size=100,
                    num_epochs=None,
                    shuffle=True)

        self.mnist_classifier.train(
                    input_fn=train_input_fn,
                    steps=100000,
                    hooks=[logging_hook])

    def eval(self):
        feature = history_feature_ratio()
        x_, test_dataset = feature.get_dataset()
        self.features = test_dataset.iloc[:, 0: 300]

        self.labels = self.process_labels(test_dataset[self.target_label])
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": self.features.values.astype(np.float32)},
                    y=self.labels.values.astype(np.float32),
                    num_epochs=1,
                    shuffle=False)
        i = 0
        j = 0
        predict_results = self.mnist_classifier.predict(input_fn=eval_input_fn)
        for predict, label in zip(predict_results, self.labels.values.astype(np.float32)):
            if (predict['probabilities'][2] > 0.90) and (predict['probabilities'][2] < 1):
                i = i +1
                print(predict, label)
                if predict['classes'] == int(label):
                    j = j + 1
        print("hhhhhhhhhh ")
        print(i, j)

def main():
    model = OneRadioStockModel()
    model.train()
    #model.eval()
    tf.app.run()

if __name__ == "__main__":
    main()
