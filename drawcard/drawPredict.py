# -*- coding:utf-8 -*-


import tensorflow as tf
import numpy as np
import functools
import ast
import logging
import json
import sys
from drawcard.doglas import Douglas


class Predict:

    # sys.setrecursionlimit(100000)
    FLAGS = ''

    def draw_predict_fn(self,inkarray,requestId):

        def get_input_fn(mode, tfrecord_pattern, batch_size):
            """Creates an input_fn that stores all the data in memory.
            Args:
             mode: one of tf.contrib.learn.ModeKeys.{TRAIN, INFER, EVAL}
             tfrecord_pattern: path to a TF record file created using create_dataset.py.
             batch_size: the batch size to output.
            Returns:
              A valid input_fn for the model estimator.
            """

            def _parse_tfexample_fn(example_proto, mode):
                """Parse a single record which is expected to be a tensorflow.Example."""
                feature_to_type = {
                    "ink": tf.VarLenFeature(dtype=tf.float32),
                    "shape": tf.FixedLenFeature([2], dtype=tf.int64)
                }
                if mode != tf.estimator.ModeKeys.PREDICT:
                    # The labels won't be available at inference time, so don't add them
                    # to the list of feature_columns to be read.
                    feature_to_type["class_index"] = tf.FixedLenFeature([1], dtype=tf.int64)

                parsed_features = tf.parse_single_example(example_proto, feature_to_type)
                labels = None
                if mode != tf.estimator.ModeKeys.PREDICT:
                    labels = parsed_features["class_index"]
                parsed_features["ink"] = tf.sparse_tensor_to_dense(parsed_features["ink"])
                return parsed_features, labels

            def _input_fn():
                """Estimator `input_fn`.
                Returns:
                  A tuple of:
                  - Dictionary of string feature name to `Tensor`.
                  - `Tensor` of target labels.
                """
                dataset = tf.data.TFRecordDataset.list_files(tfrecord_pattern)
                if mode == tf.estimator.ModeKeys.TRAIN:
                    dataset = dataset.shuffle(buffer_size=10000)
                dataset = dataset.repeat()
                # Preprocesses 10 files concurrently and interleaves records from each file.
                dataset = dataset.interleave(
                    tf.data.TFRecordDataset,
                    cycle_length=10,
                    block_length=10)
                dataset = dataset.map(
                    functools.partial(_parse_tfexample_fn, mode=mode),
                    num_parallel_calls=10)
                dataset = dataset.prefetch(10000)
                if mode == tf.estimator.ModeKeys.TRAIN:
                    dataset = dataset.shuffle(buffer_size=10000)
                # Our inputs are variable length, so pad them.
                dataset = dataset.padded_batch(
                    batch_size, padded_shapes=dataset.output_shapes)
                features, labels = dataset.make_one_shot_iterator().get_next()
                return features, labels

            return _input_fn

        def draw_image_cover(inkarray):
            newpoint = []
            for xarr, yarr in inkarray:
                linestring = 'LINESTRING('
                n = 0
                stroke = []
                xpoints = []
                ypoints = []
                if len(xarr) > 2:
                    for x in xarr:
                        linestring += str(x) + " " + str(yarr[n]) + ','
                        n += 1
                    linestring = linestring[0:len(linestring) - 1]
                    linestring += ')'
                    d = Douglas()
                    points = d.readPoint(linestring)
                    d.compress(points, points[0], points[len(points) - 1])
                    for point in points:
                        xpoints.append(int(point.x))
                        ypoints.append(int(point.y))
                else:
                    xpoints = xarr.copy()
                    ypoints = yarr.copy()
                stroke.append(tuple(xpoints))
                stroke.append(tuple(ypoints))
                newpoint.append(tuple(stroke))
            return newpoint

        def quick_sort_fn(a):
            if len(a) < 2:
                return a
            else:
                d = a[0]
                s = 0
                leftarr = []
                rightarr = []
                mid = []
                if not str(d).__contains__('#'):
                    for i in range(len(a)):
                        s = i
                        if a[i] > 0:
                            d = a[i]
                            break
                    mid.append(str(d)+"#0")
                else:
                    mid.append(str(d))
                    d = d.split('#')[0]
                for i in range(s+1, len(a)):
                    if not str(a[i]).__contains__('#'):
                        if a[i] < 0:
                            continue
                        if d < a[i]:
                            leftarr.append(str(a[i])+"#"+str(i))
                        else:
                            rightarr.append(str(a[i])+"#"+str(i))
                    else:
                        b = str(a[i]).split("#")
                        if d < b[0]:
                            leftarr.append(str(a[i]))
                        else:
                            rightarr.append(str(a[i]))
                left = quick_sort_fn(leftarr)
                right = quick_sort_fn(rightarr)
            return left+mid+right

        #使用道格拉斯-普克算法，精简笔画
        inkarray = draw_image_cover(inkarray)
        #将笔画写入tfrecord文件
        self.covert_png_tfrecord_fn(inkarray,requestId)
        estimator = self._create_estimator_and_specs(
            run_config=tf.estimator.RunConfig(
                model_dir=self.FLAGS.model_dir,
                save_checkpoints_secs=300,
                save_summary_steps=100))
        rf = self.FLAGS.predict_data + str(requestId)
        result = estimator.predict(input_fn=get_input_fn(
            mode=tf.estimator.ModeKeys.EVAL,
            tfrecord_pattern=rf,
            batch_size=2))
        for r in result:
            logging.info(r['logits'])
            biggest = r['logits'][r['predictions']]
            if biggest > self.FLAGS.succ_logit:
                return json.dumps({'code':'000000',"result":[{"clazz":r['predictions'], "logit":biggest}]})
            else:
                logits = quick_sort_fn(r['logits'])
                logging.info(logits)
                result = []
                n = 0
                for i in logits:
                    ls = i.split("#")
                    result.append({"clazz":ls[0], "logit":ls[1]})
                    n += 1
                    if n == 4:
                        break
                if len(result) == 0:
                    return json.dumps({'code':'000000'})
                return json.dumps({'code':'000000',"result":result})










    def covert_png_tfrecord_fn(self,inkarray,requestId):
        '''write png quickdraw ink into tfrecords
            inkarray: quickdraw ink like:[[[1,2],[2,3]],[[3,4],[6,4]]]
        '''
        rf = self.FLAGS.predict_data + str(requestId)
        with tf.python_io.TFRecordWriter(rf) as writer:
            stroke_lengths = [len(stroke[0]) for stroke in inkarray]
            total_points = sum(stroke_lengths)
            np_ink = np.zeros((total_points, 3), dtype=np.float32)
            current_t = 0
            if not inkarray:
                return None, None
            for stroke in inkarray:
                if len(stroke[0]) != len(stroke[1]):
                    return None, None
                for i in [0, 1]:
                    np_ink[current_t:(current_t + len(stroke[0])), i] = stroke[i]
                current_t += len(stroke[0])
                np_ink[current_t - 1, 2] = 1  # stroke_end
            # Preprocessing.
            # 1. Size normalization.
            lower = np.min(np_ink[:, 0:2], axis=0)
            upper = np.max(np_ink[:, 0:2], axis=0)
            scale = upper - lower
            scale[scale == 0] = 1
            np_ink[:, 0:2] = (np_ink[:, 0:2] - lower) / scale
            # 2. Compute deltas.
            np_ink[1:, 0:2] -= np_ink[0:-1, 0:2]
            ink = np_ink[1:, :]
            features = {}
            features["class_index"] = tf.train.Feature(int64_list=tf.train.Int64List(
                value=[1]))
            features["ink"] = tf.train.Feature(float_list=tf.train.FloatList(
                value=ink.flatten()))
            features["shape"] = tf.train.Feature(int64_list=tf.train.Int64List(
                value=ink.shape))
            f = tf.train.Features(feature=features)
            example = tf.train.Example(features=f)
            writer.write(example.SerializeToString())

    def model_fn(self, features, labels, mode, params):
        """Model function for RNN classifier.
        This function sets up a neural network which applies convolutional layers (as
        configured with params.num_conv and params.conv_len) to the input.
        The output of the convolutional layers is given to LSTM layers (as configured
        The final state of the all LSTM layers are concatenated and fed to a fully
        connected layer to obtain the final classification scores.
        Args:
          features: dictionary with keys: inks, lengtself.
          labels: one hot encoded classes
          mode: one of tf.estimator.ModeKeys.{TRAIN, INFER, EVAL}
          params: a parameter dictionary with the following keys: num_layers,
            num_nodes, batch_size, num_conv, conv_len, num_classes, learning_rate.
        Returns:
          ModelFnOps for Estimator API.
        """

        def _get_input_tensors(features, labels):
            """Converts the input dict into inks, lengths, and labels tensors."""
            # features[ink] is a sparse tensor that is [8, batch_maxlen, 3]
            # inks will be a dense tensor of [8, maxlen, 3]
            # shapes is [batchsize, 2]
            shapes = features["shape"]
            # lengths will be [batch_size]
            lengths = tf.squeeze(
                tf.slice(shapes, begin=[0, 0], size=[params.batch_size, 1]))
            inks = tf.reshape(features["ink"], [params.batch_size, -1, 3])
            if labels is not None:
                labels = tf.squeeze(labels)
            return inks, lengths, labels

        def _add_conv_layers(inks, lengths):
            """Adds convolution layers."""
            convolved = inks
            for i in range(len(params.num_conv)):
                convolved_input = convolved
                if params.batch_norm:
                    convolved_input = tf.layers.batch_normalization(
                        convolved_input,
                        training=(mode == tf.estimator.ModeKeys.TRAIN))
                # Add dropout layer if enabled and not first convolution layer.
                if i > 0 and params.dropout:
                    convolved_input = tf.layers.dropout(
                        convolved_input,
                        rate=params.dropout,
                        training=(mode == tf.estimator.ModeKeys.TRAIN))
                convolved = tf.layers.conv1d(
                    convolved_input,
                    filters=params.num_conv[i],
                    kernel_size=params.conv_len[i],
                    activation=None,
                    strides=1,
                    padding="same",
                    name="conv1d_%d" % i)
            return convolved, lengths

        def _add_regular_rnn_layers(convolved, lengths):
            """Adds RNN layers."""
            if params.cell_type == "lstm":
                cell = tf.nn.rnn_cell.BasicLSTMCell
            elif params.cell_type == "block_lstm":
                cell = tf.contrib.rnn.LSTMBlockCell
            cells_fw = [cell(params.num_nodes) for _ in range(params.num_layers)]
            cells_bw = [cell(params.num_nodes) for _ in range(params.num_layers)]
            if params.dropout > 0.0:
                cells_fw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in cells_fw]
                cells_bw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in cells_bw]
            outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                cells_fw=cells_fw,
                cells_bw=cells_bw,
                inputs=convolved,
                sequence_length=lengths,
                dtype=tf.float32,
                scope="rnn_classification")
            return outputs

        def _add_cudnn_rnn_layers(convolved):
            """Adds CUDNN LSTM layers."""
            # Convolutions output [B, L, Ch], while CudnnLSTM is time-major.
            convolved = tf.transpose(convolved, [1, 0, 2])
            lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
                num_layers=params.num_layers,
                num_units=params.num_nodes,
                dropout=params.dropout if mode == tf.estimator.ModeKeys.TRAIN else 0.0,
                direction="bidirectional")
            outputs, _ = lstm(convolved)
            # Convert back from time-major outputs to batch-major outputs.
            outputs = tf.transpose(outputs, [1, 0, 2])
            return outputs

        def _add_rnn_layers(convolved, lengths):
            """Adds recurrent neural network layers depending on the cell type."""
            if params.cell_type != "cudnn_lstm":
                outputs = _add_regular_rnn_layers(convolved, lengths)
            else:
                outputs = _add_cudnn_rnn_layers(convolved)
            # outputs is [batch_size, L, N] where L is the maximal sequence length and N
            # the number of nodes in the last layer.
            mask = tf.tile(
                tf.expand_dims(tf.sequence_mask(lengths, tf.shape(outputs)[1]), 2),
                [1, 1, tf.shape(outputs)[2]])
            zero_outside = tf.where(mask, outputs, tf.zeros_like(outputs))
            outputs = tf.reduce_sum(zero_outside, axis=1)
            return outputs

        def _add_fc_layers(final_state):
            """Adds a fully connected layer."""
            return tf.layers.dense(final_state, params.num_classes)

        # Build the model.
        inks, lengths, labels = _get_input_tensors(features, labels)
        convolved, lengths = _add_conv_layers(inks, lengths)
        final_state = _add_rnn_layers(convolved, lengths)
        logits = _add_fc_layers(final_state)
        # Compute current predictions.
        predictions = tf.argmax(logits, axis=1)
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"logits": logits, "predictions": predictions},
        )

    def _create_estimator_and_specs(self,run_config):
        """Creates an Experiment configuration based on the estimator and input fn."""

        def get_num_classes():
            classes = []
            with tf.gfile.GFile(self.FLAGS.classes_file, "r") as f:
                classes = [x for x in f]
            num_classes = len(classes)
            return num_classes

        model_params = tf.contrib.training.HParams(
            num_layers=3,
            num_nodes=128,
            batch_size=2,
            num_conv=ast.literal_eval('[48, 64, 96]'),
            conv_len=ast.literal_eval('[5, 5, 3]'),
            num_classes=get_num_classes(),
            gradient_clipping_norm=9.0,
            cell_type='lstm',
            batch_norm=False,
            dropout=0.3)

        estimator = tf.estimator.Estimator(
            model_fn=self.model_fn,
            config=run_config,
            params=model_params)

        return estimator
