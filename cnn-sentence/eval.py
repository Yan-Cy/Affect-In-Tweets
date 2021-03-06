#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
from sklearn.metrics import mean_squared_error

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("data_dir", "../datasets", "Data source for training.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/{}/vec200/checkpoints", "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

tasks = ['anger', 'fear', 'joy', 'sadness']

for task in tasks:
    print 'Running for task', task
    checkpoint_dir = FLAGS.checkpoint_dir.format(task)

    # CHANGE THIS: Load data. Load your own data here
    x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.data_dir, task, 'test')
    #y_test = np.argmax(y_test, axis=1)
    #print y_test

    # Map data into vocabulary
    #vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    #vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    #x_test = np.array(list(vocab_processor.transform(x_raw)))
    x_test, vocab_vector = data_helpers.build_vocabulary(x_raw)
    #np.save('tmp/x_test.data', x_test)
    #x_test = np.load('tmp/x_test.data.npy')
    #vocab_vector = np.load('tmp/vocab_vector.data.npy')

    print("\nEvaluating...\n")

    # Evaluation
    # ==================================================
    print checkpoint_dir
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            #predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            scores = graph.get_operation_by_name("output/scores").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            #all_predictions = []
            all_scores = []

            for x_test_batch in batches:
                batch_predictions = sess.run(scores, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                #all_predictions = np.concatenate([all_predictions, batch_predictions])
                #print batch_predictions
                batch_predictions = [x[0] for x in batch_predictions]
                all_scores = np.concatenate([all_scores, batch_predictions])



    print 'Mean Square Error:', mean_squared_error(all_scores, y_test) ** 0.5

    # Save
    with open('../datasets/{}-ratings-0to1.test.target.txt'.format(task)) as f:
        data = f.readlines()
    data = [x.strip().split('\t') for x in data]
    senid = np.array([x[0] for x in data])
    mood = np.array([x[2] for x in data])
    x_raw = np.array([x[1] for x in data])

    predictions_human_readable = np.column_stack((senid, x_raw, mood, all_scores))
    out_path = '../results/cnn/{}-pred.txt'.format(task)
    with open(out_path, 'w') as f:
        for x in predictions_human_readable:
            f.write('\t'.join(x) + '\n')


