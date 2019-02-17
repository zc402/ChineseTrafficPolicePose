import glob
import video_utils
import tensorflow as tf
import sys
import parameters as pa
import rnn_network
import argparse
import numpy as np
import os
import video_utils as vu
import metrics.edit_distance as ed
import itertools

def infer_npy(npy_path):
    tjc = np.load(npy_path)
    btjc = tjc[np.newaxis]

    BATCH_SIZE = 1
    TIME_STEP = tjc.shape[0]
    NUM_GESTURE_CLASSES = 9  # Not used

    # batch_time_feature holder:
    tf_btf = tf.placeholder(tf.float32, [BATCH_SIZE, TIME_STEP, 30])  # 10 length + 20 angle
    # batch_time label(classes) holder
    tf_btl = tf.placeholder(tf.int32, [BATCH_SIZE, TIME_STEP])
    with tf.variable_scope("rnn-net"):
        # b t c(0/1)
        btl_onehot = tf.one_hot(tf_btl, NUM_GESTURE_CLASSES, axis=-1)
        pred, state = rnn_network.build_rnn_network(tf_btf, NUM_GESTURE_CLASSES, training=False)

    sess = tf.Session()

    rnn_saver = tf.train.Saver()
    rnn_ckpt = tf.train.get_checkpoint_state("rnn_logs/")
    if rnn_ckpt:
        rnn_saver.restore(sess, rnn_ckpt.model_checkpoint_path)
    else:
        raise RuntimeError("No check point save file.")

    # summary_writer = tf.summary.FileWriter("rnn_logs/summary", sess.graph)

    # model evaluation
    btc_pred = tf.transpose(pred, [1, 0, 2])  # TBC to BTC
    bt_pred = tf.argmax(btc_pred, 2)
    # l_max = tf.argmax(btl_onehot, 2)
    correct_prediction = tf.equal(tf.argmax(btc_pred, 2), tf.argmax(btl_onehot, 2))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("Training with batch size:" + str(BATCH_SIZE))
    # Load joint pos

    # Build network

    btf = rnn_network.extract_bone_length_joint_angle(btjc)
    feed_dict = {tf_btf: btf}

    bt_pred_num = sess.run(
        bt_pred,
        feed_dict=feed_dict)
    print("pred:  " + str(bt_pred_num))

    sess.close()

    def save_label(label_list, csv_file):
        """

        :param label_list: a list of int
        :param csv_file:
        :return:
        """
        str_list = ["%d" % e for e in label_list]
        str_line = ",".join(str_list)

        with open(csv_file, 'w') as label_file:
            label_file.write(str_line)
        print("saved: %s" % csv_file)

    save_name = os.path.basename(npy_path).replace(".npy", ".csv")
    save_path = os.path.join(pa.RNN_PREDICT_OUT_FOLDER, save_name)
    save_label(bt_pred_num[0], save_path)

def predict_from_test_folder():
    csv_list_test = glob.glob(os.path.join(pa.LABEL_CSV_FOLDER_TEST, "*.csv"))
    feature_files_test = [os.path.basename(c).replace(".csv",".npy") for c in csv_list_test]
    feature_paths_test = [os.path.join(pa.RNN_SAVED_JOINTS_FOLDER, f) for f in feature_files_test]
    for npy_f in feature_paths_test:
        print("predicting %s" % npy_f)
        tf.reset_default_graph()
        infer_npy(npy_f)

def run_edit_distance_on_predict_out():
    labels = glob.glob(os.path.join(pa.LABEL_CSV_FOLDER_TEST, "*.csv"))
    sum_n, sum_i, sum_d, sum_s = 0, 0, 0, 0
    for label in labels:
        pred_name = os.path.basename(label)
        pred_path = os.path.join(pa.RNN_PREDICT_OUT_FOLDER, pred_name)
        # label: ground truth path
        # pred_path: predicted gestures path
        gt_label = vu.load_label(label)
        pred_label = vu.load_label(pred_path)
        pred_label = pred_label[pa.LABEL_DELAY_FRAMES:]  # Detection not stable at first few frames
        gt_group = itertools.groupby(gt_label)
        gt_group = [k for k, g in gt_group]
        pred_group = itertools.groupby(pred_label)
        pred_group = [k for k, g in pred_group]
        S, D, I = ed.SDI(pred_group, gt_group)
        N = len(gt_group)
        acc = (N - I - D - S) / N
        print("%s - N:%d S:%d, D:%d, I:%d, ACC:%.4f" % (pred_name, N, S, D, I, acc))
        # Sum 
        sum_n = sum_n + N
        sum_i = sum_i + I
        sum_d = sum_d + D
        sum_s = sum_s + S
    sum_acc = (sum_n-sum_i-sum_d-sum_s) / sum_n
    print("OVERALL - N:%d S:%d, D:%d, I:%d, ACC:%.4f" % (sum_n, sum_s, sum_d, sum_i, sum_acc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='detect gestures')
    parser.add_argument("-p", help="Predict videos from test folder", default=False, action="store_true")
    parser.add_argument("-e", help="Compute Edit Distance of predicted labels and ground truth labels", default=False, action="store_true")
    
    args = parser.parse_args()
    if args.p:
        predict_from_test_folder()
    elif args.e:
        run_edit_distance_on_predict_out()
    else:
        print("Please specify an argument.")
