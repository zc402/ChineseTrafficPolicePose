import video_utils
import tensorflow as tf
import sys
import parameters as pa
import rnn_network
import numpy as np
import os

LEARNING_RATE = 0.001
FLAGS = tf.flags.FLAGS

def build_training_ops(loss_tensor):
    """
    Build training ops
    :param loss_tensor:
    :return: [loss_tensor, global_step, decaying_learning_rate, train_op, summary_op]
    """
    global_step = tf.Variable(0, trainable=False)

    decaying_learning_rate = tf.train.exponential_decay(
        LEARNING_RATE, global_step, 10000, 0.8, staircase=True)

    optimizer = tf.train.AdamOptimizer(learning_rate=decaying_learning_rate)
    grads = optimizer.compute_gradients(loss_tensor)
    train_op = optimizer.apply_gradients(grads, global_step=global_step)

    # Add summary for every gradient
    for grad, var in grads:
        if grad is not None:
            if 'rnn' in var.op.name or 'rconv' in var.op.name:  # Only summary rnn gradients
                tf.summary.histogram(var.op.name + "/gradient", grad)

    summary_op = tf.summary.merge_all()
    return [loss_tensor, global_step,
            decaying_learning_rate, train_op, summary_op]


def print_log(loss_num, g_step_num, lr_num, itr):
    INTERVAL = 10
    log_dict = {}
    log_dict['Loss'] = loss_num
    log_dict['Step'] = g_step_num
    log_dict['Learning Rate'] = lr_num
    if itr % INTERVAL == 0:
        print(log_dict)


def main(argv=None):

    BATCH_SIZE = 4
    TIME_STEP = 15 * 90
    NUM_GESTURE_CLASSES = 9  # 8 classes + 1 for no move

    # batch_time_feature holder:
    tf_btf = tf.placeholder(tf.float32, [BATCH_SIZE, TIME_STEP, 30])  # 10 length + 20 angle
    # batch_time label(classes) holder
    tf_btl = tf.placeholder(tf.int32, [BATCH_SIZE, TIME_STEP])
    tf_is_training = tf.placeholder(tf.bool)
    with tf.variable_scope("rnn-net"):
        # b t c(0/1)
        btl_onehot = tf.one_hot(tf_btl, NUM_GESTURE_CLASSES, axis=-1)
        pred, state = rnn_network.build_rnn_network(tf_btf, NUM_GESTURE_CLASSES, training=tf_is_training)
        loss = rnn_network.build_rnn_loss(pred, btl_onehot)
        lgdts_tensor = build_training_ops(loss)

    sess = tf.Session()

    all_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    rnn_var = [var for var in all_var]
    rnn_saver = tf.train.Saver()
    rnn_ckpt = tf.train.get_checkpoint_state("rnn_logs/")
    if rnn_ckpt:
        rnn_saver.restore(sess, rnn_ckpt.model_checkpoint_path)
    else:
        sess.run(tf.variables_initializer(rnn_var))

    # summary_writer = tf.summary.FileWriter("rnn_logs/summary", sess.graph)

    # model evaluation
    btc_pred = tf.transpose(pred, [1, 0, 2])  # TBC to BTC
    btc_pred_max = tf.argmax(btc_pred, 2)
    l_max = tf.argmax(btl_onehot, 2)
    correct_prediction = tf.equal(tf.argmax(btc_pred, 2), tf.argmax(btl_onehot, 2))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("Training with batch size:"+str(BATCH_SIZE))
    # Load joint pos

    for itr in range(1, int(4e4)):

        btjc, btl = video_utils.random_btjc_btl(BATCH_SIZE, TIME_STEP)
        btf = rnn_network.extract_bone_length_joint_angle(btjc)
        feed_dict = {tf_btl: btl, tf_btf: btf, tf_is_training: True}
        loss_num, g_step_num, lr_num, train_op = sess.run(
            lgdts_tensor[0:4], feed_dict=feed_dict)
        print_log(loss_num, g_step_num, lr_num, itr)

        # Summary
        if itr % 50 == 1:
            btjc_test, btl_test = video_utils.random_btjc_btl(BATCH_SIZE, TIME_STEP, use_test_folder=True)
            btf_test = rnn_network.extract_bone_length_joint_angle(btjc_test)
            feed_dict_test = {tf_btl: btl_test, tf_btf: btf_test, tf_is_training: False}
            btc_pred_num, l_max_num, acc = sess.run(
                [btc_pred_max, l_max, accuracy],
                feed_dict=feed_dict_test)
            print("pred:  "+str(btc_pred_num))
            print("label: "+str(l_max_num))
            print("accuracy: " + str(acc))
            # summary_str = sess.run(lgdts_tensor[4], feed_dict=feed_dict)
            # summary_writer.add_summary(summary_str, g_step_num)

        if itr % 100 == 0:
            rnn_saver.save(sess, "rnn_logs/ckpt")
            print('Model Saved.')

    sess.close()

assert sys.version_info >= (3,5)
if __name__ == "__main__":
    tf.app.run()
exit(0)
