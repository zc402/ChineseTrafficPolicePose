import video_utils
import tensorflow as tf
import sys
import parameters as pa
import gpu_pipeline
import gpu_network
import numpy as np
import matplotlib.pyplot as plt
import skvideo.io
import os
import pysrt

LEARNING_RATE = 0.0004
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('mode', "train", "Train or test mode")

def build_training_ops(loss_tensor):
    """
    Build training ops
    :param loss_tensor:
    :return: [loss_tensor, global_step, decaying_learning_rate, train_op, summary_op]
    """
    global_step = tf.Variable(0, trainable=False)
    
    decaying_learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step,
                                                        20000, 0.8, staircase=True)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=decaying_learning_rate)
    grads = optimizer.compute_gradients(loss_tensor)
    train_op = optimizer.apply_gradients(grads, global_step=global_step)
    
    # Add summary for every gradient
    for grad, var in grads:
        if grad is not None:
            if 'rnn' in var.op.name or 'rconv' in var.op.name:  # Only summary rnn gradients
                tf.summary.histogram(var.op.name + "/gradient", grad)
    
    summary_op = tf.summary.merge_all()
    return [loss_tensor, global_step, decaying_learning_rate, train_op, summary_op]

def print_log(loss_num, g_step_num, lr_num, itr):
    INTERVAL = 1
    log_dict = {}
    log_dict['Loss'] = loss_num
    log_dict['Step'] = g_step_num
    log_dict['Learning Rate'] = lr_num
    if itr % INTERVAL == 0:
        print(log_dict)

def test_mode(sess, img_holder, btc_pred_max, batch_size):
    police_dict={0:"--", 1:"STOP", 2:"PASS", 3:"TURN LEFT", 4:"LEFT WAIT", 5:"TURN RIGHT", 6:"CNG LANE", 7:"SLOW DOWN", 8: "GET OFF"}
    metadata = skvideo.io.ffprobe(os.path.join(pa.VIDEO_FOLDER_PATH, "test.mp4"))
    total_frames = int(metadata["video"]["@nb_frames"])
    pred_list = []
    frames = skvideo.io.vread(os.path.join(pa.VIDEO_FOLDER_PATH, "test.mp4"))
    for i in range(0, total_frames-batch_size, batch_size):
        frames = frames / 255.
        feed_dict = {img_holder: frames[i:i+batch_size]}
        btc_pred_num = sess.run(btc_pred_max, feed_dict=feed_dict)
        pred = np.reshape(btc_pred_num, [-1])
        [pred_list.append(p) for p in pred]
        print("batch "+ str(i) +" done")
    file = pysrt.SubRipFile()
    for i, item in enumerate(pred_list):
        total_ms = round((1000/15) * i)
        total_s = total_ms // 1000
        total_m = total_s // 60
        start = '00:'+str(total_m % 60)+':'+str(total_s % 60)+':'+str(total_ms%1000)
        end = '00:'+str(total_m % 60)+':'+str(total_s % 60)+':'+str(total_ms%1000 + 14)
        sub = pysrt.SubRipItem(i, start=start, end=end, text=police_dict[int(item)])
        file.append(sub)
    file.save(os.path.join(pa.VIDEO_FOLDER_PATH, 'test.srt'))
    
def main(argv=None):
    TIME_STEP = 15
    NUM_CLASSES = 9

    img_holder = tf.placeholder(tf.float32, [TIME_STEP, pa.PH, pa.PW, 3])
    label_holder = tf.placeholder(tf.int32, [1, TIME_STEP])
    label_onehot = tf.one_hot(label_holder, NUM_CLASSES, axis=-1)
    
    poseNet = gpu_network.PoseNet()
    poseNet.set_var_trainable(False)
    paf_pcm = poseNet.inference_paf_pcm(img_holder)
    # Convolutional Pose Machine ended
    cpm_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    poseNet.set_var_trainable(True)
    poseNet.rnn_conv_input()
    loss_tensor, pred_tensor = poseNet.rnn_with_batch_one(label_onehot) # pred: [time_step, batch, n_classes]
    lgdts_tensor = build_training_ops(loss_tensor)
    
    # Session Saver summary_writer
    sess = tf.Session()
    cpm_saver = tf.train.Saver(var_list=cpm_var)
    cpm_ckpt = tf.train.get_checkpoint_state("logs/")
    if cpm_ckpt:
        cpm_saver.restore(sess, cpm_ckpt.model_checkpoint_path)
    else:
        raise FileNotFoundError("Tensorflow ckpt not found")
    
    all_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    rnn_var = [var for var in all_var if var not in cpm_var]
    rnn_saver = tf.train.Saver()
    rnn_ckpt = tf.train.get_checkpoint_state("rnn_logs/")
    if rnn_ckpt:
        rnn_saver.restore(sess, rnn_ckpt.model_checkpoint_path)
    else:
        sess.run(tf.variables_initializer(rnn_var))
    
    summary_writer = tf.summary.FileWriter("rnn_logs/summary", sess.graph)

    # model evaluation
    btc_pred = tf.transpose(pred_tensor, [1,0,2])
    btc_pred_max = tf.argmax(btc_pred, 2)
    l_max = tf.argmax(label_onehot, 2)
    correct_prediction = tf.equal(tf.argmax(btc_pred, 2), tf.argmax(label_onehot, 2))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    if 'test' in FLAGS.mode:
        test_mode(sess, img_holder, btc_pred_max, TIME_STEP)
        sess.close()
        exit(0)

    # Load video
    vgen = video_utils.video_frame_class_gen(1, TIME_STEP)
    for itr in range(1, int(1e7)):
        batch_time_frames, batch_time_labels = next(vgen)
        batch_img = np.reshape(batch_time_frames, [TIME_STEP, pa.PH, pa.PW, 3])
        feed_dict = {img_holder: batch_img, label_holder: batch_time_labels}
        loss_num, g_step_num, lr_num, train_op = sess.run(lgdts_tensor[0:4], feed_dict=feed_dict)
        print_log(loss_num, g_step_num, lr_num, itr)
    
        # Summary
        if itr % 50 == 0:
            btc_pred_num, l_max_num, acc = sess.run([btc_pred_max, l_max, accuracy], feed_dict=feed_dict)
            print("pred:  "+str(btc_pred_num))
            print("label: "+str(l_max_num))
            print("accuracy: " + str(acc))
            # summary_str = sess.run(lgdts_tensor[4], feed_dict=feed_dict)
            # summary_writer.add_summary(summary_str, g_step_num)
        
            rnn_saver.save(sess, "rnn_logs/ckpt")
            print('Model Saved.')

    sess.close()
    
    pass

if __name__ == "__main__":
    tf.app.run()
exit(0)
