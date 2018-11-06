import os
import numpy as np
import tensorflow as tf
from random import shuffle
from PIL import Image
import sys

# import gpu_pipeline
import gpu_network
import parameters as pa
import label_loader

# FLAGS = tf.flags.FLAGS
# tf.flags.DEFINE_integer('batch_size', 15, "Batch size for training")


BATCH_SIZE = 10
LEARNING_RATE = 0.001


def build_training_ops(loss_tensor):
    """
    Build training ops
    :param loss_tensor:
    :return: [loss_tensor, global_step, decaying_learning_rate, train_op, summary_op]
    """
    global_step = tf.Variable(0, trainable=False)
    
    decaying_learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step,
                                           50000, 0.1, staircase=True)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=decaying_learning_rate)
    grads = optimizer.compute_gradients(loss_tensor)
    train_op = optimizer.apply_gradients(grads, global_step=global_step)
    
    # Add summary for every gradient
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + "/gradient", grad)
            
    summary_op = tf.summary.merge_all()
    return [loss_tensor, global_step, decaying_learning_rate, train_op, summary_op]


def print_log(loss_num, g_step_num, lr_num, itr):
    INTERVAL = 1
    log_dict = {
        'Loss': loss_num,
        'Step': g_step_num,
        'Learning Rate': lr_num
    }
    if itr % INTERVAL == 0:
        print(log_dict)

def main(argv=None):
    print("Training with batch size: " + str(BATCH_SIZE))
    # Place Holder
    PH, PW = pa.PH, pa.PW
    HEAT_H, HEAT_W = pa.HEAT_H, pa.HEAT_W
    PCM_nhwc_holder = tf.placeholder(tf.float32, [BATCH_SIZE, HEAT_H, HEAT_W, 14])
    PAF_nhwc_holder = tf.placeholder(tf.float32, [BATCH_SIZE, HEAT_H, HEAT_W, 11 * 2])
    img_holder = tf.placeholder(tf.float32, [BATCH_SIZE, PH, PW, 3])
    # Entire network
    poseNet = gpu_network.PoseNet()
    loss_tensor = poseNet.build_paf_pcm_loss(img_holder, PCM_nhwc_holder, PAF_nhwc_holder)
    lgdts_tensor = build_training_ops(loss_tensor)
    
    # Session Saver summary_writer
    sess = tf.Session()
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state("logs/")
    if ckpt:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        # Global initializer
        sess.run(tf.global_variables_initializer())
        pa.create_necessary_folders()

    summary_writer = tf.summary.FileWriter("logs/summary", sess.graph)
    
    # Close the graph so no op can be added
    tf.get_default_graph().finalize()
    # Load samples from disk
    gen_PCM_PAF_IMG = label_loader.generator_PCM_PAF_IMG(BATCH_SIZE, (512, 512), 8)

    for itr in range(1, int(1e7)):
        BC, BA, BI = next(gen_PCM_PAF_IMG)
        feed_dict = {img_holder: BI, PCM_nhwc_holder: BC, PAF_nhwc_holder: BA}
        loss_num, g_step_num, lr_num, train_op = sess.run(lgdts_tensor[0:4], feed_dict=feed_dict)
        print_log(loss_num, g_step_num, lr_num, itr)
        
        # Summary
        if itr % 100 == 0:
            summary_str = sess.run(lgdts_tensor[4], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, g_step_num)

            saver.save(sess, "logs/ckpt")
            print('Model Saved.')

    sess.close()


# Enter main
assert sys.version_info >= (3,5)
if __name__ == "__main__":
    tf.app.run()
exit(0)
