from __future__ import division
import matplotlib.pyplot as plt
import scipy.io
import skimage.io
import skimage.transform
import os
import numpy as np
import tensorflow as tf
from random import shuffle
import nets
from PIL import Image
import dataset_util as du

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('mode', 'train', "Mode train/ test")
MAX_EPOCH = 50

BATCH_SIZE = 10
LEARNING_RATE = 0.0005
log_dict = {}


# Fetch samples from shuffled sample list
def samples_generator():
    """Return (img,heat)  heat 0-5: pcm, 6-15: paf"""
    label_list, _ = du.load_labels_from_disk()
    for epoch in range(0, MAX_EPOCH):
        # Current Epoch
        log_dict['Epoch'] = str(epoch)
        shuffle(label_list)
        # Process single image
        for num, mpi_sample in enumerate(label_list):
            pcm_paf = du.get_gaussian_paf_gt(du.HEAT_H, du.HEAT_W, mpi_sample)
            img_heat_list = du.prepare_network_input(mpi_sample, pcm_paf)
            # Process single person
            for img_heat in img_heat_list:
                yield(img_heat)
    yield None


def main(argv=None):
    # Holder tensor for images and labels
    image_holder = tf.placeholder(tf.float32, shape=[None, du.IN_H, du.IN_W, 3], name="input_image")
    pcm_holder = tf.placeholder(tf.float32, shape=[None, du.IN_HEAT_H, du.IN_HEAT_W, 6], name='pcm_holder')
    paf_holder = tf.placeholder(tf.float32, shape=[None, du.IN_HEAT_H, du.IN_HEAT_W, 10], name='paf_holder')
    # Build netowrk
    pose_net = nets.PoseNet()
    conv_4_2 = pose_net.vgg_10(image_holder, trainable=False)
    pose_net.inference_pose(conv_4_2)
    # Build loss tensor
    total_loss = pose_net.loss_l1_l2(pcm_holder, paf_holder, BATCH_SIZE)

    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    grads = optimizer.compute_gradients(total_loss)

    # Summary
    nets.add_gradient_summary(grads)
    # img_summary_op = person_predictor.add_img_summary()
    summary_op = tf.summary.merge_all()
    train_op = optimizer.apply_gradients(grads, global_step=global_step)
    # Session and Saver
    sess = tf.Session()
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state("logs/")
    if ckpt:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        # Global initializer
        sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter("logs/summary", sess.graph)
    img_summary_writer = tf.summary.FileWriter("logs/summary-img")
    # Load samples from disk
    samples_gen = samples_generator()
    # Start Feeding the network
    itr = 0
    while True:
        # Fetch images for a batch
        batch_images, batch_pcm, batch_paf = ([], [], [])

        # Construct a batch
        for i in range(0, BATCH_SIZE):
            img_heat = next(samples_gen)
            if img_heat is None:  # Reached MAX_EPOCH
                print("Done Training.")
                sess.close()
                exit(0)  # End the training
            img = img_heat[0]
            heat = img_heat[1]  # shape: (pcm*paf, H, W)
            heat = np.transpose(heat, axes=[1, 2, 0])  # shape: (H, W, pcm*paf)
            batch_pcm.append(heat[:, :, :6])
            batch_paf.append(heat[:, :, 6:])
            batch_images.append(img)

        # Feed the network
        # if FLAGS.mode == "train":

        feed_dict = {image_holder: batch_images, pcm_holder: batch_pcm, paf_holder: batch_paf}
        sess.run(train_op, feed_dict)

        if itr % 5 == 0:
            train_loss, summary_str = sess.run([total_loss, summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, sess.run(global_step))
            # Test images
            # test_img_list = []
            # for i in range(BATCH_SIZE):
            #     test_img_list.append(next(test_img_gen))
            # test_feed = {image_holder: test_img_list}
            # summary_img_str = sess.run(img_summary_op, feed_dict=test_feed)
            # img_summary_writer.add_summary(summary_img_str, sess.run(global_step))

        if itr % 5 == 0:
            saver.save(sess, "logs/ckpt")

        log_dict['itr'] = itr

        if itr % 1 == 0:
            print(log_dict)

        itr += 1


if __name__ == "__main__":
    tf.app.run()


exit(0)
