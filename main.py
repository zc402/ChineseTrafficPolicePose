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

    label_list, test = du.load_labels_from_disk()
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
    image_holder = tf.placeholder(tf.float32, shape=[None, PH, PW, 3], name="input_image")
    person_predictor = nets.PersonPredictor(image_holder)  # person inference network
    # Get output shape
    output_h, output_w = (person_predictor.output_shape[1], person_predictor.output_shape[2])
    heatmap_gt_holder = tf.placeholder(tf.float32, shape=[None, output_h, output_w, 1], name="person_heatmap_gt")
    # Build loss tensor
    person_predictor.build_loss(heatmap_gt_holder, BATCH_SIZE)
    global_step = tf.Variable(0, trainable=False)

    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    grads = optimizer.compute_gradients(person_predictor.total_loss)

    # Summary
    nets.add_gradient_summary(grads)
    img_summary_op = person_predictor.add_img_summary()
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
    test_img_gen = test_img_generator()
    # Start Feeding the network
    itr = 0
    while True:
        # Fetch images for a batch
        batch_images, batch_labels = ([], [])
        debug_batch_img_ori = []

        # Construct a batch
        for i in range(0, BATCH_SIZE):
            la_im = next(samples_gen)  # (label, image)
            if la_im is None:  # Reached MAX_EPOCH
                print("Done Training.")
                sess.close()
                exit(0)  # End the training
            else:
                batch_labels.append(la_im[0])
                batch_images.append(la_im[1])
                debug_batch_img_ori.append(la_im[2])

        # person location Gaussian heatmap
        batch_heatmap_gt = []
        for i in range(0, BATCH_SIZE):
            heatmap_scale_h = output_h / batch_labels[i].img_h
            heatmap_scale_w = output_w / batch_labels[i].img_w

            heatmap = gaussian_image(output_h, output_w, batch_labels[i], [heatmap_scale_h, heatmap_scale_w])

            batch_heatmap_gt.append(heatmap)

            # plt.figure(1)
            # plt.subplot(211)
            # plt.imshow(batch_heatmap_gt[i])
            # plt.subplot(212)
            # plt.imshow(debug_batch_img_ori[i])
            # plt.show()
            # print("a")

        batch_heatmap_gt = np.asarray(batch_heatmap_gt, np.float32)[:, :, :, np.newaxis]

        # Feed the network
        if FLAGS.mode == "train":
            feed_dict = {image_holder: batch_images, heatmap_gt_holder: batch_heatmap_gt}
        sess.run(train_op, feed_dict)

        if itr % 200 == 0:
            train_loss, summary_str = sess.run([person_predictor.total_loss, summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, sess.run(global_step))
            # Test images
            test_img_list = []
            for i in range(BATCH_SIZE):
                test_img_list.append(next(test_img_gen))
            test_feed = {image_holder: test_img_list}
            summary_img_str = sess.run(img_summary_op, feed_dict=test_feed)
            img_summary_writer.add_summary(summary_img_str, sess.run(global_step))

        if itr % 200 == 0:
            saver.save(sess, "logs/ckpt")

        if itr % 10 == 0:
            print("iteration: " + str(itr))

        itr += 1


if __name__ == "__main__":
    tf.app.run()


exit(0)
