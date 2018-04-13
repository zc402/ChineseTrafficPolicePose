import video_utils
import tensorflow as tf
import sys
import parameters
import gpu_pipeline
import gpu_network
import numpy as np


assert sys.version_info >= (3,5)

BATCH_SIZE = 15
def save_evaluated_heatmaps():
    # Place Holder
    PH, PW = parameters.PH, parameters.PW
    ipjc_holder = tf.placeholder(tf.float32, [BATCH_SIZE, 8, 6, 3])
    img_holder = tf.placeholder(tf.float32, [BATCH_SIZE, PH, PW, 3])
    # Entire network
    paf_pcm_tensor = gpu_network.PoseNet().inference_paf_pcm(img_holder)
    
    # Session Saver summary_writer
    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state("logs/")
        if ckpt:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise FileNotFoundError("Tensorflow ckpt not found")
    
        # Close the graph so no op can be added
        tf.get_default_graph().finalize()
        # Load samples from disk
        samples_gen = gpu_pipeline.training_samples_gen(BATCH_SIZE)
        for itr in range(1, int(1e7)):
            batch_img, batch_ipjc = next(samples_gen)
            feed_dict = {img_holder: batch_img}
            paf_pcm = sess.run(paf_pcm_tensor, feed_dict=feed_dict)
            print(paf_pcm.shape)
        
        
        

def main(argv=None):
    save_evaluated_heatmaps()

if __name__ == "__main__":
    tf.app.run()
exit(0)
