
import parameters as pa
import tensorflow as tf
import gpu_network
import rnn_network
import cv2
import numpy as np
import evaluation_util

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('f', None, "path of file input")

def main(argv=None):

    
    if FLAGS.f is not None:
        _run_on_video_file()
    else:
        _run_on_camera()
    
    
def _run_on_video_file():
    evaluate = evaluation_util.build_evaluation_network()
    analytic_picture = evaluation_util.result_analyzer()
    
    cap = cv2.VideoCapture(FLAGS.f)

    while cap.isOpened():
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32)
        np.true_divide(rgb, 255., out=rgb)
        pred, pcm, joint_xy, lsc18 = evaluate(rgb[np.newaxis])
        final_out = analytic_picture(frame, pred, pcm, joint_xy, lsc18)
        
        cv2.imshow('frame', final_out)
        key = cv2.waitKey(1)
        if key == 27:  # Esc key to stop
            evaluate(None)
            break

    cap.release()
    cv2.destroyAllWindows()


def _run_on_camera():
    evaluate = evaluation_util.build_evaluation_network()
    analytic_picture = evaluation_util.result_analyzer()
    cap = cv2.VideoCapture(0)
    # Prediction output
    # Camera output
    square_image = np.zeros([512, 512, 3], dtype=np.uint8)

    assert cap.isOpened(), "Camera not enabled."
    while cap.isOpened():
        ret, frame = cap.read()
    
        rgb_norm = np.zeros([1, 512, 512, 3], dtype=np.float32)  # rgb/255.
    
        square_image[0:480, 0:512, :] = frame[0:480, 0:512, :]
    
        rgb = cv2.cvtColor(square_image, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32)
        np.true_divide(rgb, 255., out=rgb_norm)
    
        pred, pcm, joint_xy, lsc18 = evaluate(rgb_norm)
        final_out = analytic_picture(square_image, pred, pcm, joint_xy, lsc18)
    
        cv2.imshow('webcam', final_out)
    
        key = cv2.waitKey(5)
        if key == 27:  # Esc key to stop
            evaluate(None)
            break

    cap.release()
    cv2.destroyAllWindows()
    exit(0)

if __name__ == "__main__":
    tf.app.run()

