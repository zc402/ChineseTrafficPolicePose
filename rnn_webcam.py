
import parameters as pa
import tensorflow as tf
import gpu_network
import rnn_network
import cv2
import numpy as np
import evaluation


def main(argv=None):
    evaluate = evaluation.build_evaluation_network()
    
    cap = cv2.VideoCapture(0)
    # Camera output
    cam_out = np.zeros([512, 512, 3], dtype=np.uint8)
    rgb_norm = np.zeros([1, 512, 512, 3], dtype=np.float32) # rgb/255.
    # Prediction output
    map_h = 512
    map_w = 512
    sk_out = np.zeros([map_h, map_w, 3], np.uint8)
    # Analyzed data
    ana_out = np.zeros([512, 512, 3], dtype=np.uint8)
    final_out = np.zeros([512, 512*3, 3], dtype=np.uint8)
    assert cap.isOpened(),"Camera not enabled."
    while cap.isOpened():
        ret, frame = cap.read()

        
        
        cam_out[0:480, 0:512, :] = frame[0:480, 0:512, :]
        sk_out.fill(0)
        ana_out.fill(0)
        
        rgb = cv2.cvtColor(cam_out, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32)
        np.true_divide(rgb, 255., out=rgb_norm)

        pred, pcm, joint_xy, lsc18 = evaluate(rgb_norm)
        
        # Heatmap image
        heatmap_out = np.sum(pcm[0], axis=2) * 255.
        heatmap_out.astype(np.uint8)
        # Skeleton image
        b_colors = [(255, 0, 0), (255, 128, 0), (255, 255, 0), (64, 255, 0),
                    (0, 255, 255), (0, 0, 255), (255, 0, 255)]
        b_colors_bgr = [(b, g, r) for r, g, b in b_colors]

        # Inside one image
        for b_num, (b1, b2) in enumerate(pa.bones):
            if np.less(
                joint_xy[b1, :],
                0).any() or np.less(
                joint_xy[b2, :],
                0).any():
                continue  # no detection
            x1 = int(joint_xy[b1, 0] * map_w)
            y1 = int(joint_xy[b1, 1] * map_h)
            x2 = int(joint_xy[b2, 0] * map_w)
            y2 = int(joint_xy[b2, 1] * map_h)
            cv2.line(sk_out, (x1, y1), (x2, y2), b_colors_bgr[b_num], 4)
        


        # Draw Extracted 18 features
        cv2.putText(ana_out, "Relative Length of Bones", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        for n, l in enumerate(np.concatenate(([1.0], lsc18[0, ::3]))):
            x2 = x1 = 20 + n*30
            y1 = 60
            y2 = int(l*100.) + y1
            cv2.line(ana_out, (x1, y1), (x2, y2), b_colors_bgr[n], 4)
            cv2.putText(ana_out, "%1.1f" % l, (x2-10, y1+130), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255),
                        1)

        cv2.putText(ana_out, "Sine Values", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        for n, l in enumerate(lsc18[0, 1::3]):
            x2 = x1 = 20 + n * 30
            y1 = 270 + 50
            y2 = int(l * 50.) + y1
            cv2.line(ana_out, (x1, y1), (x2, y2), b_colors_bgr[n+1], 4)
            cv2.putText(ana_out, "%1.1f" % l, (x2-10, y1 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255),
                        1)
        
        cv2.putText(ana_out, "Cosine Values", (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        for n, l in enumerate(lsc18[0, 2::3]):
            x2 = x1 = 250 + n * 30
            y1 = 270 + 50
            y2 = int(l * 50.) + y1
            cv2.line(ana_out, (x1, y1), (x2, y2), b_colors_bgr[n+1], 4)
            cv2.putText(ana_out, "%1.1f" % l, (x2-10, y1 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255),
                        1)
        pred_text = pa.police_dict[pred[0]]
        print(pred_text)
        # cv2.putText(sk_out, pred_text, (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
        np.concatenate((cam_out, sk_out, ana_out), axis=1, out=final_out)
        cv2.imshow('webcam', final_out)
        # cv2.imshow('Camera', cam_out)
        # cv2.imshow('Skeleton', sk_out)
        # cv2.imshow('Analyzation', ana_out)
        cv2.imshow('Heatmaps', heatmap_out)
        key = cv2.waitKey(5)
        if key == 27:  # Esc key to stop
            evaluate(None)
            break

    cap.release()
    cv2.destroyAllWindows()


    exit(0)

if __name__ == "__main__":
    tf.app.run()

