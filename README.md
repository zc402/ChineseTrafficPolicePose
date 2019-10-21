ChineseTrafficPolicePose is a network that classify
**8 kinds of Chinese traffic police commanding poses** by analyzing visual information.

ChineseTrafficPolicePose 是一个仅依靠视觉信息区分８种中国交警指挥手势的网络

# Paper Available Online 19 October 2019
https://doi.org/10.1016/j.neucom.2019.07.103

# Police Gesture Dataset
We publish the **Police Gesture Dataset**, which contains the videos of Chinese traffic police commanding gestures, and ground truth gesture labels for each video frame.

Police Gesture Dataset Download link: [Google Drive](https://drive.google.com/drive/folders/13KHZpweTE1vRGAMF7wqMDE35kDw40Uym?usp=sharing)

# Police Gesture Recognizer

**Notice: This gif is outdated. current version support prediction for FULL BODY, include legs. Check the videos in our dataset for examples of supported videos.**

<p align="center">
    <img src="doc/media/real-time.gif", width="480">
</p>

**Watch Videos**:
- [Frame by frame detection - Youtube Video](https://youtu.be/DmKFpD1K7gQ)
- [Realtime detection - Youtube Video](https://youtu.be/EjHp2RPuZqc)

**Environment**
- Only support `Python3`
- Use `Tensorflow` with GPU support

**Training**
- Download keypoint dataset from AI Challenger (~20GB).
- Rename the downloaded 4 folders to `"train", "test_a", "test_b", "val"`.
- Extract downloaded dataset to `parameters.TRAIN_FOLDER`. You may change the content of this parameter according to your path.
- Run `python3 PAF_train.py` to train the keypoint network.
- Download our **Traffic Police Gesture** dataset (~2GB) according to **Dataset** section.
- Extract .csv files to `dataset/csv_train` and `dataset/csv_test`.
- Extract .mp4 files to `dataset/policepose_video`.
- Run `python3 PAF_detect.py dataset/policepose_video -a` to parse videos to skeletal data.
- Run `python3 rnn_train.py` to train LSTM using labels from `dataset/csv_train` and skeletal data from `./dataset/gen/rnn_saved_joints`.
- Run `python3 rnn_detect.py -p` to predict test videos using name list from `dataset/csv_test` and skeletal data from `./dataset/gen/rnn_saved_joints`.
- Run `Python3 rnn_detect.py -e` to print **Edit Distance** of predicted labels with ground truth labels from  `dataset/csv_test`.
