ChineseTrafficPolicePose is a network that classify
**8 kinds of Chinese traffic police commanding poses** by analyzing visual information.
</br>ChineseTrafficPolicePose 是一个仅依靠视觉信息区分８种中国交警指挥手势的网络
<p align="center">
    <img src="doc/media/real-time.gif", width="480">
</p>

**Watch Videos**:
- [Frame by frame detection - Youtube Video](https://youtu.be/DmKFpD1K7gQ)
- [Realtime detection - Youtube Video](https://youtu.be/EjHp2RPuZqc)

**Environment**
- Tested on Ubuntu 16.04 with laptop GPU gtx980m.
- There must be **only one person** inside the video. Multiperson support is under developing.

**Quick Start**
- Download model file: [model - Jianguoyun](https://www.jianguoyun.com/p/DTxk84UQ9_LMBhiN1VU), extract and place 2 folders to policepose/logs and policepose/rnn_logs accrodingly.
- Run <code>python3 evaluation.py</code> to test on laptop camera.
- To use OpenCV camera with python3, you might need to compile OpenCV from source.

