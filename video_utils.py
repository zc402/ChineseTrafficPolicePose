import skvideo.io
import json
from skimage.viewer import ImageViewer
import pysrt
import sys

assert sys.version_info >= (3,5)

video = skvideo.io.vread("dataset/policepose_video/20180412.mp4")#,num_frames=4*1800)
print(video.shape)
metadata = skvideo.io.ffprobe("dataset/policepose_video/20180412.mp4")
print(metadata.keys())
print(json.dumps(metadata["video"], indent=4))
# viewer = ImageViewer(video[-1])
# viewer.show()

def class_per_frame(srt, total_frames):
    """
    Convert srt subtitle to class per frame array assuming constant frame rate of 30
    :param srt: subtitle path
    :return: class per frame array
    """
    subs = pysrt.open(srt)
    # Time of each frame (Millisecond)
    time_of_frame_list = [time / 30 * 1000 for time in range(total_frames)]
    
    def class_of_one_frame(frame_num):
        """
        :param frame_num:
        :return: class of designated frame
        """
        for sub in subs:
            if sub.start.ordinal < time_of_frame_list[frame_num] < sub.end.ordinal:
                return sub.text_without_tags
        # No subtitle annotated
        return "0"
    
    frame_class_list = [[num, class_of_one_frame(num)] for num in range(total_frames)]
    
    pass

class_per_frame("dataset/policepose_video/20180412.srt", 10000)

    
