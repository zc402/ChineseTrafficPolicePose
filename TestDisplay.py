import cv2
from matplotlib import pyplot as plt

import label_loader
import bidirectional_resize as bir

llist = label_loader.load_aicha_to_list("/media/zc/Ext4-1TB/AI_challenger_keypoint")
print(len(llist))
l = llist[0]
path, anno = l

img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

r_img, rec = bir.resize_img(img, (512, 512))

label_loader.anno_resize(anno, (512, 512), rec)
jwh = label_loader.heatmap_label(anno, (512, 512), 8)
