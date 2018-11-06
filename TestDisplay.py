import cv2
from matplotlib import pyplot as plt
import label_loader

gen_PCM_PAF_IMG = label_loader.generator_PCM_PAF_IMG(2, (512, 512), 8)
BC,BA,BI = next(gen_PCM_PAF_IMG)
