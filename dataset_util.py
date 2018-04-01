import numpy as np
import scipy
import scipy.ndimage.interpolation
import matplotlib.pyplot as plt
import scipy.io
import os
import os.path
from PIL import Image
import pickle
import skimage.io
import skimage.transform
import tensorflow as tf

MPI_LABEL_PATH = "./dataset/MPI/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat"
MPI_LABEL_OBJ_PATH = "./dataset/gen/label_obj"
IMAGE_FOLDER_PATH = "./dataset/MPI/images"
RESIZED_IMAGE_FOLDER_PATH = "./dataset/gen/resized_images"
PAF_FOLDER_PATH = "./dataset/gen/heatmaps"


def __get_pixels_between_points(p1, p2, img_shape, half_line_width=1.0):
    """Return a list of pixel (x,y) index between 2 points"""
    assert len(img_shape) == 2
    pixels = []
    p1p2 = p2-p1
    abs_p1p2 = np.linalg.norm(p1p2)
    for h in range(img_shape[0]):
        for w in range(img_shape[1]):

            p3 = np.asarray([w, h]).astype(np.float32)
            p1p3 = p3-p1

            d = abs(np.cross(p1p2, p1p3) / abs_p1p2)
            p = np.dot(p1p2, p1p3) / abs_p1p2  # projection length with sign

            if d < half_line_width and -1.9 <= p <= abs_p1p2 + 1.9:  # Constant: range of two end points
                pixels.append([w, h])
    return pixels


def __draw_part_affinity_field(p1, p2, img_shape):
    """
    Part Affinity Field for 2 joints
    draw unit vector of p1->p2 on the line on heatmap
    """
    heatmap = np.zeros(shape=[img_shape[0], img_shape[1], 2])
    p1p2 = p2 - p1
    abs_p1p2 = np.linalg.norm(p1p2)
    if abs_p1p2 == 0: return heatmap
    unit_vector = p1p2 / abs_p1p2
    pixels = __get_pixels_between_points(p1, p2, img_shape)
    for pixel in pixels:
        heatmap[pixel[1], pixel[0], :] = unit_vector
    return heatmap


def compute_limb_connection_evidence(p1, p2, vector_heatmap):
    """
    Evidence of two part belongs to one person
    """
    pixels = __get_pixels_between_points(p1, p2, [vector_heatmap.shape[0], vector_heatmap.shape[1]])
    p1p2 = p2 - p1
    abs_p1p2 = np.linalg.norm(p1p2)
    unit_vector = p1p2 / abs_p1p2
    pixels_value = []
    for pixel in pixels:
        pixels_value.append(vector_heatmap[pixel[1], pixel[0], :])
    pixel_unit_dot_products = [np.dot(p, unit_vector) for p in pixels_value]
    evidence = np.mean(pixel_unit_dot_products)
    return evidence


class MPISample:
    pass
class Annorect:
    pass


# Load all training labels from file
def load_labels_from_disk():
    """
    mpi_sample_list[]
        mpi_sample
            name
            img_size(x, y)
            annorect_list[]
                annorect
                    objpos(x, y)
                    joint_list[]
                        joint(id, x, y)
    """
    # If labels already saved in python object:
    if os.path.isfile(MPI_LABEL_OBJ_PATH):
        with open(MPI_LABEL_OBJ_PATH, 'rb') as fileInput:
            train_test_sample_list = pickle.load(fileInput)
            print("Labels Loaded from py obj")
            return train_test_sample_list

    # If labels not saved:
    err_count = 0
    mat = scipy.io.loadmat(MPI_LABEL_PATH)
    release = mat['RELEASE'][0, 0]
    sample_size = release['img_train'].shape[1]
    train_label_list = []
    test_label_list = []
    # imgidx: image idx
    for imgidx in range(0, sample_size):
        # mpi_sample: store mat information in python
        mpi_sample = MPISample()
        # anno_image_mat: all annotations of 1 image
        anno_image_mat = release['annolist'][0, imgidx]
        mpi_sample.name = anno_image_mat['image'][0, 0]['name'][0]
        try:
            # img_size: (x,y) length of original image
            image_path = os.path.join(IMAGE_FOLDER_PATH, mpi_sample.name)
            ori_img = Image.open(image_path)
            mpi_sample.img_size = ori_img.size

        except FileNotFoundError:
            continue

        try:
            # annorect_mat: body annotations of 1 image
            annorect_mat = anno_image_mat['annorect']
            mpi_sample.annorect_list = list()
            # ridx: person idx in 1 image
            if annorect_mat.shape[1] == 0:
                raise ValueError("no person rect annotation")
            # Skip if no person pos label
            for ridx in range(0, annorect_mat.shape[1]):
                annorect_person_mat = annorect_mat[0, ridx]
                annorect = Annorect()
                # objpos: rough human position in the image
                human_x = annorect_person_mat['objpos'][0, 0]['x'][0, 0]
                human_y = annorect_person_mat['objpos'][0, 0]['y'][0, 0]
                annorect.objpos = (human_x, human_y)
                # Joints
                annorect.joint_list = list()
                joint_num = release['annolist'][0, imgidx]['annorect'][0, ridx]['annopoints'][0, 0]['point'].shape[1]
                for joint_arr_id in range(joint_num):
                    joint_id = release['annolist'][0, imgidx]['annorect'][0, ridx]['annopoints'][0, 0]['point'][0]['id'][joint_arr_id][0, 0]
                    if joint_id == int(8) or joint_id == int(9):
                        continue  # Upper neck and head top, no visibility annotation
                    is_visible = release['annolist'][0, imgidx]['annorect'][0, ridx]['annopoints'][0, 0]['point'][0]['is_visible'][joint_arr_id][0, 0]
                    if is_visible == '1' or is_visible == 1:
                        annorect.joint_list.append(
                            (release['annolist'][0, imgidx]['annorect'][0, ridx]['annopoints'][0, 0]['point'][0]
                             ['id'][joint_arr_id][0, 0],
                             release['annolist'][0, imgidx]['annorect'][0, ridx]['annopoints'][0, 0]['point'][0]
                             ['x'][joint_arr_id][0, 0],
                             release['annolist'][0, imgidx]['annorect'][0, ridx]['annopoints'][0, 0]['point'][0]
                             ['y'][joint_arr_id][0, 0])
                    )
                # Add annorect to annorect_list
                mpi_sample.annorect_list.append(annorect)
            train_label_list.append(mpi_sample)
        except:
            # A field was not found in annotation
            err_count += 1
            test_label_list.append(mpi_sample)

    print("(Dump to test) Invalid samples: " + str(err_count))  # Total skipped images
    # Save to file
    with open(MPI_LABEL_OBJ_PATH, 'wb') as fileOutput:
        pickle.dump((train_label_list, test_label_list), fileOutput, pickle.HIGHEST_PROTOCOL)
    return train_label_list, test_label_list


def __draw_gaussian_heatmap(map_h, map_w, c_x, c_y):
    """Build gaussian heatmap with one center"""
    variance = 1
    heatmap = np.zeros([map_h, map_w])
    for idx in np.ndindex(map_h, map_w):
        dist_sq = (idx[0] - c_y) ** 2 + (idx[1] - c_x) ** 2
        exponent = dist_sq / 2.0 / variance / variance
        heatmap[idx[0], idx[1]] = np.exp(-exponent)
    return heatmap


def draw_joint_gaussian_heatmaps(map_h, map_w, mpi_sample):
    """
    Draw joint gaussian heatmaps for all people inside image
    Convert annotation scale inside the function
    10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist
    """
    # [0 - r wrist, 1 - r elbow, 2 - r shoulder, 3 - l shoulder, 4 - l elbow, 5 - l wrist]
    heatmaps = np.zeros([6, map_h, map_w])
    scale_x = map_w / mpi_sample.img_size[0]
    scale_y = map_h / mpi_sample.img_size[1]
    for annorect in mpi_sample.annorect_list:
        for joint in annorect.joint_list:
            jid = joint[0]
            if 10 <= jid <= 15:
                c_x = joint[1] * scale_x
                c_y = joint[2] * scale_y
                heatmaps[jid-10, :, :] += __draw_gaussian_heatmap(map_h, map_w, c_x, c_y)
    return heatmaps


def draw_part_affinity_fields(map_h, map_w, mpi_sample):
    """
    Draw heatmap of vectors between joints
    r_wrist -0-> r_elbow -1-> r shoulder -2-> l shoulder -3-> l_elbow -4-> l_wrist
    """
    heatmaps = np.zeros([5, map_h, map_w, 2])  # bones, h, w, vector
    scale_x = map_w / mpi_sample.img_size[0]
    scale_y = map_h / mpi_sample.img_size[1]
    # Find jid 10-11, 11-12, .., 14-15

    def find_idx_of_jid(joint_list, jid):
        for index, joint in enumerate(joint_list):
            if joint[0] == jid:
                return index
        return None

    for annorect in mpi_sample.annorect_list:
        joint_list = annorect.joint_list
        for i in range(10, 15):  # 10 11 12 13 14
            joint_A = find_idx_of_jid(joint_list, i)
            joint_B = find_idx_of_jid(joint_list, i+1)
            if joint_A and joint_B:  # Both 2 joints exists
                p1 = np.array([joint_list[joint_A][1] * scale_x,
                               joint_list[joint_A][2] * scale_y])
                p2 = np.array([joint_list[joint_B][1] * scale_x,
                               joint_list[joint_B][2] * scale_y])
                heatmaps[i-10, :, :] += __draw_part_affinity_field(p1, p2, (map_h, map_w))
    return heatmaps


def get_gaussian_paf_gt(map_h, map_w, mpi_sample):
    file_path = os.path.join(PAF_FOLDER_PATH, mpi_sample.name)
    # If maps already saved in python object:
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as fileInput:
            pcm_paf = pickle.load(fileInput)
            return pcm_paf
    else:  # If maps not saved:
        PCM = draw_joint_gaussian_heatmaps(map_h, map_w, mpi_sample)  # Part Confidence Maps
        PAF = draw_part_affinity_fields(map_h, map_w, mpi_sample)
        pcm_paf = (PCM, PAF)
        with open(file_path, 'wb') as fileOutput:
            pickle.dump(pcm_paf, fileOutput, pickle.HIGHEST_PROTOCOL)
        return pcm_paf


# Resize original image
PH, PW = (376, 656)
IN_H, IN_W = (376, 376)
HEAT_H, HEAT_W = (47, 82)
IN_HEAT_H, IN_HEAT_W = (47, 47)


def prepare_network_input(mpi_sample, pcm_paf):
    """Prepare input (square) for pose detection"""

    resized_image_path = os.path.join(RESIZED_IMAGE_FOLDER_PATH, mpi_sample.name)
    image_re = skimage.io.imread(resized_image_path)
    image = image_re / 255.0
    padded_image = np.pad(image, ((IN_H//2, IN_H//2), (IN_W//2, IN_W//2), (0, 0)), mode='constant', constant_values=0)

    paf_t = np.transpose(pcm_paf[1], [0, 3, 1, 2])
    paf_r = np.reshape(paf_t, newshape=[paf_t.shape[0] * paf_t.shape[1], paf_t.shape[2], paf_t.shape[3]])
    # paf_r [5*2, h, w]
    # 0-5: pcm, 6-15: paf
    heatmaps = np.concatenate([pcm_paf[0], paf_r], axis=0)
    padded_maps = np.pad(heatmaps, ((0, 0), (IN_HEAT_H//2, IN_HEAT_H//2), (IN_HEAT_W//2, IN_HEAT_W//2)), mode='constant', constant_values=0)

    scale_in_h = PH / mpi_sample.img_size[1]
    scale_in_w = PW / mpi_sample.img_size[0]
    scale_heat_h = heatmaps.shape[1] / mpi_sample.img_size[1]
    scale_heat_w = heatmaps.shape[2] / mpi_sample.img_size[0]

    img_heat_list = []
    for annorect in mpi_sample.annorect_list:
        # Image
        img_cxy = int(round(annorect.objpos[0] * scale_in_w)) + IN_H//2, int(round(annorect.objpos[1] * scale_in_h)) + IN_W//2
        assert IN_H % 2 == 0 and IN_W % 2 == 0  # the crop is designed for length which %2==0
        cropped_img = padded_image[img_cxy[1]-IN_H//2: img_cxy[1]+IN_H//2, img_cxy[0]-IN_W//2: img_cxy[0]+IN_W//2, :]
        assert cropped_img.shape == (IN_H, IN_W, padded_image.shape[2])
        # Heatmap
        heat_cxy = int(round(annorect.objpos[0] * scale_heat_w)) + IN_HEAT_W//2, int(round(annorect.objpos[1] * scale_heat_h)) + IN_HEAT_H//2
        assert IN_HEAT_H % 2 == 1 and IN_HEAT_W % 2 == 1  # The crop is designed to be so
        cropped_heat = padded_maps[:, heat_cxy[1]-IN_HEAT_H//2: heat_cxy[1]+IN_HEAT_H//2+1, heat_cxy[0]-IN_HEAT_W//2: heat_cxy[0]+IN_HEAT_W//2+1]
        assert cropped_heat.shape == (padded_maps.shape[0], IN_HEAT_H, IN_HEAT_W)
        img_heat_list.append((cropped_img, cropped_heat))

    img_heat_list = image_augment(img_heat_list)
    return img_heat_list


def image_augment(img_heat_list):
    """Random augmentation"""
    new_im_he_list = []
    for im_heat in img_heat_list:
        im = im_heat[0]
        heat = im_heat[1]

        # Flip !!flip will swap left and right
        # if np.random.choice(2) == 1:
           #  im = np.flip(im, axis=1)
           #  heat = np.flip(heat, axis=2)

        # Rotate
        angle = np.random.random() * 60 - 30  # -30 - 30 C
        im = scipy.ndimage.interpolation.rotate(im, angle, axes=(0, 1), reshape=False)
        heat = scipy.ndimage.interpolation.rotate(heat, angle, axes=(1, 2), reshape=False)

        im = np.clip(im, 0.0, 1.0)

        new_im_he = (im, heat)
        new_im_he_list.append(new_im_he)
    return new_im_he_list


def resize_imgs():
    t_labels, _ = load_labels_from_disk()
    for i, t_label in enumerate(t_labels):
        name = t_label.name
        file = os.path.join(IMAGE_FOLDER_PATH, name)
        im = Image.open(file)
        re_im = im.resize((PW, PH), Image.ANTIALIAS)
        out_path = os.path.join(RESIZED_IMAGE_FOLDER_PATH, name)
        re_im.save(out_path)
        print(str(i) + ' ' + name)


RESIZED_RATIO_KEPT = "./dataset/gen/ratio_kept"
# [Image][Person][Joint][x,y,mask]
IPJC_FILE = "./dataset/gen/ipjc.npy"
INAME_FILE = "./dataset/gen/iname.npy"
# [ImageName]
def resize_imgs_keep_ratio():
    """Resize with ratio unchanged"""
    train_labels, _ = load_labels_from_disk()
    target_ratio = PW / PH
    ipjc_list = list() # Image pjc
    iname_list = list() # Image name
    excluded = 0
    for i, t_label in enumerate(train_labels):
        name = t_label.name
        file = os.path.join(IMAGE_FOLDER_PATH, name)
        im = Image.open(file)
        np_im = np.asarray(im.convert("RGB"))
        ori_size = im.size
        ori_ratio = ori_size[0] / ori_size[1]
        if ori_ratio >= target_ratio:
            # Depends on width
            zoom_ratio = PW / ori_size[0]
            bg = np.zeros((int(ori_size[0] / target_ratio), ori_size[0], 3), np.uint8)
            bg = bg[:ori_size[1], :ori_size[0], :] + np_im[:, :, :]

        elif ori_ratio < target_ratio:
            # Depends on height
            zoom_ratio = PH / ori_size[1]
            bg = np.zeros((ori_size[1], int(ori_size[1] * target_ratio), 3), np.uint8)
            bg = bg[:ori_size[1], :ori_size[0], :] + np_im[:, :, :]

        re_im = Image.fromarray(bg, 'RGB')
        re_im = re_im.resize((PW, PH), Image.ANTIALIAS)
        out_path = os.path.join(RESIZED_RATIO_KEPT, name)
        # re_im.save(out_path)

        print(str(i) + ' ' + name)

        # Generate array labels
        if len(t_label.annorect_list) > 8: # More than 8 person
            excluded = excluded + 1
            continue
        single_image = np.zeros((8, 6, 3), dtype=np.float32)
        for j, anno in enumerate(t_label.annorect_list): # Each person
            for k, joint in enumerate(anno.joint_list): # Each joint
                if 10 <= joint[0] <= 15:
                    index = joint[0] - 10
                    single_image[j][index][0] = joint[1] * zoom_ratio
                    single_image[j][index][1] = joint[2] * zoom_ratio
                    single_image[j][index][2] = 1
        ipjc_list.append(single_image)
        iname_list.append(name)

    np.save(IPJC_FILE, np.asarray(ipjc_list))
    np.save(INAME_FILE, np.asarray(iname_list))
    print(str(excluded) + " excluded")
    pass

