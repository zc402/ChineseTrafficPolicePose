import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.io
import os
from PIL import Image
import pickle
import skimage.io

MPI_LABEL_PATH = "./dataset/MPI/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat"
MPI_LABEL_OBJ_PATH = "./dataset/label_obj"
IMAGE_FOLDER_PATH = "./dataset/MPI/images"
PAF_FOLDER_PATH = "./dataset/MPI/heatmaps"


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
            img_size(img_w, img_h)
            annorect_list[]
                annorect
                    objpos(x,y)
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
            # img_h, img_w: height and width of original image
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
                for joint in range(joint_num):
                    annorect.joint_list.append(
                        (release['annolist'][0, imgidx]['annorect'][0, ridx]['annopoints'][0, 0]['point'][0]
                         ['id'][joint][0, 0],
                         release['annolist'][0, imgidx]['annorect'][0, ridx]['annopoints'][0, 0]['point'][0]
                         ['x'][joint][0, 0],
                         release['annolist'][0, imgidx]['annorect'][0, ridx]['annopoints'][0, 0]['point'][0]
                         ['y'][joint][0, 0])
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


label, test = load_labels_from_disk()
for num, mpi_sample in enumerate(label):
    pcm_paf = get_gaussian_paf_gt(47, 82, mpi_sample)
    print(str(num) + " " + mpi_sample.name)
