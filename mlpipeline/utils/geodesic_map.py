import os
import GeodisTK
import FastGeodis
import time
import numpy as np
import SimpleITK as sitk
from PIL import Image
import torch
import torch.nn.functional as functional
from skimage import feature
from skimage.morphology import skeletonize, erosion, dilation, ball
import argparse


def geodesic_distance_3d(I, S, spacing, lamb=1, n_iters=4, mode='exact'):
    '''
    Get 3D geodesic disntance by raser scanning.
    I: input image array, can have multiple channels, with shape [D, H, W] or [D, H, W, C]
       Type should be np.float32.
    S: binary image where non-zero pixels are used as seeds, with shape [D, H, W]
       Type should be np.uint8.
    spacing: a tuple of float numbers for pixel spacing along D, H and W dimensions respectively.
    lamb: weighting betwween 0.0 and 1.0
          if lamb==0.0, return spatial euclidean distance without considering gradient
          if lamb==1.0, the distance is based on gradient only without using spatial distance
    iter: number of iteration for raster scanning.
    '''
    if mode == 'fast':
        I = torch.from_numpy(I).unsqueeze_(0).unsqueeze_(0).to('cuda')
        S = torch.from_numpy(1 - S.astype(np.float32)).unsqueeze_(0).unsqueeze_(0).to('cuda')
        v = 1e10
        gd = np.squeeze(FastGeodis.generalised_geodesic3d(I, S, spacing, v, lamb, n_iters).detach().cpu().numpy())
        return gd
    else:
        return GeodisTK.geodesic3d_raster_scan(I, S, spacing, lamb, n_iters)


def normalize_zero_to_one(var):
    var -= var.min()
    var /= var.max()
    return var


def background_geodesic_map(img, nb_classes=1):
    temp = np.zeros_like(img[0])
    count = 0
    for c in range(1,nb_classes):
        if img[c,...].sum() == 0:
            continue
        temp += img[c,...]
        count +=1

    if count == 0:
        print("No foreground object!!!!")
        img[0,...] = np.ones_like(img[c,...])
    else:
        img[0,...] = 1- temp/count
    return img


def invert_geodesic_maps(img, invert_type=None, gamma=1):
    if invert_type == "exp":
        img = np.exp(-img)
    elif invert_type == "exp_gamma":
        img = np.exp(- gamma * img)
    else:
        img = img.max()-img
    img = normalize_zero_to_one(img)
    return img


def get_data(input_name, is_seg=False):

    if not os.path.isfile(input_name):
        print("File not exists:", input_name)
        return -1

    img = sitk.ReadImage(input_name)
    np_img = sitk.GetArrayFromImage(img)
    spacing_raw = img.GetSpacing()
    if is_seg:
        return np.asarray(np_img, np.uint8), spacing_raw
    else:
        return np.asarray(np_img, np.float32), spacing_raw


def write_data4d(arr, save_path):
    img = sitk.GetImageFromArray(arr, isVector=False)
    sitk.WriteImage(img, save_path)


def mask_erosion(img, sizes=1):
    footprint = ball(sizes)
    e_img = erosion(img, footprint)
    return e_img


def mask_dilation(img, sizes=1):
    footprint = ball(sizes)
    e_img = dilation(img, footprint)
    return e_img


def get_skeleton(img):
    seed = np.zeros_like(img, np.uint8)
    seed = skeletonize(img)
    seed[seed>0] = 1
    return seed


def canny3d(img):
    seed = np.zeros_like(img, np.uint8)
    imgs = []
    for i in range(img.shape[0]):
        edge = feature.canny(img[i, ...])
        imgs.append(edge)

    img_edge = np.stack(imgs, axis=0)
    return img_edge


def get_fg_skeleton_seeds(seg, nb_classes=1):
    seeds = np.zeros((nb_classes, seg.shape[1], seg.shape[2], seg.shape[3]), np.uint8)
    mask = seg.copy()
    for c in range(0, nb_classes):
        seeds[c] = get_skeleton(mask[c])

    return seeds


def get_boundary_seeds(seg, nb_classes):
    is_onehot = seg.shape[0] == nb_classes
    assert is_onehot
    if is_onehot:
        seeds = np.zeros((nb_classes, seg.shape[1], seg.shape[2], seg.shape[3]), np.uint8)
    else:
        seeds = np.zeros((nb_classes, seg.shape[0], seg.shape[1], seg.shape[2]), np.uint8)

    mask = seg.copy()
    start_index = int(not is_onehot)

    for c in range(start_index, nb_classes):
        if is_onehot:
            bg_mask = (mask[c] == 0)
        else:
            bg_mask = mask != c
        seeds[c] = canny3d(bg_mask)

    return seeds


def get_geodesic_distance(img_path, seg, nb_classes=1, dataset='FLARE', lamb=0.5):
    img, _ = get_data(img_path)
    assert seg.shape[0] == nb_classes

    if dataset == 'FLARE':
        spacing = [2.5, 2.0, 2.0]

    elif dataset == 'BraTS':
        spacing = [1.0, 1.0, 1.0]
        seg[seg == 4] = 3

    else:
        spacing = [1.0, 1.0, 1.0]
        print('New the dataset, define spacing or get spacing from data!!!')

    geodesic_maps = np.zeros((nb_classes, seg.shape[1], seg.shape[2], seg.shape[3]), np.float32)
    seeds = get_fg_skeleton_seeds(seg, nb_classes)

    for c in range(0, nb_classes):
        if seeds[c].sum() == 0:
            print("seeds[{}] is zero".format(c))
            continue

        gd = geodesic_distance_3d(img, seeds[c], spacing, lamb, mode="fast")

        geodesic_maps[c] = invert_geodesic_maps(gd, "exp_gamma", 1/gd.mean())

        if geodesic_maps[c].max() > 1.0 or geodesic_maps[c].min() < 0:
            print("geodesic_maps[c] is not [0,1]", c, geodesic_maps[c].max(), geodesic_maps[c].min())

        if np.isnan(geodesic_maps[c]).sum() != 0 :
            print("geodesic_maps[c] is Nan", c)

    geodesic_maps = background_geodesic_map(geodesic_maps, nb_classes)
    return geodesic_maps


def get_SG_distance(img_path, seg_path, nb_classes=1, dataset='FLARE', lamb=0.5, mode='fast'):
    img, _ = get_data(img_path)
    seg, _ = get_data(seg_path, is_seg=True)

    if dataset == 'FLARE':
        spacing = [2.5, 2.0, 2.0]

    elif dataset == 'BraTS':
        spacing = [1.0, 1.0, 1.0]
        seg[seg == 4] = 3

    else:
        spacing = [1.0, 1.0, 1.0]
        print('New the dataset, define spacing or get spacing from data!!!')

    sg_maps = np.zeros((nb_classes, seg.shape[0], seg.shape[1], seg.shape[2]), np.float32)
    sg_maps = np.zeros((nb_classes, seg.shape[0], seg.shape[1], seg.shape[2]), np.float32)
    seeds = get_boundary_seeds(seg, nb_classes + 1)

    for c in range(1, nb_classes + 1):
        if seeds[c].sum() == 0:
            print("seeds[{}] is zero".format(c))
            continue
        channel_id = c - 1
        fg_mask = seg == c
        bg_mask = seg != c
        non_brain_mask = img == 0
        bd_mask = seeds[c]

        gd = geodesic_distance_3d(img, bd_mask, spacing, lamb=lamb, mode=mode)
        sg_maps[channel_id] = gd.copy()
        gd_dt = geodesic_distance_3d(img, bd_mask, spacing, lamb=0, mode=mode)
        roi_dt = gd_dt <= gd_dt[fg_mask].max()
        # roi_gd = gd <= gd[fg_mask].max()
        roi = roi_dt # & roi_gd
        # max_fg = gd[roi].max()
        max_bg = gd[roi & bg_mask].max()
        sg_maps[channel_id][~roi] = max_bg
        # uncertainty_map = uncertainty_map / max_fg
        sg_maps[channel_id][bg_mask] = -sg_maps[channel_id][bg_mask]
        sg_maps[channel_id][non_brain_mask] = -max_bg

        if np.isnan(sg_maps[channel_id]).sum() != 0 :
            print("geodesic_maps[c] is Nan", c)

    # sg_maps = sg_maps / sg_maps.max()

    print(f"geodesic_maps: max = {sg_maps.max()}, min = {sg_maps.min()}")

    return sg_maps


def merge_seg(seg_path):
    seg, _ = get_data(seg_path, is_seg=True)
    combined_seg = [(seg == 1) | (seg == 4), (seg == 1) | (seg == 4) | (seg == 2), seg == 4]
    combined_seg = np.stack(combined_seg, axis=0)
    return combined_seg


def get_SGC_distance(img_path, seg, nb_classes=1, dataset='FLARE', lamb=0.5, mode='fast'):
    img, _ = get_data(img_path)
    if seg.ndim == 3:
        seg = np.expand_dims(seg, axis=0)
    assert seg.shape[0] == nb_classes

    if dataset == 'FLARE':
        spacing = [2.5, 2.0, 2.0]
    elif dataset == 'BraTS':
        spacing = [1.0, 1.0, 1.0]
    elif dataset == 'LGG':
        spacing = [1.0, 1.0, 1.0]
    else:
        spacing = [1.0, 1.0, 1.0]
        print('New the dataset, define spacing or get spacing from data!!!')

    sg_maps = np.zeros((nb_classes, seg.shape[1], seg.shape[2], seg.shape[3]), np.float32)
    sg_maps = np.zeros((nb_classes, seg.shape[1], seg.shape[2], seg.shape[3]), np.float32)
    seeds = get_boundary_seeds(seg, nb_classes)
    print(seg.shape, img.shape, seeds.shape)

    for c in range(0, nb_classes):
        if seeds[c].sum() == 0:
            print("seeds[{}] is zero".format(c))
            continue
        channel_id = c
        fg_mask = seg[c, ...].astype(bool)
        bg_mask = (seg[c, ...] == 0).astype(bool)
        non_brain_mask = img == 0
        bd_mask = seeds[c]

        gd = geodesic_distance_3d(img, bd_mask, spacing, lamb=lamb, mode=mode)
        sg_maps[channel_id] = gd.copy()
        gd_dt = geodesic_distance_3d(img, bd_mask, spacing, lamb=0, mode=mode)
        roi_dt = gd_dt <= gd_dt[fg_mask].max()
        # roi_gd = gd <= gd[fg_mask].max()
        roi = roi_dt # & roi_gd
        # max_fg = gd[roi].max()
        max_bg = gd[roi & bg_mask].max()
        sg_maps[channel_id][~roi] = max_bg
        # uncertainty_map = uncertainty_map / max_fg
        sg_maps[channel_id][bg_mask] = -sg_maps[channel_id][bg_mask]
        sg_maps[channel_id][non_brain_mask] = -max_bg

        if np.isnan(sg_maps[channel_id]).sum() != 0 :
            print("geodesic_maps[c] is Nan", c)

    # sg_maps = sg_maps / sg_maps.max()
    print(f"geodesic_maps: max = {sg_maps.max()}, min = {sg_maps.min()}")
    return sg_maps


def write_mask(img, output_path, method, img_name, suffix="gd"):
    save_dir = os.path.join(output_path, method, img_name)
    print(f'Write mask {suffix} using {method} to {save_dir}...')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, img_name + f"_{suffix}.nii.gz")
    write_data4d(img, save_path)
    print(" ")


def generate_geodesic_maps(method, dataset='FLARE', nb_classes=1, data_path='', output_path=''):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    mode = 'fast'
    print(mode, method)

    img_name_list = []
    for img_index, img_name in enumerate(os.listdir(data_path)):
        img_name_list.append(img_name)

        print("{}. Starting.. {}".format(len(img_name_list), img_name))

        if dataset == 'FLARE':
            img_path = os.path.join(data_path, img_name, img_name + "_img.nii.gz")
            seg_path = os.path.join(data_path, img_name, img_name + "_seg.nii.gz")

            if method == 'ug':
                img_gd = get_geodesic_distance(img_path, seg_path, nb_classes, dataset)
            elif method == 'sg':
                img_gd = get_SG_distance(img_path, seg_path, nb_classes, dataset)

        elif dataset == 'BraTS':
            flair_path = os.path.join(data_path, img_name, img_name + "_flair.nii.gz")
            t1ce_path = os.path.join(data_path, img_name, img_name + "_t1ce.nii.gz")
            t1_path = os.path.join(data_path, img_name, img_name + "_t1.nii.gz")
            t2_path = os.path.join(data_path, img_name, img_name + "_t2.nii.gz")
            seg_path = os.path.join(data_path, img_name, img_name + "_seg.nii.gz")

            if method == 'ug':
                seg = merge_seg(seg_path)
                flair_gd = get_geodesic_distance(flair_path, seg, nb_classes, dataset)
                t1ce_gd = get_geodesic_distance(t1ce_path, seg, nb_classes, dataset)
                t1_gd = get_geodesic_distance(t1_path, seg, nb_classes, dataset)
                t2_gd = get_geodesic_distance(t2_path, seg, nb_classes, dataset)

                img_gd = (flair_gd  + t1ce_gd + t1_gd + t2_gd)/4.0

                write_mask(img_gd, output_path, method, img_name)

            elif method == 'sg':
                flair_gd = get_SG_distance(flair_path, seg_path, nb_classes, dataset, mode=mode)
                write_mask(flair_gd, output_path, f'{mode}_{method}', img_name, f"{mode}_{method}_flair")

                t1ce_gd = get_SG_distance(t1ce_path, seg_path, nb_classes, dataset, mode=mode)
                write_mask(t1ce_gd, output_path, f'{mode}_{method}', img_name, f"{mode}_{method}_t1ce")

                t1_gd = get_SG_distance(t1_path, seg_path, nb_classes, dataset, mode=mode)
                write_mask(t1_gd, output_path, f'{mode}_{method}', img_name, f"{mode}_{method}_t1")

                t2_gd = get_SG_distance(t2_path, seg_path, nb_classes, dataset, mode=mode)
                write_mask(t2_gd, output_path, f'{mode}_{method}', img_name, f"{mode}_{method}_t2")

                if img_index < 5:
                    print(
                        t1ce_gd.shape, t1ce_gd.min(), t1ce_gd.max(),
                        t1_gd.shape, t2_gd.shape,
                        flair_gd.shape,flair_gd.min(), flair_gd.max(),
                    )

            elif method == 'sgc':
                seg = merge_seg(seg_path)

                flair_gd = get_SGC_distance(flair_path, seg, nb_classes, dataset, mode=mode)
                write_mask(flair_gd, output_path, f'{mode}_{method}', img_name, f"{mode}_sg_flair")

                t1ce_gd = get_SGC_distance(t1ce_path, seg, nb_classes, dataset, mode=mode)
                write_mask(t1ce_gd, output_path, f'{mode}_{method}', img_name, f"{mode}_sg_t1ce")

                t1_gd = get_SGC_distance(t1_path, seg, nb_classes, dataset, mode=mode)
                write_mask(t1_gd, output_path, f'{mode}_{method}', img_name, f"{mode}_sg_t1")

                t2_gd = get_SGC_distance(t2_path, seg, nb_classes, dataset, mode=mode)
                write_mask(t2_gd, output_path, f'{mode}_{method}', img_name, f"{mode}_sg_t2")

                if img_index < 5:
                    print(
                        t1ce_gd.shape, t1ce_gd.min(), t1ce_gd.max(),
                        t1_gd.shape, t2_gd.shape,
                        flair_gd.shape,flair_gd.min(), flair_gd.max(),
                    )

            elif method == 'ls':
                epsilon = 0.5
                seg = merge_seg(seg_path)
                gd = np.zeros_like(seg).astype(float)
                for i in range(0, seg.shape[0]):
                    gt = torch.tensor(seg[i, ...], dtype=int).view(1, seg.shape[1], seg.shape[2], seg.shape[3])
                    soft_label = functional.one_hot(gt, 2).permute(0, 4, 1, 2, 3).to(torch.float)
                    soft_label = torch.flip(soft_label, dims=[1])
                    soft_label *= (1 - epsilon)
                    soft_label += (epsilon / num_classes)
                    soft_label = soft_label[0, 0, :, :].cpu().numpy()
                    gd[i, ...] = soft_label

                write_mask(gd, output_path, f'{method}', img_name, f"{mode}_ls")

        elif dataset == "LGG":
            method = 'sgc'
            flair_0_path = os.path.join(data_path, img_name, img_name + "_flair_0.nii.gz")
            flair_1_path = os.path.join(data_path, img_name, img_name + "_flair_1.nii.gz")
            flair_2_path = os.path.join(data_path, img_name, img_name + "_flair_2.nii.gz")
            seg_path = os.path.join(data_path, img_name, img_name + "_seg.nii.gz")
            seg, _ = get_data(seg_path, is_seg=True)

            flair_0_gd = get_SGC_distance(flair_0_path, seg, nb_classes, mode=mode)
            write_mask(flair_0_gd, output_path, f'{mode}_{method}', img_name, f"{mode}_sgc_flair_0")
            flair_1_gd = get_SGC_distance(flair_1_path, seg, nb_classes, mode=mode)
            write_mask(flair_1_gd, output_path, f'{mode}_{method}', img_name, f"{mode}_sgc_flair_1")
            flair_2_gd = get_SGC_distance(flair_2_path, seg, nb_classes, mode=mode)
            write_mask(flair_2_gd, output_path, f'{mode}_{method}', img_name, f"{mode}_sgc_flair_2")

            if img_index < 5:
                print(
                    flair_0_gd.shape, flair_0_gd.min(), flair_0_gd.max(),
                    flair_1_gd.shape, flair_2_gd.shape,
                    seg.shape,
                )

        else:
            print('Add the dataset similar to FLARE')

    print("Total number of images processed: {}".format(len(img_name_list)))


if __name__ == "__main__":
    '''
    dataset: dataset name [FLARE, BraTS]
    num_classes: number of classes in the dataset
    input_dir: input data directory containing folder-wise data of volume and its mask
    output_dir: output data directory of classwise Geodesic maps
    num_seeds: number of random seed points to generate Geodesic maps (optional)
    '''
    parser = argparse.ArgumentParser(description='Geodesic Maps')
    parser.add_argument('--dataset', default='BraTS', help="options:[FLARE, BraTS]")
    # parser.add_argument('--num_classes', default=5, type=int, help="number of classes")
    parser.add_argument('--input_dir', help='input data directory')
    parser.add_argument('--output_dir', help='output data directory')
    args = parser.parse_args()

    if args.dataset == 'BraTS':
        num_classes = 3 # excluding BG
    elif args.dataset == 'FLARE':
        num_classes = 4
    elif args.dataset == 'LGG':
        num_classes = 1

    generate_geodesic_maps('sgc', args.dataset, num_classes, args.input_dir, args.output_dir)
