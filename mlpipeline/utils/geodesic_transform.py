import os
import click
from pathlib import Path

import nibabel as nib
import numpy as np
import cv2
import torch
from natsort import natsorted

import mlpipeline.utils.common as utils
from mlpipeline.utils.geodesic_map import write_data4d, merge_seg, get_data


def transform_geo_dist(dist, gt, label_name):
    margin = 0.5

    def _transform_geo_dist_layer(dist_layer, gt_layer):
        fg = (gt_layer == 1)
        bg = (gt_layer == 0)
        dist_layer[fg] = torch.clamp_min(dist_layer[fg], min=0.0)
        dist_layer[bg] = torch.clamp_max(dist_layer[bg], max=0.0)

        if torch.sum(fg) == 0:
            return torch.full_like(dist_layer, -1)

        fg_max = dist_layer[fg].max()
        dist_layer[bg] = torch.clamp_min(dist_layer[bg], min=-fg_max)

        dist_layer[fg] = dist_layer[fg] / (fg_max + 1e-8)
        dist_layer[bg] = dist_layer[bg] / (fg_max + 1e-8)

        if label_name in ["fast_sgc_margin"]:
            dist_layer[fg] = (dist_layer[fg] / 1.0) * (1.0 - margin) + margin
            dist_layer[bg] = (dist_layer[bg] / 1.0) * (1.0 - margin) - margin

        dist_layer = torch.clamp(dist_layer, min=-1.0, max=1.0)

        if label_name == "fast_sgc_clamp":
            dist_layer[(gt_layer > 0) & (dist_layer <= 0)] = dist_layer[dist_layer > 0.0001].min()
            dist_layer[(gt_layer == 0) & (dist_layer >= 0)] = dist_layer[dist_layer < -0.0001].max()

        return dist_layer

    assert not torch.any(torch.isnan(dist))
    for index in range(0, dist.shape[-1]):
        dist[..., index] = _transform_geo_dist_layer(dist[..., index], gt[..., index])
    assert not torch.any(torch.isnan(dist))
    return dist


def transform_gd_dist(dist, gt):
    def _transform_gd_dist_layer(dist_layer, gt_layer):
        fg = (gt_layer == 1)
        bg = (gt_layer == 0)

        dist_layer[fg] = torch.clamp_min(dist_layer[fg], min=0.52)
        dist_layer[bg] = torch.clamp_max(dist_layer[bg], max=0.48)
        return dist_layer

    assert not torch.any(torch.isnan(dist))
    for index in range(0, dist.shape[-1]):
        dist[..., index] = _transform_gd_dist_layer(dist[..., index], gt[..., index])
    assert not torch.any(torch.isnan(dist))
    return dist


def show_data(dataset, root_dir, gt_dir, temp_dir):
    sub_dirs = natsorted(list(Path(root_dir).glob("*")))

    for si, sub_dir in enumerate(sub_dirs[:5]):
        paths = list(Path(sub_dir).glob("*.nii.gz"))
        sub_dir_name = sub_dir.name
        raw_gt_path = Path(gt_dir) / sub_dir_name / f"{sub_dir_name}_seg.nii.gz"

        if dataset == "BRATS":
            raw_gt = np.transpose(merge_seg(raw_gt_path), [3, 2, 1, 0])
        else:
            raw_gt = get_data(raw_gt_path, is_seg=True)[0]
            raw_gt = np.transpose(np.expand_dims(raw_gt, axis=0), [3, 2, 1, 0])
        # count_data = []

        for pi, path in enumerate(paths):
            name = path.stem.replace(".nii", "")
            raw_geo_gt = nib.load(path).get_fdata()

            for class_index in range(0, raw_geo_gt.shape[-1]):
                hard_class_data = raw_gt[..., class_index]
                soft_class_data = raw_geo_gt[..., class_index]
                sum_by_depth = np.sum(np.sum(hard_class_data, axis=0), axis=0)
                crowded_depth = np.argmax(sum_by_depth)
                layer = raw_geo_gt[:, :, crowded_depth, class_index]

                if si == pi == 0:
                    print(name, soft_class_data.min(), soft_class_data.max())
                    print(soft_class_data[hard_class_data == 0].max(), soft_class_data[hard_class_data == 1].min())
                    depth_index = 84 if dataset == "BRATS" else 15
                    if class_index == 0:
                        print(layer[depth_index, 92:112])
                    elif class_index == 1:
                        print(layer[depth_index, 79:99])
                    elif class_index == 2:
                        print(layer[depth_index, 130:142])

                layer_image = np.clip(layer * 128 + 128, a_min=0, a_max=255).astype(np.uint8)
                heatmap = utils.convert_grayscale_to_heatmap(layer_image)
                cv2.imwrite(os.path.join(temp_dir, f"{name}_{class_index}_soft.png"), heatmap)
        # print(count_data)
    return


def run_transformation(label_name, dataset, root_dir, gt_dir, output_dir):
    sub_dirs = natsorted(list(Path(root_dir).glob("*")))
    root_name = Path(root_dir).stem
    print(root_dir, len(sub_dirs))

    for si, sub_dir in enumerate(sub_dirs):
        paths = list(Path(sub_dir).glob("*.nii.gz"))
        sub_dir_name = sub_dir.name
        print(sub_dir_name)
        os.makedirs(Path(output_dir) / sub_dir_name, exist_ok=True)

        hard_gt_path = Path(gt_dir) / sub_dir_name / f"{sub_dir_name}_seg.nii.gz"
        if dataset == "BRATS":
            hard_gt = np.transpose(merge_seg(hard_gt_path), [3, 2, 1, 0])
        else:
            hard_gt = get_data(hard_gt_path, is_seg=True)[0]
            hard_gt = np.transpose(np.expand_dims(hard_gt, axis=0), [3, 2, 1, 0])
        hard_gt = torch.tensor(hard_gt, device="cuda:0")

        for path in paths:
            name = path.stem.replace(".nii", "")
            raw_geo_gt = nib.load(path).get_fdata()

            geo_gt = torch.tensor(raw_geo_gt, device="cuda:0")
            if label_name == "gd_normed":
                geo_gt = transform_gd_dist(geo_gt, hard_gt)
            else:
                geo_gt = transform_geo_dist(geo_gt, hard_gt, label_name)
            if si < 5:
                print(raw_geo_gt.shape, raw_geo_gt.max(), geo_gt.shape, geo_gt.min(), geo_gt.max())

            geo_gt = geo_gt.cpu().numpy()
            permuted_geo_gt = np.transpose(geo_gt, [3, 2, 1, 0])
            output_name = name.replace(f"_{root_name}_", f"_{label_name}_")
            output_path = Path(output_dir) / sub_dir_name / f"{output_name}.nii.gz"
            write_data4d(permuted_geo_gt, output_path)
    return


@click.command()
@click.option("--label_name", default="fast_sgc_margin")
@click.option("--dataset")
@click.option("--root_dir")
@click.option("--gt_dir")
@click.option("--output_dir")
def main(label_name: str, dataset: str, root_dir: str, gt_dir: str, output_dir: str):
    temp_dir = "/mnt/d/workspace/lab_work/output/temp"
    os.makedirs(temp_dir, exist_ok=True)

    run_transformation(label_name, dataset, root_dir, gt_dir, output_dir)
    show_data(dataset, output_dir, gt_dir, temp_dir)


if __name__ == "__main__":
    main()
