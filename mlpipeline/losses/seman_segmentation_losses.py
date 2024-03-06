import torch
import torch.nn as nn
import torch.nn.functional as functional
import monai.losses
from einops import rearrange
from torch.nn.modules.loss import _Loss

from segmentation_models_pytorch.losses import JaccardLoss, DiceLoss, TverskyLoss, FocalLoss, LovaszLoss
from segmentation_models_pytorch.losses import SoftBCEWithLogitsLoss
from monai.losses import DiceCELoss, DiceFocalLoss


class BinaryFocalLoss(nn.Module):
    def __init__(self):
        super(BinaryFocalLoss, self).__init__()

    @staticmethod
    def binary_focal(pred, gt, gamma=2, *args):
        return -gt * torch.log(pred) * torch.pow(1 - pred, gamma)

    def forward(self, pred, gt, gamma=2, eps=1e-6, *args):
        pred = torch.clamp(pred, eps, 1 - eps)
        loss1 = self.binary_focal(pred, gt, gamma=gamma)
        loss2 = self.binary_focal(1 - pred, 1 - gt, gamma=gamma)
        loss = loss1 + loss2
        return loss.mean()


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pred, gt):
        loss = self.criterion(pred, gt.squeeze(dim=1).long())
        return loss


class DiceBCE(nn.Module):
    def __init__(self, mode, from_logits, smooth, pos_weight):
        super(DiceBCE, self).__init__()
        self.dice = DiceLoss(mode=mode, from_logits=from_logits, smooth=smooth)
        self.bce = SoftBCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]), smooth_factor=smooth)

    def forward(self, pred, gt):
        loss = self.dice(pred, gt)
        loss = loss + self.bce(pred, gt.float())
        return loss


class SignedFocalLoss(nn.Module):
    def __init__(
        self, loss_type=1, base_loss="l1", alpha=1.0, beta=1.0,
        pos_weight=1.0,
        use_opposite_sign=True,
        directional_weight=1.0,
        am_constant=0.0,
    ):
        super(SignedFocalLoss, self).__init__()
        self.base_loss = base_loss

        if base_loss == "smoothl1":
            self.criterion = nn.SmoothL1Loss(reduction="none")
        elif base_loss == "l2":
            self.criterion = nn.MSELoss(reduction="none")
        elif base_loss == "bce":
            self.criterion = nn.BCELoss(reduction="none")
        else:
            self.criterion = nn.L1Loss(reduction="none")

        self.loss_type = loss_type
        self.alpha = alpha
        self.beta = beta
        self.pos_weight = pos_weight
        self.use_opposite_sign = use_opposite_sign
        self.am_constant = am_constant
        self.directional_weight = directional_weight

    def compute_loss(self, pred, gt):
        if self.base_loss in ["bce"]:
            pred = (pred + 1.0) / 2.0
            gt = (gt + 1.0) / 2.0

        loss = self.criterion(pred, gt)
        pos_mask = (gt > 0)

        am_filtering_weight = torch.ones_like(gt)
        if self.am_constant > 0.1:
            neg_loss = loss[~pos_mask]
            mu = neg_loss.mean()
            beta = 2.0 + (mu - 2.0) / self.am_constant
            am_filtering_weight[(~pos_mask) & (loss < beta)] = 0.0

        unsigned_weight = torch.pow(torch.abs(pred - gt), self.beta) * self.alpha * am_filtering_weight
        same_sign_mask = (torch.sign(pred) * torch.sign(gt) > 0)
        signed_weight = torch.clamp(
            unsigned_weight, min=1.0, max=None) if self.use_opposite_sign else torch.ones_like(gt)
        weight = torch.where(
            same_sign_mask,
            unsigned_weight,
            signed_weight,
        )
        if self.pos_weight > 1.0:
            weight = weight * torch.where(
                pos_mask,
                torch.full_like(gt, fill_value=self.pos_weight),
                torch.ones_like(gt),
            )
        if self.directional_weight < 1.0:
            high_abs_pred_mask = (torch.abs(pred) > torch.abs(gt))
            good_direction_mask = same_sign_mask & high_abs_pred_mask
            weight = weight * torch.where(
                good_direction_mask,
                torch.full_like(gt, fill_value=self.directional_weight),
                torch.ones_like(gt),
            )

        if self.loss_type == 3:
            abs_value = torch.maximum(torch.abs(pred), torch.abs(gt))
            weight_by_abs = 1.0 / abs_value
            weight = weight * weight_by_abs
        if self.loss_type not in [2]:
            weight = weight.detach()

        loss = loss * weight

        if self.am_constant > 0.1:
            num_pos = pos_mask.sum()
            num_neg = am_filtering_weight.sum()
            loss = torch.sum(loss) / (num_pos + num_neg)

        else:
            # loss = torch.mean(loss, dim=(0, 2, 3))
            # loss = torch.sum(loss, dim=-1)
            loss = torch.mean(loss, dim=(0, 2, 3, 4))

        return loss

    def forward(self, pred, gt):
        loss = 0.0

        for i in range(0, gt.shape[1]):
            pred_class = pred[:, i:(i + 1), :, :, :]
            gt_class = gt[:, i:(i + 1), :, :, :]
            loss = loss + self.compute_loss(pred_class, gt_class)

        loss = loss / gt.shape[1]
        return loss


class SimpleFocalLoss(nn.Module):
    def __init__(
        self, base_loss="l1", alpha=1.0, beta=1.0,
        pos_weight=1.0,
    ):
        super(SimpleFocalLoss, self).__init__()
        self.base_loss = base_loss

        if base_loss == "smoothl1":
            self.criterion = nn.SmoothL1Loss(reduction="none")
        elif base_loss == "l2":
            self.criterion = nn.MSELoss(reduction="none")
        elif base_loss == "bce":
            self.criterion = nn.BCELoss(reduction="none")
        else:
            self.criterion = nn.L1Loss(reduction="none")

        self.alpha = alpha
        self.beta = beta
        self.pos_weight = pos_weight

    def compute_loss(self, pred, gt):
        if self.base_loss in ["bce"]:
            pred = (pred + 1.0) / 2.0
            gt = (gt + 1.0) / 2.0

        loss = self.criterion(pred, gt)
        weight = torch.pow(torch.abs(pred - gt), self.beta) * self.alpha

        if self.pos_weight > 1.0:
            pos_mask = (gt > 0)
            weight = weight * torch.where(
                pos_mask,
                torch.full_like(gt, fill_value=self.pos_weight),
                torch.ones_like(gt),
            )

        weight = weight.detach()
        loss = loss * weight

        if self.base_loss == "l1d":
            loss = torch.mean(loss, dim=(0, 2, 3))
            loss = torch.sum(loss, dim=-1)
        else:
            loss = torch.mean(loss, dim=(0, 2, 3, 4))

        return loss

    def forward(self, pred, gt):
        loss = 0.0

        for i in range(0, gt.shape[1]):
            pred_class = pred[:, i:(i + 1), :, :, :]
            gt_class = gt[:, i:(i + 1), :, :, :]
            loss = loss + self.compute_loss(pred_class, gt_class)

        loss = loss / gt.shape[1]
        return loss


class DistanceLoss(nn.Module):
    def __init__(self, name, **kwargs):
        super(DistanceLoss, self).__init__()
        self.distance_name = name.replace("distance-", "").lower()
        loss_type = kwargs.get("loss_type", 1)
        base_loss = kwargs.get("base_loss", "l1")
        alpha = kwargs.get("alpha", 1.0)
        beta = kwargs.get("beta", 1.0)
        pos_weight = kwargs.get("pos_weight", 1.0)
        use_opposite_sign = kwargs.get("use_opposite_sign", True)
        am_constant = kwargs.get("am_constant", 0)
        directional_weight = kwargs.get("directional_weight", 1.0)

        self.criterion = {
            "l2": nn.MSELoss(),
            "l1": nn.L1Loss(),
            "smoothl1": nn.SmoothL1Loss(),
            "signedfocal": SignedFocalLoss(
                loss_type=loss_type,
                base_loss=base_loss,
                alpha=alpha,
                beta=beta,
                pos_weight=pos_weight,
                use_opposite_sign=use_opposite_sign,
                am_constant=am_constant,
                directional_weight=directional_weight,
            ),
            "focal": SimpleFocalLoss(
                base_loss=base_loss,
                alpha=alpha,
                beta=beta,
                pos_weight=pos_weight,
            ),
        }[self.distance_name]

    def forward(self, pred, gt):
        if self.distance_name == "l2":
            loss = self.criterion(torch.tanh(pred).float(), gt.float())
        else:
            loss = self.criterion(torch.tanh(pred), gt)
        return loss


class ProductLoss(nn.Module):
    def __init__(self):
        super(ProductLoss, self).__init__()
        self.smooth = 1e-6
        self.criterion = nn.L1Loss()

    def forward(self, pred, gt):
        # gt = gt[:, 1:, :, :]
        pred = torch.tanh(pred)
        assert gt.shape[1] == pred.shape[1] == 1

        pred = pred.view(pred.shape[0], -1)
        gt = gt.view(gt.shape[0], -1)

        pt2 = torch.sum(pred * pred, dim=-1)
        yt2 = torch.sum(gt * gt, dim=-1)
        ytpt = torch.sum(gt * pred, dim=-1)

        loss = (ytpt + self.smooth) / (ytpt + pt2 + yt2 + self.smooth)
        loss = -torch.mean(loss)
        loss = loss + self.criterion(pred, gt)
        return loss


def create_segmentation_loss(name, **kwargs):
    mode = kwargs["mode"]
    from_logits = kwargs["from_logits"]
    smooth = kwargs["smooth"]
    loss_type = kwargs.get("loss_type", 1)
    pos_weight = kwargs.get("pos_weight", 1.0)

    if name == "dice":
        return DiceLoss(mode=mode, from_logits=from_logits, smooth=smooth)
    elif name == "jaccard":
        return JaccardLoss(mode=mode, from_logits=from_logits, smooth=smooth)
    elif name == "tversky":
        return TverskyLoss(mode=mode, from_logits=from_logits, smooth=smooth)
    elif name == "focal":
        return FocalLoss(mode=mode, **kwargs)
    elif name == "binary-focal":
        return BinaryFocalLoss()
    elif name == "lovasz":
        return LovaszLoss(**kwargs)
    elif name == "bce":
        return SoftBCEWithLogitsLoss()
    elif name == "ce":
        return CrossEntropyLoss()
    elif name == "dicebce":
        return DiceBCE(mode=mode, from_logits=from_logits, smooth=smooth, pos_weight=pos_weight)
    elif name == "dicefocal":
        return DiceFocalLoss(include_background=False)
    elif "distance-" in name:
        return DistanceLoss(name, **kwargs)
    elif name == "product":
        return ProductLoss()
    else:
        raise ValueError(f'Not support loss {name}.')
