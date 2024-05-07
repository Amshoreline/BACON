import numpy as np
from skimage import measure
from scipy.ndimage.filters import gaussian_filter
import torch
import torch.nn as nn
import torch.nn.functional as F


def one_minus_dice(outputs, targets, weight):
    '''
    Params:
        outputs: FloatTensor(bs, num_classes, d, h, w) or (bs, num_classes, h, w)
            before softmax
        targets: LongTensor (bs, d, h, w) or (bs, h, w)
        weight:  FloatTensor(num_classes, ) 
    Return:
        loss
    '''
    device = outputs.device
    bs, num_classes, *_ = outputs.size()
    outputs = outputs.softmax(dim=1).view(bs, num_classes, -1)
    targets = (
        targets.view(bs, 1, -1)
        == torch.arange(num_classes).view(1, -1, 1).float().to(device)
    )  # convert to one-hot encoding
    dice_tensor = (
        2 * torch.sum(outputs * targets, dim=(0, 2))
        / (torch.sum(outputs, dim=(0, 2)) + torch.sum(targets, dim=(0, 2)) + 1e-12)
    )
    return torch.sum((1 - dice_tensor) * weight)


def soft_cross_entropy(outputs, targets, weight):
    '''
    Params:
        outputs: FloatTensor (bs, num_classes, d, h, w) or (bs, num_classes, h, w)
            before softmax
        targets: FloatTensor (bs, num_classes, d, h, w) or (bs, num_classes, h, w)
        weight:  FloatTensor(num_classes, )
    Return:
        loss
    '''
    # We ingore the weight
    bs, num_classes, *_ = outputs.size()
    outputs = outputs.view(bs, num_classes, -1)
    log_softmax_outputs = F.log_softmax(outputs, dim=1)  # (bs, num_classes, -1)
    targets = targets.view(bs, num_classes, -1)
    return torch.mean(torch.sum(-(log_softmax_outputs * targets), dim=1))


def smooth_slice_loss(outputs, targets):
    '''
    Params:
        outputs: FloatTensor (bs, num_classes, d, h, w) or (bs, num_classes, h, w)
            before softmax
        targets: FloatTensor (bs, num_classes, d, h, w) or (bs, num_classes, h, w)
    Return:
        loss
    From: http://guanbinli.com/papers/Semi-supervised%20Spatial%20Temporal%20Attention%20Network%20for%20Video%20Polyp%20Segmentation.pdf
    '''
    outputs = outputs.softmax(dim=1)
    loss_left = F.smooth_l1_loss(outputs[:, :, : -1], targets[:, :, 1 :], beta=0.1)
    loss_center = F.smooth_l1_loss(outputs, targets, beta=0.1)
    loss_right = F.smooth_l1_loss(outputs[:, :, 1 :], targets[:, :, : -1], beta=0.1)
    return (loss_left + loss_center + loss_right) / 3.0


def smooth_voxel_loss(outputs, targets):
    '''
    Params:
        outputs: FloatTensor (bs, num_classes, d, h, w) or (bs, num_classes, h, w)
            before softmax
        targets: FloatTensor (bs, num_classes, d, h, w) or (bs, num_classes, h, w)
    Return:
        loss
    '''
    log_softmax_outputs = F.log_softmax(outputs, dim=1)  # (bs, num_classes, d, h, w)
    with torch.no_grad():
        max_targets = F.max_pool3d(targets, kernel_size=3, stride=1)  # (bs, num_classes, d - 2, h - 2, w - 2)
        avg_targets = F.avg_pool3d(targets, kernel_size=3, stride=1)  # (bs, num_classes, d - 2, h - 2, w - 2)
    max_loss = F.smooth_l1_loss(log_softmax_outputs[..., 1 : -1, 1 : -1, 1 : -1], max_targets, beta=0.1)
    avg_loss = F.smooth_l1_loss(log_softmax_outputs[..., 1 : -1, 1 : -1, 1 : -1], avg_targets, beta=0.1)
    loss_center = F.smooth_l1_loss(outputs, targets, beta=0.1)
    return (max_loss + avg_loss + loss_center) / 3.0


def neighbor_voxel_loss(outputs, targets):
    '''
    Params:
        outputs: FloatTensor (bs, num_classes, d, h, w) or (bs, num_classes, h, w)
            before softmax
        targets: FloatTensor (bs, num_classes, d, h, w) or (bs, num_classes, h, w)
    Return:
        loss
    '''
    log_softmax_outputs = F.log_softmax(outputs, dim=1)  # (bs, num_classes, d, h, w)
    with torch.no_grad():
        max_targets = F.max_pool3d(targets, kernel_size=3, stride=1)  # (bs, num_classes, d - 2, h - 2, w - 2)
        avg_targets = F.avg_pool3d(targets, kernel_size=3, stride=1)  # (bs, num_classes, d - 2, h - 2, w - 2)
    max_ce_loss = torch.mean(torch.sum(-(log_softmax_outputs[..., 1 : -1, 1 : -1, 1 : -1] * max_targets), dim=1))
    avg_ce_loss = torch.mean(torch.sum(-(log_softmax_outputs[..., 1 : -1, 1 : -1, 1 : -1] * avg_targets), dim=1))
    return (max_ce_loss + avg_ce_loss) / 2


def get_gaussian(patch_size, sigma_scale=1. / 8):
    '''
    Parmas:
        patch_size: (d, h, w) or (h, w)
    Return:
        gaussian_importance_map.shape = (d, h, w) or (h, w)
    '''
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)
    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0]
    )
    return gaussian_importance_map


def slide_infer_3d(images, gaussian_map, model, num_classes, strides=None):
    '''
    Parameters:
        images: FloatTensor(bs, c, d, h, w)
        gaussian_map: FloatTensor(1, 1, d', h', w')
        model: nn.Module
    '''
    *_, infer_d, infer_h, infer_w = gaussian_map.shape
    *_, origin_d, origin_h, origin_w = images.shape
    device = images.device
    # pad if need, TODO: change to padding in two sides
    pad_d = max(0, infer_d - origin_d)
    pad_h = max(0, infer_h - origin_h)
    pad_w = max(0, infer_w - origin_w)
    images = np.pad(images.cpu().numpy(), ((0, 0), (0, 0), (0, pad_d), (0, pad_h), (0, pad_w)))
    images = torch.tensor(images).to(device)
    #
    bs, c, d, h, w = images.shape
    #
    if strides is None:
        stride_d, stride_h, stride_w = infer_d // 2, infer_h // 2, infer_w // 2
    else:
        stride_d, stride_h, stride_w = strides
    # stride_d, stride_h, stride_w = 4, 16, 16
    # print(f'infer with patch_size {(infer_d, infer_h, infer_w)} and stride {(stride_d, stride_h, stride_w)}')   
    pred_res = torch.zeros(bs, num_classes, d, h, w)
    for d_off in range(0, d - infer_d + stride_d, stride_d):
        if (d_off + infer_d) > d:
            d_off = d - infer_d
        for h_off in range(0, h - infer_h + stride_h, stride_h):
            if (h_off + infer_h) > h:
                h_off = h - infer_h
            for w_off in range(0, w - infer_w + stride_w, stride_w):
                if (w_off + infer_w) > w:
                    w_off = w - infer_w
                pred = model(images[..., d_off : d_off + infer_d, h_off : h_off + infer_h, w_off : w_off + infer_w])
                pred = pred.softmax(dim=1).cpu()
                pred_res[..., d_off : d_off + infer_d, h_off : h_off + infer_h, w_off : w_off + infer_w] += gaussian_map * pred
    return pred_res[..., : origin_d, : origin_h, : origin_w]


def slide_infer_2d(images, gaussian_map, model, num_classes, strides=None):
    '''
    Parameters:
        images: FloatTensor(bs, c, d, h, w)
        gaussian_map: FloatTensor(1, 1, h', w')
        model: nn.Module
    '''
    *_, infer_h, infer_w = gaussian_map.shape
    *_, origin_d, origin_h, origin_w = images.shape
    device = images.device
    # pad if need, TODO: change to padding in two sides
    pad_h = max(0, infer_h - origin_h)
    pad_w = max(0, infer_w - origin_w)
    images = np.pad(images.cpu().numpy(), ((0, 0), (0, 0), (0, 0), (0, pad_h), (0, pad_w)))
    images = torch.tensor(images).to(device)
    #
    bs, c, d, h, w = images.shape
    images = images.permute(0, 2, 1, 3, 4).contiguous().view(bs * d, c, h, w)
    #
    if strides is None:
        stride_h, stride_w = infer_h // 2, infer_w // 2
    else:
        stride_h, stride_w = strides
    # print(f'infer with patch_size {(infer_d, infer_h, infer_w)} and stride {(stride_d, stride_h, stride_w)}')   
    pred_res = torch.zeros(bs * d, num_classes, h, w)
    for h_off in range(0, h - infer_h + stride_h, stride_h):
        if (h_off + infer_h) > h:
            h_off = h - infer_h
        for w_off in range(0, w - infer_w + stride_w, stride_w):
            if (w_off + infer_w) > w:
                w_off = w - infer_w
            pred = model(images[..., h_off : h_off + infer_h, w_off : w_off + infer_w])
            pred = pred.softmax(dim=1).cpu()
            pred_res[..., h_off : h_off + infer_h, w_off : w_off + infer_w] += gaussian_map * pred
    pred_res = pred_res.view(bs, d, num_classes, h, w).permute(0, 2, 1, 3, 4).contiguous()
    return pred_res[..., : origin_h, : origin_w]


def slide_infer(images, gaussian_map, model, num_classes, strides=None):
    if len(gaussian_map.shape) == 5:
        return slide_infer_3d(images, gaussian_map, model, num_classes, strides)
    else:
        return slide_infer_2d(images, gaussian_map, model, num_classes, strides)


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def update_ema_variables(model, ema_model, global_step, alpha=0.95):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        # ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax)


def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax)


def none_loss(preds, targets):
    return torch.zeros(1).to(preds.device)


class VanillaModel(nn.Module):

    def __init__(self, num_classes, infer_size, infer_strides=None, dice_weight=None, ce_weight=None):
        super(VanillaModel, self).__init__()
        self.infer_size = infer_size
        self.threeD = (len(infer_size) == 3)
        self.gaussian_map = torch.tensor(get_gaussian(self.infer_size))[None, None]
        self.num_classes = num_classes
        self.unet_body = None
        #
        if dice_weight is None:
            dice_weight = [0.] + [1.] * (num_classes - 1)
        self.dice_weight = torch.tensor(dice_weight).float() / np.sum(dice_weight)
        if ce_weight is None:
            ce_weight = [1.] * num_classes
        self.ce_weight = torch.tensor(ce_weight).float()
        self.infer_strides = infer_strides

    def forward(self, data_dict):
        imgs = data_dict['images']    # (bs, 1, d, h, w) or (bs, 1, depth_2d, h, w)
        targets = data_dict['labels'] # (bs, d, h, w)    or (bs, depth_2d, h, w)
        masks = data_dict['masks']    # (bs, 1, d, h, w) or (bs, 1, depth_2d, h, w)
        if self.threeD:
            preds = self.unet_body(imgs) * masks
        else:
            bs, c, depth_2d, h, w = imgs.shape
            imgs = imgs.permute(0, 2, 1, 3, 4).contiguous().view(bs * depth_2d, c, h, w)
            targets = targets.view(bs * depth_2d, h, w)
            masks = masks.view(bs * depth_2d, 1, h, w)
            preds = self.unet_body(imgs) * masks
        dice_loss = one_minus_dice(preds, targets, self.dice_weight.to(imgs.device))
        ce_loss = F.cross_entropy(preds, targets.long(), self.ce_weight.to(imgs.device))
        # ce_loss = self.cross_entropy(preds, targets.long())
        total_loss = dice_loss + ce_loss
        return ['DiceLoss', 'CELoss', 'Total'], [round(dice_loss.item(), 3), round(ce_loss.item(), 3), total_loss.item()], total_loss

    def infer(self, data_dict):
        output = slide_infer(data_dict['images'], self.gaussian_map, self.unet_body, self.num_classes, self.infer_strides)
        return torch.max(output, dim=1)

    def update(self, ):
        pass


class MeanTeacher(VanillaModel):
    
    def __init__(self, alpha, rampup_length, consist_loss_name, ssup_range, **kwargs):
        super().__init__(**kwargs)
        self.teacher_unet_body = None
        self.alpha = alpha
        self.rampup_length = rampup_length
        self.consist_criterion = {'mse': softmax_mse_loss, 'kl': softmax_kl_loss}[consist_loss_name]
        assert (type(ssup_range) is list) and (set(ssup_range).issubset({'label', 'unlabel'}))
        self.ssup_range = ssup_range
        self.global_step = 0

    def forward(self, data_dict):
        # Supervised loss
        imgs = data_dict['images_1']  # (bs, 1, d, h, w)
        targets = data_dict['labels'] # (bs, d, h, w)
        masks = data_dict['masks']    # (bs, 1, d, h, w)
        preds = self.unet_body(imgs)
        dice_loss = one_minus_dice(preds * masks, targets, self.dice_weight.to(imgs.device))
        ce_loss = F.cross_entropy(preds * masks, targets.long(), self.ce_weight.to(imgs.device))
        # Consistency loss
        ano_imgs = data_dict['images_2']
        with torch.no_grad():
            ano_preds = self.teacher_unet_body(ano_imgs).detach()
        ssup_masks = torch.zeros_like(masks).to(masks.device)
        if 'label' in self.ssup_range:
            ssup_masks = ssup_masks + masks
        if 'unlabel' in self.ssup_range:
            ssup_masks = ssup_masks + (1 - masks)
        consist_loss = self.consist_criterion(preds * ssup_masks, ano_preds * ssup_masks)
        consist_weight = sigmoid_rampup(self.global_step, self.rampup_length)
        #
        total_loss = dice_loss + ce_loss + consist_weight * consist_loss
        return (
            ['DiceLoss', 'CELoss', 'W.Consist', 'Consist', 'Total'],
            [round(dice_loss.item(), 3), round(ce_loss.item(), 3), round(consist_weight, 3), round(consist_loss.item(), 3), total_loss.item()],
            total_loss
        )

    def update(self, ):
        with torch.no_grad():
            update_ema_variables(self.unet_body, self.teacher_unet_body, self.global_step, self.alpha)
        self.global_step += 1

    def parameters(self, ):
        return self.unet_body.parameters()


class FixMatch(VanillaModel):

    def __init__(self, pseudo_thres, rampup_length, ssup_unlabel_only, **kwargs):
        super().__init__(**kwargs)
        self.pseudo_thres = pseudo_thres
        self.rampup_length = rampup_length
        assert ssup_unlabel_only in [True, False]
        self.ssup_unlabel_only = ssup_unlabel_only
        self.global_step = 0

    def forward(self, data_dict):
        # Supervised loss
        imgs = data_dict['images_1']
        targets = data_dict['labels']
        masks = data_dict['masks']    # (bs, 1, d, h, w)
        preds = self.unet_body(imgs)
        dice_loss = one_minus_dice(preds * masks, targets, self.dice_weight.to(imgs.device))
        ce_loss = F.cross_entropy(preds * masks, targets.long(), self.ce_weight.to(imgs.device))
        # Consistency loss
        ano_imgs = data_dict['images_2']
        with torch.no_grad():
            ano_preds = self.unet_body(ano_imgs).detach()  # (bs, #classes, D, H, W)
            ano_preds = ano_preds.softmax(dim=1)
            pseudo_coefs, pseudo_targets = torch.max(ano_preds, dim=1)  # (bs, D, H, W)
            pseudo_masks = (pseudo_coefs > self.pseudo_thres).float()
            pseudo_masks = pseudo_masks[:, None]  # (bs, 1, D, H, W)
            if self.ssup_unlabel_only:
                # pseudo_masks = pseudo_masks * (targets == 0)[:, None].float()
                pseudo_masks = pseudo_masks * (1 - masks).float()
            pseudo_cnts = round(torch.mean(pseudo_masks).item(), 3)
        consist_dice_loss = one_minus_dice(preds * pseudo_masks, pseudo_targets, self.dice_weight.to(imgs.device))
        consist_ce_loss = F.cross_entropy(preds * pseudo_masks, pseudo_targets.long(), self.ce_weight.to(imgs.device))
        consist_weight = sigmoid_rampup(self.global_step, self.rampup_length)
        #
        total_loss = dice_loss + ce_loss + consist_weight * (consist_dice_loss + consist_ce_loss)
        return (
            ['DiceLoss', 'CELoss', 'W.Cons', 'C.Cons', 'DiceCons', 'CECons', 'Total'],
            [
                round(dice_loss.item(), 3), round(ce_loss.item(), 3),
                round(consist_weight, 3), pseudo_cnts,
                round(consist_dice_loss.item(), 3), round(consist_ce_loss.item(), 3),
                total_loss.item()
            ],
            total_loss
        )

    def update(self, ):
        self.global_step += 1


def get_MLP_w_BN(in_channels, hidden_size, out_channels):
    return nn.Sequential(
        nn.Linear(in_channels, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, out_channels)
    )


class BYOL(VanillaModel):

    def __init__(self, alpha, repr_size, proj_size, **kwargs):
        super().__init__(**kwargs)
        self.teacher_unet_body = None
        self.alpha = alpha
        self.proj = get_MLP_w_BN(repr_size, proj_size, proj_size)
        self.teacher_proj = get_MLP_w_BN(repr_size, proj_size, proj_size) 
        self.pred = get_MLP_w_BN(proj_size, proj_size, proj_size)
        self.global_step = 0

    def forward(self, data_dict):
        # Supervised loss
        imgs = data_dict['images_1']  # (bs, 1, d, h, w)
        targets = data_dict['labels'] # (bs, d, h, w)
        feats = self.unet_body(imgs, ret_deep_feat=True) # (bs, repr_size, d', h', w')
        bs, repr_size, *_ = feats.shape
        reprs = feats.view(bs, repr_size, -1).permute(0, 2, 1).contiguous().view(-1, repr_size)  # (bs * d'h'w', repr_size)
        preds = self.pred(self.proj(reprs)) # (bs * d'h'w', proj_size)
        ano_imgs = data_dict['images_2']
        with torch.no_grad():
            ano_feats = self.teacher_unet_body(ano_imgs, ret_deep_feat=True).detach()
            ano_reprs = ano_feats.view(bs, repr_size, -1).permute(0, 2, 1).contiguous().view(-1, repr_size)
            ano_projs = self.teacher_proj(ano_reprs).detach()
        consist_loss = F.mse_loss(preds, ano_projs)
        return (
            ['Loss'],
            [round(consist_loss.item(), 3)],
            consist_loss
        )

    def update(self, ):
        with torch.no_grad():
            update_ema_variables(self.unet_body, self.teacher_unet_body, self.global_step, self.alpha)
            update_ema_variables(self.proj, self.teacher_proj, self.global_step, self.alpha)
        self.global_step += 1

    def parameters(self, ):
        return list(self.unet_body.parameters()) + list(self.proj.parameters()) + list(self.pred.parameters())

    def infer(self, data_dict):
        imgs = data_dict['images']  # (bs, 1, d, h, w)
        return torch.ones_like(imgs, dtype=torch.long).squeeze(1), torch.zeros_like(imgs, dtype=torch.long).squeeze(1)


def get_ConvPred(in_channels, hidden_size, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, hidden_size, kernel_size=1),
        nn.BatchNorm3d(hidden_size),
        nn.ReLU(),
        nn.Conv3d(hidden_size, out_channels, kernel_size=1)
    )


class ContraTeacher(VanillaModel):

    def __init__(
            self,
            proj_size, alpha, rampup_length,
            ssup_range, pseudo_thres, ssup_loss_name='ce',
            consist_loss_name='mse', hidden_size=-1, **kwargs):
        '''
        Parmas:
            proj_size: #channels in feature map
            alpha: momentum update
            rampup_length: momentum update
            consist_loss_name: 'mse' or 'kl', unavailable temporarily
            ssup_range: list, each element is 'label' or 'unlabel'
            pseudo_thres: threshold of pseudo labeling
        '''
        super().__init__(**kwargs)
        self.teacher_unet_body = None
        self.alpha = alpha
        self.rampup_length = rampup_length
        #
        if hidden_size == -1:
            hidden_size = proj_size
        self.conv_pred = get_ConvPred(proj_size, hidden_size, proj_size)
        self.noise_pred = get_ConvPred(proj_size * 2, hidden_size, 1)
        # self.consist_criterion = {'mse': softmax_mse_loss, 'kl': softmax_kl_loss}[consist_loss_name]
        self.consist_criterion = {'mse': F.mse_loss, 'none': none_loss, 'l1': F.l1_loss}[consist_loss_name]
        #
        assert (type(ssup_range) is list) and (set(ssup_range).issubset({'label', 'unlabel'}))
        self.ssup_range = ssup_range
        self.pseudo_thres = pseudo_thres
        self.ssup_loss_name = ssup_loss_name
        #
        self.global_step = 0

    def forward(self, data_dict):
        # Supervised loss
        imgs = data_dict['images_1']  # (bs, 1, d, h, w)
        targets = data_dict['labels'] # (bs, d, h, w)
        masks = data_dict['masks']    # (bs, 1, d, h, w), mask for valid slices
        feats, preds = self.unet_body(imgs, ret_last_feat=True)
        pred_feats = self.conv_pred(feats)  # (bs, proj_size, d, h, w)
        dice_loss = one_minus_dice(preds * masks, targets, self.dice_weight.to(imgs.device))
        ce_loss = F.cross_entropy(preds * masks, targets.long(), self.ce_weight.to(imgs.device))
        # Consistency loss & SSup loss
        ano_imgs = data_dict['images_2']
        with torch.no_grad():
            ano_feats, ano_preds = self.teacher_unet_body(ano_imgs, ret_last_feat=True)
            ano_feats = ano_feats.detach()
            ano_preds = ano_preds.detach().softmax(dim=1)
            pseudo_coefs, pseudo_targets = torch.max(ano_preds, dim=1)  # (bs, D, H, W)
            ssup_masks = (pseudo_coefs > self.pseudo_thres).float()
            ssup_masks = ssup_masks[:, None]  # (bs, 1, D, H, W)
        consist_loss = self.consist_criterion(pred_feats, ano_feats)
        #
        ssup_upbound_masks = torch.zeros_like(masks).to(masks.device)
        if 'label' in self.ssup_range:
            ssup_upbound_masks = ssup_upbound_masks + masks
        if 'unlabel' in self.ssup_range:
            ssup_upbound_masks = ssup_upbound_masks + (1 - masks)
        ssup_masks = ssup_masks * ssup_upbound_masks
        ssup_ratio = round(torch.mean(ssup_masks).item(), 3)
        if self.ssup_loss_name == 'ce': 
            if self.pseudo_thres == -1:
                ssup_dice_loss = torch.zeros(1).to(imgs.device)
                ssup_ce_loss = soft_cross_entropy(preds * ssup_masks, ano_preds, None)
            else:
                ssup_dice_loss = one_minus_dice(preds * ssup_masks, pseudo_targets, self.dice_weight.to(imgs.device))
                ssup_ce_loss = F.cross_entropy(preds * ssup_masks, pseudo_targets.long(), self.ce_weight.to(imgs.device))
        elif self.ssup_loss_name == 'smooth_slice':
            ssup_dice_loss = smooth_slice_loss(preds * ssup_masks, ano_preds)
            ssup_ce_loss = torch.zeros(1).to(imgs.device)
        elif self.ssup_loss_name == 'neighbor_voxel':
            ssup_dice_loss = torch.zeros(1).to(imgs.device)
            ssup_ce_loss = neighbor_voxel_loss(preds * ssup_masks, ano_preds)
        elif self.ssup_loss_name == 'smooth_voxel':
            ssup_dice_loss = smooth_voxel_loss(preds * ssup_masks, ano_preds)
            ssup_ce_loss = torch.zeros(1).to(imgs.device)
        elif self.ssup_loss_name == 'denoise':
            noise_preds = self.noise_pred(torch.cat([feats, ano_feats], dim=1))  # (bs, proj_size, d, h, w)
            ssup_dice_loss = F.mse_loss(noise_preds, data_dict['noises_1'])
            ssup_ce_loss = torch.zeros(1).to(imgs.device)
        else:
            assert self.ssup_loss_name == 'none'
            ssup_dice_loss = torch.zeros(1).to(imgs.device)
            ssup_ce_loss = torch.zeros(1).to(imgs.device)
        consist_weight = sigmoid_rampup(self.global_step, self.rampup_length)
        #
        total_loss = dice_loss + ce_loss + consist_weight * (consist_loss + ssup_dice_loss + ssup_ce_loss)
        return (
            [
                'DiceLoss', 'CELoss',
                'W.Consist', 'Consist', 'SSupDiceLoss', 'SSupCELoss', 'SSupRatio',
                'Total'
            ],
            [
                round(dice_loss.item(), 3), round(ce_loss.item(), 3),
                round(consist_weight, 3), round(consist_loss.item(), 3), round(ssup_dice_loss.item(), 3), round(ssup_ce_loss.item(), 3), ssup_ratio,
                total_loss.item(),
            ],
            total_loss
        )

    def update(self, ):
        with torch.no_grad():
            update_ema_variables(self.unet_body, self.teacher_unet_body, self.global_step, self.alpha)
        self.global_step += 1

    def parameters(self, ):
        return list(self.unet_body.parameters()) + list(self.conv_pred.parameters())


class EnsembleModel(nn.Module):

    def __init__(self, num_classes, infer_size):
        super().__init__()
        self.unet_bodies = None
        self.num_classes = num_classes
        self.infer_size = infer_size
        self.gaussian_map = torch.tensor(get_gaussian(self.infer_size))[None, None]

    def infer(self, data_dict):
        outputs = []
        for unet_body in self.unet_bodies:
            output = slide_infer(data_dict['images'], self.gaussian_map, unet_body, self.num_classes)
            outputs.append(output[None])
        weights = [
            [1, 1, 1, 1, 2, 5],
            [1, 1, 1, 1, 2, 5],
            [1, 1, 1, 1, 2, 5],
            [1, 1, 1, 1, 5, 5],
            [1, 1, 1, 1, 2, 5],
        ]
        weights = torch.tensor(weights).view(5, 1, 6, 1, 1, 1).to(output.device)
        output = torch.mean(torch.cat(outputs, dim=0) * weights, dim=0)
        return torch.max(output, dim=1)

