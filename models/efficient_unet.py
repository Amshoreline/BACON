import torch
import torch.nn.functional as F
from torch import nn

from .archs import VanillaModel, one_minus_dice
from .archs_domain import RandomConv, CutOut, MixStyle, RSC, ContraConv
from segmentation_models_pytorch import Unet


class UNetBody(torch.nn.Module):

    def __init__(self, in_channels, classes):
        super(UNetBody, self).__init__()
        self.model = Unet(
            encoder_name='efficientnet-b2',
            encoder_weights=None,
            in_channels=in_channels,
            classes=classes,
            activation=None
        )

    def decode(self, *features):
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder
        head = features[0]
        skips = features[1:]
        #
        last_feat = None
        x = self.model.decoder.center(head)
        for i, decoder_block in enumerate(self.model.decoder.blocks):
            skip = skips[i] if i < len(skips) else None
            if skip is None:
                last_feat = x
            x = decoder_block(x, skip)
        return x, last_feat

    def forward(self, x, ret_last_feat=False):
        self.enc_features = self.model.encoder(x)
        # self.decoder_output = self.model.decoder(*self.enc_features)
        self.decoder_output, last_feat = self.decode(*self.enc_features)
        masks = self.model.segmentation_head(self.decoder_output)
        if ret_last_feat:
            return last_feat, masks
        else:
            return masks


class AugUNetBody(UNetBody):

    def _encoding(self, x, aug_enc_convs=None):
        stages = self.model.encoder.get_stages()
        #
        block_number = 0.0
        drop_connect_rate = self.model.encoder._global_params.drop_connect_rate
        #
        features = []
        for i in range(self.model.encoder._depth + 1):
            if aug_enc_convs:
                x = aug_enc_convs[i](x)
            # Identity and Sequential stages
            if i < 2:
                x = stages[i](x)
            # Block stages need drop_connect rate
            else:
                for module in stages[i]:
                    drop_connect = drop_connect_rate * block_number / len(self.model.encoder._blocks)
                    block_number += 1.0
                    x = module(x, drop_connect)
            features.append(x)
        return features

    def forward(self, x, aug_enc_convs=None):
        self.enc_features = self._encoding(x, aug_enc_convs)
        self.decoder_output = self.model.decoder(*self.enc_features)
        masks = self.model.segmentation_head(self.decoder_output)
        return masks


class EfficientUNet(VanillaModel):

    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        self.unet_body = UNetBody(in_channels=in_channels, classes=num_classes)


class EfficientRandomConv(RandomConv):
    
    def __init__(self, in_channels, num_classes, conv_type, **kwargs):
        super().__init__(num_classes=num_classes, conv_type=conv_type, **kwargs)
        if conv_type == 'enc_gin_conv':
            self.unet_body = AugUNetBody(in_channels=in_channels, classes=num_classes)
        else:
            self.unet_body = UNetBody(in_channels=in_channels, classes=num_classes)


class EfficientCutOut(CutOut):
    
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        self.unet_body = UNetBody(in_channels=in_channels, classes=num_classes)


class EfficientMixStyle(MixStyle):
    
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        self.unet_body = UNetBody(in_channels=in_channels, classes=num_classes)
    
    def forward(self, data_dict):
        imgs = data_dict['images']    # (bs, 1, d, h, w) or (bs, 1, depth_2d, h, w)
        targets = data_dict['labels'] # (bs, d, h, w)    or (bs, depth_2d, h, w)
        masks = data_dict['masks']    # (bs, 1, d, h, w) or (bs, 1, depth_2d, h, w)
        assert not self.threeD
        bs, c, depth_2d, h, w = imgs.shape
        imgs = imgs.permute(0, 2, 1, 3, 4).contiguous().view(bs * depth_2d, c, h, w)
        targets = targets.view(bs * depth_2d, h, w)
        masks = masks.view(bs * depth_2d, 1, h, w)
        # unwrap UNetBody to apply MixStyle
        enc_features = self.unet_body.model.encoder(imgs)
        enc_features = [self._mix_style(feat) for feat in enc_features]
        decoder_output = self.unet_body.model.decoder(*enc_features)
        preds = self.unet_body.model.segmentation_head(decoder_output)
        preds = preds * masks
        #
        dice_loss = one_minus_dice(preds, targets, self.dice_weight.to(imgs.device))
        ce_loss = F.cross_entropy(preds, targets.long(), self.ce_weight.to(imgs.device))
        total_loss = dice_loss + ce_loss
        return ['DiceLoss', 'CELoss', 'Total'], [round(dice_loss.item(), 3), round(ce_loss.item(), 3), total_loss.item()], total_loss


class EfficientRSC(RSC):
    
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        self.unet_body = UNetBody(in_channels=in_channels, classes=num_classes)
    
    def forward(self, data_dict):
        imgs = data_dict['images']    # (bs, 1, d, h, w) or (bs, 1, depth_2d, h, w)
        targets = data_dict['labels'] # (bs, d, h, w)    or (bs, depth_2d, h, w)
        masks = data_dict['masks']    # (bs, 1, d, h, w) or (bs, 1, depth_2d, h, w)
        assert not self.threeD
        bs, c, depth_2d, h, w = imgs.shape
        imgs = imgs.permute(0, 2, 1, 3, 4).contiguous().view(bs * depth_2d, c, h, w)
        targets = targets.view(bs * depth_2d, h, w)
        masks = masks.view(bs * depth_2d, 1, h, w)
        # unwrap UNetBody to apply RSC
        enc_features = self.unet_body.model.encoder(imgs)
        decoder_output = self.unet_body.model.decoder(*enc_features)
        decoder_output = self._challenge(decoder_output, targets, self.unet_body.model.segmentation_head)
        preds = self.unet_body.model.segmentation_head(decoder_output)
        preds = preds * masks
        #
        dice_loss = one_minus_dice(preds, targets, self.dice_weight.to(imgs.device))
        ce_loss = F.cross_entropy(preds, targets.long(), self.ce_weight.to(imgs.device))
        total_loss = dice_loss + ce_loss
        return ['DiceLoss', 'CELoss', 'Total'], [round(dice_loss.item(), 3), round(ce_loss.item(), 3), total_loss.item()], total_loss


class EfficientContraConv(ContraConv):
    
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        self.unet_body = UNetBody(in_channels=in_channels, classes=num_classes)
