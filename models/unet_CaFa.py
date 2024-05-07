import copy
import math
import numpy as np
import torch
from torch import nn

try:
    from .unet import UNetBody
    from .archs import VanillaModel, EnsembleModel
except:
    from unet import UNetBody
    from archs import VanillaModel, EnsembleModel


class TwinMerge(nn.Module):
    
    def __init__(self, num_modals, channels, threeD=True, c_attn=True, s_attn=True):
        '''
        Params:
            num_modals: int
            channels: #channels of each feature map
            threeD: True/False
            attn: True/False, whether to perform attention
        '''
        super().__init__()
        self.c_attn = c_attn
        self.s_attn = s_attn
        if threeD:
            norm_op = nn.InstanceNorm3d
            conv_op = nn.Conv3d
            avg_pool_op = nn.AdaptiveAvgPool3d
            max_pool_op = nn.AdaptiveMaxPool3d
        else:
            norm_op = nn.InstanceNorm2d
            conv_op = nn.Conv2d
            avg_pool_op = nn.AdaptiveAvgPool2d
            max_pool_op = nn.AdaptiveMaxPool2d
        if self.c_attn:
            self.bn = norm_op(num_modals * channels, eps=1e-05, affine=True)
            self.down_conv = conv_op(num_modals * channels, channels, kernel_size=1, padding=0, bias=False)
            self.avg_pool = avg_pool_op(1)
            self.max_pool = max_pool_op(1)
            self.c_conv = nn.Sequential(
                conv_op(channels, channels // 8, kernel_size=1, bias=False),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                conv_op(channels // 8, channels, kernel_size=1, bias=False),
            )
            if self.s_attn:
                self.s_conv = conv_op(2, 1, kernel_size=7, padding=3, bias=False)
        else:
            self.fake_params = nn.Embedding(1, 1)

    def forward(self, x_list):
        '''
        Params:
            x_list: [torch.Tensor(bs, c, d, h, w), ...]
        Return:
            x: torch.Tensor(bs, c, d, h, w)
        '''
        if not self.c_attn:
            x = x_list[0]
            for ano_x in x_list[1 :]:
                x = x + ano_x
            return x
        else:
            x_cat = torch.cat(x_list, dim=1)  # (bs, num_modals * c, d, h, w)
            x_cat = self.bn(x_cat)
            x = self.down_conv(x_cat)  # (bs, c, d, h, w)
            # channel attention
            avg_vec = self.c_conv(self.avg_pool(x))
            max_vec = self.c_conv(self.max_pool(x))
            c_alpha = (avg_vec + max_vec).sigmoid()  # (bs, c, 1, 1, 1)
            x = x * c_alpha
            if self.s_attn:
                # spatial attention
                avg_map = torch.mean(x, dim=1, keepdim=True)  # (bs, 1, d, h, w)
                max_map, _ = torch.max(x, dim=1, keepdim=True)  # (bs, 1, d, h, w)
                s_alpha = self.s_conv(torch.cat([avg_map, max_map], dim=1)).sigmoid()  # (bs, 1, d, h, w)
                x = x * s_alpha
            for ano_x in x_list:
                x = x + ano_x
            return x


class Embeddings(nn.Module):
    
    def __init__(self, feature_size, patch_size, in_channels, hidden_size, dropout_rate=0.1):
        '''
        Params:
            feature_size: (D, H, W) or (H, W)
            patch_size: (d, h, w), or (h, w)
        '''
        super(Embeddings, self).__init__()
        print('Feature shape', feature_size)
        print('Patch shape', patch_size)
        n_patches = np.prod(feature_size) // np.prod(patch_size)
        if len(feature_size) == 3:
            # threeD
            self.patch_embeddings = nn.Conv3d(
                in_channels=in_channels, out_channels=hidden_size,
                kernel_size=patch_size, stride=patch_size
            )
        else:
            # twoD
            self.patch_embeddings = nn.Conv2d(
                in_channels=in_channels, out_channels=hidden_size,
                kernel_size=patch_size, stride=patch_size
            )
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, hidden_size), requires_grad=True)
        nn.init.trunc_normal_(self.position_embeddings, std=0.2)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, features):
        # features.shape = (bs, c, D, H, W)
        x = self.patch_embeddings(features)
        x = x.view(x.shape[0], x.shape[1], -1) # (bs, hidden_size, n_patches)
        x = x.transpose(-1, -2)  # (bs, n_patches, hidden_size)
        assert x.shape[1] == self.position_embeddings.shape[1]
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class MLP(nn.Module):
    
    def __init__(self, channels, hidden_size, dropout_rate=0.1, out_channels=None):
        super(MLP, self).__init__()
        if out_channels is None:
            out_channels = channels
        self.fc1 = nn.Linear(channels, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_channels)
        self.act_fn = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        #
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MMAttention(nn.Module):
    
    def __init__(self, hidden_size, num_heads):
        super(MMAttention, self).__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        #
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        #
        self.out = nn.Linear(hidden_size, hidden_size)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states_list):
        num_modals = len(hidden_states_list)
        # *_layer.shape = (bs, num_heads, num_patches, feat_dim)
        query_layer_list = [
            self.transpose_for_scores(self.query(hidden_states))
            for hidden_states in hidden_states_list
        ]
        key_layer_list = [
            self.transpose_for_scores(self.key(hidden_states))
            for hidden_states in hidden_states_list
        ]
        value_layer_list = [
            self.transpose_for_scores(self.value(hidden_states))
            for hidden_states in hidden_states_list
        ]
        attention_output_list = []
        for modal_ind in range(num_modals):
            query_layer = query_layer_list[modal_ind]
            key_layer = torch.cat(
                [key_layer_ for ind, key_layer_ in enumerate(key_layer_list) if ind != modal_ind],
                dim=2
            )  # (bs, num_heads, num_patches * (num_modals - 1), feat_dim)
            value_layer = torch.cat(
                [value_layer_ for ind, value_layer_ in enumerate(value_layer_list) if ind != modal_ind],
                dim=2
            )  # (bs, num_heads, num_patches * (num_modals - 1), feat_dim)
            # attention_scores.shape = (bs, num_heads, num_patches, num_patches * (num_modals - 1))
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_probs = self.softmax(attention_scores / math.sqrt(self.attention_head_size))
            # context_layer.shape = (bs, num_heads, num_patches, feat_dim)
            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[: -2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)
            attention_output_list.append(self.out(context_layer))
        return attention_output_list


class MMBlock(nn.Module):
    
    def __init__(self, hidden_size, num_heads):
        super(MMBlock, self).__init__()
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = MLP(hidden_size, hidden_size)
        self.attn = MMAttention(hidden_size, num_heads)

    def forward(self, x_list):
        # attention
        h_list = x_list
        x_list = self.attn(x_list)
        x_list = [self.attention_norm(x + h) for x, h in zip(x_list, h_list)]
        # feed forward
        x_list = [self.ffn_norm(self.ffn(x) + x) for x in x_list]
        return x_list


class MMEncoder(nn.Module):
    
    def __init__(self, hidden_size, num_layers, num_heads):
        super(MMEncoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        for _ in range(num_layers):
            layer = MMBlock(hidden_size, num_heads)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x_list):
        for layer_block in self.layer:
            x_list = layer_block(x_list)
        x_list = [self.encoder_norm(x) for x in x_list]
        return x_list


class MMTransformer(nn.Module):
    
    def __init__(
            self,
            in_channels, out_channels, hidden_size,
            feature_size, patch_size,
            num_layers, num_heads,
        ):
        super().__init__()
        self.embeddings = Embeddings(feature_size, patch_size, in_channels, hidden_size)
        self.down_sampled_size = np.array(feature_size) // np.array(patch_size)
        self.encoder = MMEncoder(hidden_size, num_layers, num_heads)
        if len(feature_size) == 3:
            # threeD
            self.channel_conv = nn.Conv3d(hidden_size, out_channels, 1, 1, 0)
        else:
            # twoD
            self.channel_conv = nn.Conv2d(hidden_size, out_channels, 1, 1, 0)

    def forward(self, features_list):
        x_list = [self.embeddings(features) for features in features_list]
        x_list = self.encoder(x_list)
        #
        res = []
        for x in x_list:
            x = x.transpose(-1, -2)
            x = x.view(x.shape[0], x.shape[1], *self.down_sampled_size)
            x = self.channel_conv(x)
            res.append(x)
        return res


class UNetCaFaBody(nn.Module):

    def __init__(
            self,
            num_modals, infer_size, num_pool=5,
            num_trans_layers=2, num_trans_heads=16,
            threeD=True, **kwargs
        ):
        super().__init__()
        self.num_modals = num_modals
        self.encoders = [
            UNetBody(in_channels=1, num_pool=(num_pool-1), threeD=threeD, **kwargs)
            for _ in range(num_modals)
        ]
        for encoder in self.encoders:
            del encoder.conv_blocks_localization
            del encoder.tu
            del encoder.seg_outputs
        self.encoders = nn.ModuleList(self.encoders)
        self.decoder = UNetBody(in_channels=1, num_pool=num_pool, threeD=threeD, **kwargs)
        del self.decoder.conv_blocks_context
        del self.decoder.td
        #
        self.mergers = []
        for num_features in self.decoder.skip_num_features_list:
            self.mergers.append(TwinMerge(num_modals, num_features, threeD))
        self.mergers = nn.ModuleList(self.mergers)
        patch_size = self.decoder.pool_op_kernel_sizes[-1]
        feature_size = (
            np.array(infer_size)
            // np.prod(np.array(self.decoder.pool_op_kernel_sizes)[: -1], axis=0)
        )
        self.trans_context = MMTransformer(
            in_channels=self.decoder.skip_num_features_list[-1],
            out_channels=self.decoder.skip_num_features_list[-1],
            hidden_size=self.decoder.skip_num_features_list[-1],
            feature_size=feature_size, patch_size=patch_size,
            num_layers=num_trans_layers, num_heads=num_trans_heads
        )

    def encode(self, x, modal_ind):
        skips = []
        one_modal_x = x[:, modal_ind : modal_ind + 1]
        encoder = self.encoders[modal_ind]
        for d in range(len(encoder.conv_blocks_context) - 1):
            one_modal_x = encoder.conv_blocks_context[d](one_modal_x)
            skips.append(one_modal_x)
            if not encoder.convolutional_pooling:
                one_modal_x = encoder.td[d](one_modal_x)
        one_modal_x = encoder.conv_blocks_context[-1](one_modal_x)
        skips.append(one_modal_x)
        return skips

    def forward(self, x):
        skips_list = []
        for modal_ind in range(self.num_modals):
            skips = self.encode(x, modal_ind)
            skips_list.append(skips)
        skips_list_T = list(zip(*skips_list))
        skips = [
            self.mergers[skip_ind](skips_list_T[skip_ind])
            for skip_ind in range(len(skips_list_T))
        ]
        x_list = self.trans_context(skips_list_T[-1])
        x = self.mergers[-1](x_list)
        #
        for u in range(len(self.decoder.tu)):
            x = self.decoder.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.decoder.conv_blocks_localization[u](x)
        seg_output = self.decoder.final_nonlin(self.decoder.seg_outputs[-1](x))
        return seg_output


class UNetCaFa(VanillaModel):
    
    def __init__(
            self, in_channels, num_classes, infer_size, num_trans_layers=2,
            num_pool=5, base_num_features=32, pool_op_kernel_sizes=None, **kwargs
        ):
        super().__init__(num_classes=num_classes, infer_size=infer_size, **kwargs)
        self.unet_body = UNetCaFaBody(
            num_modals=in_channels, infer_size=infer_size, num_pool=num_pool, num_trans_layers=num_trans_layers,
            num_classes=num_classes, threeD=(len(infer_size)==3), base_num_features=base_num_features,
            pool_op_kernel_sizes=pool_op_kernel_sizes,
        )


class UNetCaFaEnsemble(EnsembleModel):

    def __init__(
            self, in_channels, num_classes, infer_size, num_bodies,
            num_pool=5, base_num_features=32, pool_op_kernel_sizes=None, **kwargs
        ):
        super().__init__(num_classes=num_classes, infer_size=infer_size, **kwargs)
        self.unet_bodies = nn.ModuleList(
            [
                UNetCaFaBody(
                    num_modals=in_channels, infer_size=infer_size,
                    num_classes=num_classes, threeD=(len(infer_size)==3),
                    num_pool=num_pool, base_num_features=base_num_features,
                    pool_op_kernel_sizes=pool_op_kernel_sizes,
                )
                for _ in range(num_bodies)
            ]
        )


if __name__ == '__main__':
    infer_size = (128, 128)
    model = UNetCaFaBody(3, infer_size, threeD=(len(infer_size)==3), num_classes=2).cuda()
    x = torch.zeros(4, 3, *infer_size).cuda()
    with torch.no_grad():
        output = model(x)
    print(output.shape)
