import math
import random
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import scipy.fftpack
from torchvision.ops import deform_conv2d
try:
    from scipy.special import comb
except:
    from scipy.misc import comb

from .archs import VanillaModel, one_minus_dice, slide_infer


def randconv(image, fg_mask=None, K=3, mix=True, p=0.5):
    '''
    Outputs the image or the random convolution applied on the image.

    Args:
        image (torch.Tensor): input image
        K (int): maximum kernel size of the random convolution
        mix (bool)
        p (float): probability, 0 <= p <= 1
    '''

    p0 = torch.rand(1).item()
    if p0 < p:
        return image
    else:
        k = torch.randint(0, K + 1,  (1, )).item()  # k \in {0, 1, 2, ..., K}
        random_convolution = nn.Conv2d(1, 1, 2 * k + 1, padding=k).to(image.device)
        torch.nn.init.normal_(
            random_convolution.weight,
            0, 1. / ((2 * k + 1) ** 2)
        )
        image_rc = random_convolution(image)
        if mix:
            alpha = torch.rand(1,).item()
            return alpha * image + (1 - alpha) * image_rc
        else:
            return image_rc


class GradlessGCReplayNonlinBlock(nn.Module):
    
    def __init__(
            self, out_channel=32, in_channel=3, scale_pool=[1, 3], layer_id=0,
            use_act=True, requires_grad=False
        ):
        """
        Conv-leaky relu layer. Efficient implementation by using group convolutions
        """
        super(GradlessGCReplayNonlinBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.scale_pool = scale_pool
        self.layer_id = layer_id
        self.use_act = use_act
        self.requires_grad = requires_grad
        assert requires_grad == False

    def forward(self, x_in):
        """
        Args:
            x_in: [ nb (original), nc (original), nx, ny ]
        """
        # random size of kernel
        idx_k = torch.randint(high=len(self.scale_pool), size = (1,))
        k = self.scale_pool[idx_k[0]]
        #
        nb, nc, nx, ny = x_in.shape
        ker = torch.randn([self.out_channel * nb, self.in_channel, k, k], requires_grad=self.requires_grad).cuda()
        shift = torch.randn([self.out_channel * nb, 1, 1], requires_grad=self.requires_grad).cuda() * 1.0
        x_in = x_in.view(1, nb * nc, nx, ny)
        x_conv = F.conv2d(x_in, ker, stride=1, padding=(k // 2), dilation=1, groups=nb)
        x_conv = x_conv + shift
        if self.use_act:
            x_conv = F.leaky_relu(x_conv)
        x_conv = x_conv.view(nb, self.out_channel, nx, ny)
        return x_conv


class GINGroupConv(nn.Module):

    def __init__(
            self, out_channel=1, in_channel=1, interm_channel=2, scale_pool=[1, 3],
            n_layer=4, out_norm='frob',
        ):
        '''
        GIN
        '''
        super(GINGroupConv, self).__init__()
        self.scale_pool = scale_pool # don't make it tool large as we have multiple layers
        self.n_layer = n_layer
        self.layers = []
        self.out_norm = out_norm
        self.out_channel = out_channel
        self.layers.append(
            GradlessGCReplayNonlinBlock(
                out_channel=interm_channel, in_channel=in_channel,
                scale_pool=scale_pool, layer_id=0
            ).cuda()
        )
        for ii in range(n_layer - 2):
            self.layers.append(
                GradlessGCReplayNonlinBlock(
                    out_channel=interm_channel, in_channel=interm_channel,
                    scale_pool=scale_pool, layer_id=(ii + 1)
                ).cuda()
            )
        self.layers.append(
            GradlessGCReplayNonlinBlock(
                out_channel=out_channel, in_channel=interm_channel,
                scale_pool=scale_pool, layer_id=(n_layer - 1),
                use_act=False
            ).cuda()
        )
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x_in, fg_mask=None):
        # if isinstance(x_in, list):
        #     x_in = torch.cat(x_in, dim = 0)
        nb, nc, nx, ny = x_in.shape
        alphas = torch.rand(nb)[:, None, None, None] # nb, 1, 1, 1
        alphas = alphas.repeat(1, nc, 1, 1).cuda() # nb, nc, 1, 1
        #
        x = self.layers[0](x_in.detach())
        for blk in self.layers[1: ]:
            x = blk(x)
        # x = x.clamp(0, 1)  # DEBUG
        mixed = alphas * x + (1.0 - alphas) * x_in
        # print(x_in.quantile(torch.tensor([0, 0.05, 0.5, 0.95, 1.0]).to(x_in.device)).cpu().tolist())
        # print(x.quantile(torch.tensor([0, 0.05, 0.5, 0.95, 1.0]).to(x_in.device)).cpu().tolist())
        if self.out_norm == 'frob':
            _in_frob = torch.norm(x_in.view(nb, nc, -1), dim=(-1, -2), p='fro', keepdim=False)
            _in_frob = _in_frob[:, None, None, None].repeat(1, nc, 1, 1)
            _self_frob = torch.norm(mixed.view(nb, self.out_channel, -1), dim=(-1,-2), p='fro', keepdim=False)
            _self_frob = _self_frob[:, None, None, None].repeat(1, self.out_channel, 1, 1)
            # print(mixed.quantile(torch.tensor([0, 0.05, 0.5, 0.95, 1.0]).to(x_in.device)).cpu().tolist())
            mixed = mixed * (1.0 / (_self_frob.detach() + 1e-5)) * _in_frob.detach()
            # print(mixed.quantile(torch.tensor([0, 0.05, 0.5, 0.95, 1.0]).to(x_in.device)).cpu().tolist())
        # exit(0)
        # mixed = mixed.clamp(0, 1)  # DEBUG
        return mixed


def fftind(size):
    """ Returns a numpy array of shifted Fourier coordinates k_x k_y.
        
        Input args:
            size (integer): The size of the coordinate array to create
        Returns:
            k_ind, numpy array of shape (2, size, size) with:
                k_ind[0,:,:]:  k_x components
                k_ind[1,:,:]:  k_y components
                
        Example:
        
            print(fftind(5))
            
            [[[ 0  1 -3 -2 -1]
            [ 0  1 -3 -2 -1]
            [ 0  1 -3 -2 -1]
            [ 0  1 -3 -2 -1]
            [ 0  1 -3 -2 -1]]

            [[ 0  0  0  0  0]
            [ 1  1  1  1  1]
            [-3 -3 -3 -3 -3]
            [-2 -2 -2 -2 -2]
            [-1 -1 -1 -1 -1]]]
            
        """
    k_ind = np.mgrid[:size, :size] - int( (size + 1)/2 )
    k_ind = scipy.fftpack.fftshift(k_ind)
    return( k_ind )


def get_gaussian_random_field(alpha = 3.0,
                          size = 128, 
                          flag_normalize = True):
    """ Returns a numpy array of shifted Fourier coordinates k_x k_y.
        
        Input args:
            alpha (double, default = 3.0): 
                The power of the power-law momentum distribution
            size (integer, default = 128):
                The size of the square output Gaussian Random Fields
            flag_normalize (boolean, default = True):
                Normalizes the Gaussian Field:
                    - to have an average of 0.0
                    - to have a standard deviation of 1.0

        Returns:
            gfield (numpy array of shape (size, size)):
                The random gaussian random field
                
        Example:
        import matplotlib
        import matplotlib.pyplot as plt
        example = gaussian_random_field()
        plt.imshow(example)
        """
        
        # Defines momentum indices
    k_idx = fftind(size)

        # Defines the amplitude as a power law 1/|k|^(alpha/2)
    amplitude = np.power( k_idx[0]**2 + k_idx[1]**2 + 1e-10, -alpha/4.0 )
    amplitude[0,0] = 0
    
        # Draws a complex gaussian random noise with normal
        # (circular) distribution
    noise = np.random.normal(size = (size, size)) \
        + 1j * np.random.normal(size = (size, size))
    
        # To real space
    gfield = np.fft.ifft2(noise * amplitude).real
    
        # Sets the standard deviation to one
    if flag_normalize:
        gfield = gfield - np.mean(gfield)
        gfield = gfield/np.std(gfield)
        
    return gfield


def pro_randconv(
        image, fg_mask=None,
        sigma_gamma=0.5, sigma_beta=0.5, b_g=1.0, alpha=10, b_delta=0.2,
        max_rep_times=10, K=3,
    ):
    '''
    Outputs the image or the random convolution applied on the image.

    Args:
        image (torch.Tensor): input image, image[i, j, k, l] \in [0, 1]
    '''
    image = image.clamp(0, 1)
    eps = 0.01
    origin_image = image
    image = (image - 0.5) * 2
    # for deform conv
    weight = torch.zeros(1, 1, K, K)
    torch.nn.init.normal_(weight, 0, 1. / (K ** 2))
    sigma_g = random.uniform(eps, b_g)
    gaussian_smooth = ( 
        -(
            (torch.arange(K) - K // 2).view(-1, 1).float() ** 2
            + (torch.arange(K) - K // 2).view(1, -1).float() ** 2
        ) / (2 * sigma_g ** 2)
    ).exp()
    weight = weight * gaussian_smooth[None, None]
    weight = weight.to(image.device)
    #
    sigma_delta = random.uniform(eps, b_delta)
    offset = torch.zeros(image.shape[0], 2 * K * K, image.shape[2], image.shape[3])
    torch.nn.init.normal_(offset, 0, sigma_delta)
    # gaussian_field = get_gaussian_random_field(alpha, image.shape[2])  # (H, W), W=H
    # gaussian_field = torch.tensor(gaussian_field).float()
    # offset = offset * gaussian_field[None, None]
    offset = offset.to(image.device)
    # for affine transform
    gamma = torch.zeros(1, image.shape[1], 1, 1)
    torch.nn.init.normal_(gamma, 0, sigma_gamma)
    gamma = gamma.to(image.device)
    beta = torch.zeros(1, image.shape[1], 1, 1)
    torch.nn.init.normal_(beta, 0, sigma_beta)
    beta = beta.to(image.device)
    #
    rep_times = random.sample(range(1, max_rep_times + 1), 1)[0]
    for _ in range(rep_times):
        # deform conv
        image = deform_conv2d(image, offset, weight, padding=(K // 2))
        # image = F.conv2d(image, weight, padding=(K // 2))  # For debug
        # standardization & affine transform
        image = gamma * (
            (image - image.mean(dim=(2, 3), keepdim=True))
            / (image.std(dim=(2, 3), keepdim=True) + eps)
        ) + beta
        # tanh
        image = image.tanh()
    image = (image + 1) / 2
    #
    # import os
    # from PIL import Image
    # os.system('mkdir -p debug')
    # images = image.cpu().numpy()  # (bs, 1, d, h, w)
    # images = images.reshape(-1, *images.shape[-2 :])  # (bs * d, h, w)
    # images = (images - np.percentile(images, 1)) / (np.percentile(images, 99) - np.percentile(images, 1))
    # rgb_images = (np.clip(np.concatenate([images[..., None]] * 3, axis=3), 1e-4, 1 - 1e-4) * 255).astype(np.uint8)
    # for image_ind, rgb_image in enumerate(rgb_images):
    #     Image.fromarray(rgb_image).save(f'debug/{image_ind}.jpg')
    # exit(0)
    # mix
    alphas = torch.rand(image.shape[0])[:, None, None, None] # nb, 1, 1, 1
    alphas = alphas.repeat(1, image.shape[1], 1, 1).to(image.device) # nb, nc, 1, 1
    image = alphas * image + (1.0 - alphas) * origin_image
    return image


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.
       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000
        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


class BezierCurve:
    
    def __init__(self, ):
        points_1 = [[0, 0], [0, 0], [0.5, 0.5], [1, 1]]
        xvals_1, yvals_1 = bezier_curve(points_1, nTimes=100000)
        xvals_1 = np.sort(xvals_1)
        self.xvals_1, self.yvals_1 = xvals_1, yvals_1
        #
        points_2 = [[0, 0], [0.25, 0.75], [0.75, 0.25], [1, 1]]
        xvals_2, yvals_2 = bezier_curve(points_2, nTimes=100000)
        xvals_2 = np.sort(xvals_2)
        yvals_2 = np.sort(yvals_2)
        self.xvals_2, self.yvals_2 = xvals_2, yvals_2
        #
        points_3 = [[0, 0], [0.25, 0.75], [0.75, 0.25], [1, 1]]
        xvals_3, yvals_3 = bezier_curve(points_3, nTimes=100000)
        xvals_3 = np.sort(xvals_3)
        self.xvals_3, self.yvals_3 = xvals_3, yvals_3
        #
        points_4 = [[0, 0], [0.125, 0.875], [0.875, 0.125], [1, 1]]
        xvals_4, yvals_4 = bezier_curve(points_4, nTimes=100000)
        xvals_4 = np.sort(xvals_4)
        yvals_4 = np.sort(yvals_4)
        self.xvals_4, self.yvals_4 = xvals_4, yvals_4
        #
        points_5 = [[0, 0], [0.125, 0.875], [0.875, 0.125], [1, 1]]
        xvals_5, yvals_5 = bezier_curve(points_5, nTimes=100000)
        xvals_5 = np.sort(xvals_5)
        self.xvals_5, self.yvals_5 = xvals_5, yvals_5

    def __call__(self, image, fg_mask, case=None):
        """
        slices, nonlinear_slices_2, nonlinear_slices_4 are source-similar images
        nonlinear_slices_1, nonlinear_slices_3, nonlinear_slices_5 are source-dissimilar images
        """
        device = image.device
        slices = image.clamp(0, 1).cpu().numpy()
        bg_mask = (fg_mask == 0).cpu().numpy()
        if not case:
            # case = np.random.randint(0, 6)
            case = np.random.randint(0, 3) * 2 + 1
        if case == 1:
            nonlinear_slices_1 = np.interp(slices, self.xvals_1, self.yvals_1)
            # nonlinear_slices_1[nonlinear_slices_1 >= 0.9] = 0
            nonlinear_slices_1[bg_mask] = slices[bg_mask]
            slices = nonlinear_slices_1
        elif case == 2:
            nonlinear_slices_2 = np.interp(slices, self.xvals_2, self.yvals_2)
            slices = nonlinear_slices_2
        elif case == 3:
            nonlinear_slices_3 = np.interp(slices, self.xvals_3, self.yvals_3)
            # nonlinear_slices_3[nonlinear_slices_3 == 1] = 0
            nonlinear_slices_3[bg_mask] = slices[bg_mask]
            slices = nonlinear_slices_3
        elif case == 4:
            nonlinear_slices_4 = np.interp(slices, self.xvals_4, self.yvals_4)
            slices = nonlinear_slices_4
        elif case == 5:
            nonlinear_slices_5 = np.interp(slices, self.xvals_5, self.yvals_5)
            # nonlinear_slices_5[nonlinear_slices_5 == 1] = 0
            nonlinear_slices_5[bg_mask] = slices[bg_mask]
            slices = nonlinear_slices_5
        # print(case, np.min(slices), np.max(slices))
        # DEBUG
        # import os
        # from PIL import Image
        # os.system('mkdir -p debug')
        # images = slices  # (bs, 1, d, h, w)
        # images = images.reshape(-1, *images.shape[-2 :])  # (bs * d, h, w)
        # images = (images - np.percentile(images, 1)) / (np.percentile(images, 99) - np.percentile(images, 1))
        # rgb_images = (np.clip(np.concatenate([images[..., None]] * 3, axis=3), 1e-4, 1 - 1e-4) * 255).astype(np.uint8)
        # for image_ind, rgb_image in enumerate(rgb_images):
        #     Image.fromarray(rgb_image).save(f'debug/{image_ind}.jpg')
        # exit(0)
        return torch.tensor(slices).float().to(device)


class RandomConv(VanillaModel):

    def __init__(self, conv_type='rand_conv', **kwargs):
        super().__init__(**kwargs)
        self.lamb = 10.
        self.conv_type = conv_type
        if conv_type == 'rand_conv':
            self.conv = randconv
        elif conv_type == 'gin_conv':
            self.conv = GINGroupConv()
        elif conv_type == 'enc_gin_conv':
            channels = [1, 1, 32, 24, 48, 120]
            self.enc_convs = [GINGroupConv(channel, channel, channel) for channel in channels]
        elif conv_type == 'pro_conv':
            self.conv = pro_randconv
        elif conv_type == 'bezier':
            self.conv = BezierCurve()
        else:
            raise Exception('Unknown conv_type', conv_type)

    def forward(self, data_dict):
        images = data_dict['images']    # (bs, c, d, h, w) or (bs, c, depth_2d, h, w)
        targets = data_dict['labels'] # (bs, d, h, w)    or (bs, depth_2d, h, w)
        masks = data_dict['masks']    # (bs, 1, d, h, w) or (bs, 1, depth_2d, h, w)
        # fg_masks = data_dict['fg_masks']    # (bs, c, d, h, w) or (bs, c, depth_2d, h, w)
        assert not self.threeD
        bs, c, depth_2d, h, w = images.shape
        images = images.permute(0, 2, 1, 3, 4).contiguous().view(bs * depth_2d, c, h, w)
        masks = masks.view(bs * depth_2d, 1, h, w)
        fg_masks = None
        # fg_masks = fg_masks.permute(0, 2, 1, 3, 4).contiguous().view(bs * depth_2d, c, h, w)
        if self.conv_type == 'enc_gin_conv':
            preds_list = torch.cat(
                [
                    (self.unet_body(images, self.enc_convs) * masks)[None]
                    for _ in range(3)
                ],
                dim=0
            )  # (3, bs * depth_2d, num_classes, h, w)
        else:
            with torch.no_grad():
                conved_images_list = [self.conv(images, fg_masks).detach() for _ in range(3)]
            preds_list = torch.cat(
                [
                    (self.unet_body(conved_images) * masks)[None]
                    for conved_images in conved_images_list
                ],
                dim=0
            )  # (3, bs * depth_2d, num_classes, h, w)
        prob_preds_list = preds_list.softmax(dim=2)
        with torch.no_grad():
            prob_preds_mean = torch.mean(prob_preds_list, dim=0).detach()
        # Consist loss
        consist_loss = 0.
        for preds in preds_list:
            consist_loss = consist_loss + F.kl_div(
                F.log_softmax(preds, dim=1), prob_preds_mean,
            )
        # Segmentation loss
        targets = targets.view(bs * depth_2d, h, w)
        dice_loss = one_minus_dice(preds_list[0], targets, self.dice_weight.to(images.device))
        ce_loss = F.cross_entropy(preds_list[0], targets.long(), self.ce_weight.to(images.device))
        # ce_loss = self.cross_entropy(preds, targets.long())
        total_loss = dice_loss + ce_loss + self.lamb * consist_loss
        return (
            ['DiceLoss', 'CELoss', 'Consist', 'Total'],
            [round(dice_loss.item(), 3), round(ce_loss.item(), 3), round(consist_loss.item(), 3), total_loss.item()],
            total_loss
        )

    # def infer(self, data_dict):
    #     images = data_dict['images']  # (bs, 1, d, h, w) or (bs, 1, depth_2d, h, w)
    #     assert not self.threeD
    #     bs, c, depth_2d, h, w = images.shape
    #     images = images.permute(0, 2, 1, 3, 4).contiguous().view(bs * depth_2d, c, h, w)
    #     fg_masks = data_dict['fg_masks']    # (bs, c, d, h, w) or (bs, c, depth_2d, h, w)
    #     fg_masks = fg_masks.permute(0, 2, 1, 3, 4).contiguous().view(bs * depth_2d, c, h, w)
    #     with torch.no_grad():
    #         outputs = []
    #         # for _ in range(3):
    #         for case in range(6):
    #             conved_images = self.conv(images, fg_masks, case).detach()  # (bs * depth_2d, c, h, w)
    #             conved_images = conved_images.view(bs, depth_2d, c, h, w).permute(0, 2, 1, 3, 4)
    #             outputs.append(slide_infer(conved_images, self.gaussian_map, self.unet_body, self.num_classes, self.infer_strides)[None])
    #         output = torch.mean(torch.cat(outputs, dim=0), dim=0)
    #     return torch.max(output, dim=1)


class CutOut(VanillaModel):

    def __init__(self, num_holes, cut_length, **kwargs):
        super().__init__(**kwargs)
        self.num_holes = num_holes
        self.cut_length = cut_length

    def _cutout(self, image, target):
        """
        Args:
            image (Tensor): Tensor of size (C, H, W).
            target (Tensor): Tensor of size (H, W).
        Returns:
            Tensor: image & target with num_holes of dimension length x length cut out of it.
        """
        h = image.size(1)
        w = image.size(2)
        mask = np.ones((h, w), np.float32)
        for _ in range(self.num_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.cut_length // 2, 0, h)
            y2 = np.clip(y + self.cut_length // 2, 0, h)
            x1 = np.clip(x - self.cut_length // 2, 0, w)
            x2 = np.clip(x + self.cut_length // 2, 0, w)
            mask[y1 : y2, x1 : x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(image).to(image.device)
        image = image * mask
        target = target * mask
        return image, target

    def forward(self, data_dict):
        images = data_dict['images']    # (bs, 1, d, h, w) or (bs, 1, depth_2d, h, w)
        targets = data_dict['labels'] # (bs, d, h, w)    or (bs, depth_2d, h, w)
        for i in range(images.shape[0]):
            for j in range(images.shape[2]):
                images[i, :, j], targets[i, j] = self._cutout(images[i, :, j], targets[i, j])
        return super().forward(data_dict)


class MixStyle(VanillaModel):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random', **kwargs):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__(**kwargs)
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def _mix_style(self, x):
        '''
        Args:
            x (Tensor): feature, shape = (B, C, H, W)
        '''
        if not self.training or not self._activated:
            return x
        if random.random() > self.p:
            return x
        #
        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig
        #
        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)
        #
        if self.mix == 'random':
            # random shuffle
            perm = torch.randperm(B)
        elif self.mix == 'crossdomain':
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1) # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(B // 2)]
            perm_a = perm_a[torch.randperm(B // 2)]
            perm = torch.cat([perm_b, perm_a], 0)
        else:
            raise NotImplementedError
        #
        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)
        #
        return x_normed * sig_mix + mu_mix


class RSC(VanillaModel):

    def __init__(self, iters_per_epoch, **kwargs):
        super().__init__(**kwargs)
        self.iters_per_epoch = iters_per_epoch
        self.iter = 0
        self.percent = 1 / 3

    def _challenge(self, x, target, seg_head):
        '''
        Args:
            target (Tensor): shape of (B, H, W)
        '''
        interval = 10
        self.iter += 1
        epoch = self.iter // self.iters_per_epoch
        if epoch % interval == 0:
            self.pecent = min(3.0 / 10 + (epoch / interval) * 2.0 / 10, 0.7)
        #
        seg_head.eval()
        x_new = x.clone().detach()
        x_new = Variable(x_new.data, requires_grad=True)
        output = seg_head(x_new)  # (B, num_classes, H, W)
        class_num = output.shape[1]
        num_rois = x_new.shape[0]
        num_channel = x_new.shape[1]
        H, W = x_new.shape[2 :]
        HW = x_new.shape[2] * x_new.shape[3]
        #
        seg_head.zero_grad()
        one_hot = target[:, None] == torch.arange(class_num).view(1, -1, 1, 1).to(x.device)
        torch.sum(output * one_hot).backward()
        grads_val = x_new.grad.clone().detach()
        grad_channel_mean = torch.mean(grads_val.view(num_rois, num_channel, -1), dim=2)
        channel_mean = grad_channel_mean
        grad_channel_mean = grad_channel_mean.view(num_rois, num_channel, 1, 1)
        spatial_mean = torch.sum(x_new * grad_channel_mean, dim=1)
        spatial_mean = spatial_mean.view(num_rois, HW)
        seg_head.zero_grad()
        #
        choose_one = random.randint(0, 9)
        if choose_one <= 4:
            # ---------------------------- spatial -----------------------
            spatial_drop_num = math.ceil(HW * 1 / 3.0)
            th18_mask_value = torch.sort(
                spatial_mean, dim=1, descending=True
            )[0][:, spatial_drop_num]
            th18_mask_value = th18_mask_value.view(num_rois, 1).expand(num_rois, HW)
            mask_all_cuda = torch.where(
                spatial_mean > th18_mask_value,
                torch.zeros(spatial_mean.shape).cuda(),
                torch.ones(spatial_mean.shape).cuda()
            )
            mask_all = mask_all_cuda.reshape(num_rois, H, W).view(num_rois, 1, H, W)
        else:
            # -------------------------- channel ----------------------------
            vector_thresh_percent = math.ceil(num_channel * 1 / 3.2)
            vector_thresh_value = torch.sort(
                channel_mean, dim=1, descending=True
            )[0][:, vector_thresh_percent]
            vector_thresh_value = vector_thresh_value.view(num_rois, 1).expand(num_rois, num_channel)
            vector = torch.where(
                channel_mean > vector_thresh_value,
                torch.zeros(channel_mean.shape).cuda(),
                torch.ones(channel_mean.shape).cuda()
            )
            mask_all = vector.view(num_rois, num_channel, 1, 1)
        # ----------------------------------- batch ----------------------------------------
        with torch.no_grad():
            cls_prob_before = F.softmax(output, dim=1).detach()
            x_new_view_after = x_new * mask_all
            x_new_view_after = seg_head(x_new_view_after).detach()
            cls_prob_after = F.softmax(x_new_view_after, dim=1).detach()
        #
        before_vector = torch.sum(one_hot * cls_prob_before, dim=1)
        after_vector = torch.sum(one_hot * cls_prob_after, dim=1)
        change_vector = before_vector - after_vector - 0.0001
        change_vector = torch.where(
            change_vector > 0,
            change_vector,
            torch.zeros(change_vector.shape).cuda()
        )
        th_fg_value = torch.sort(
            change_vector, dim=0, descending=True
        )[0][int(round(float(num_rois) * self.pecent))]
        drop_index_fg = change_vector.gt(th_fg_value).long()
        ignore_index_fg = 1 - drop_index_fg
        not_01_ignore_index_fg = ignore_index_fg.nonzero()[:, 0]
        mask_all[not_01_ignore_index_fg.long(), :] = 1
        #
        seg_head.train()
        mask_all = Variable(mask_all, requires_grad=True)
        return x * mask_all


def get_masks(shape, ratio):
    assert (ratio > 0) and (ratio < 1)
    bs, _, h, w = shape
    masks = torch.zeros(bs, 1, h, w)
    mask_h, mask_w = int(h * ratio), int(w * ratio)
    for b_ind in range(bs):
        left_h = random.randint(a=0, b=(h - mask_h))
        left_w = random.randint(a=0, b=(w - mask_w))
        masks[b_ind, :, left_h : (left_h + mask_h), left_w : (left_w + mask_w)] = 1
    return masks


def get_ConvProj(in_channels, hidden_size, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden_size, kernel_size=1),
        nn.BatchNorm2d(hidden_size),
        nn.ReLU(),
        nn.Conv2d(hidden_size, out_channels, kernel_size=1)
    )
    

class ContraConv(VanillaModel):

    def __init__(
            self, num_classes,
            conv_times=3, conv_pre_norm=False,
            use_mask=False, mask_ratio=-1, mask_alpha=1.0,
            use_div_convs=False,
            sup_on_all=False, consist_kind='KL',
            bound_size=1,
            lamb=10, beta=0.05, contra_tau=0.07, contra_rampup=20_000,
            contra_kind='MultiPos',
            **kwargs
        ):
        super().__init__(num_classes=num_classes, **kwargs)
        self.num_classes = num_classes
        #
        self.conv_times = conv_times
        assert conv_pre_norm in [True, False]
        self.conv_pre_norm = conv_pre_norm
        # self.conv = GINGroupConv(out_norm=conv_out_norm)
        self.conv = GINGroupConv()
        #
        assert use_mask in [True, False]
        self.use_mask = use_mask
        self.mask_ratio = mask_ratio
        assert (mask_alpha <= 1.0 + 1e-6) and (mask_alpha >= 0.0 - 1e-6)
        self.mask_alpha = mask_alpha
        #
        assert use_div_convs in [True, False]
        self.use_div_convs = use_div_convs
        #
        assert sup_on_all in [True, False]
        self.sup_on_all = sup_on_all
        for sub_kind in consist_kind.split('+'):
            assert sub_kind in ['KL', 'MSEE', 'LocContra', 'GlobContra', 'BoundContra', '']
        self.consist_kind = consist_kind
        if 'LocContra' in self.consist_kind:
            self.loc_proj = get_ConvProj(32, 32, 32)
        if 'GlobContra' in self.consist_kind:
            self.glob_proj = get_ConvProj(32, 32, 32)
        if 'BoundContra' in self.consist_kind:
            self.bound_proj = get_ConvProj(32, 32, 32)
        #
        assert bound_size > 0
        self.bound_size = bound_size
        self.lamb = lamb
        self.beta = beta
        self.contra_tau = contra_tau
        self.contra_rampup = contra_rampup
        #
        assert contra_kind in ['MultiPos', 'InfoNCE', 'PixelInfoNCE']
        self.contra_kind = contra_kind
        #
        self.cur_iter = 0

    def forward(self, data_dict):
        images = data_dict['images']    # (bs, c, d, h, w) or (bs, c, depth_2d, h, w)
        if self.conv_pre_norm:
            images = images - 0.5
        targets = data_dict['labels']   # (bs, d, h, w)    or (bs, depth_2d, h, w)
        assert not self.threeD
        bs, c, depth_2d, h, w = images.shape
        device = images.device
        images = images.permute(0, 2, 1, 3, 4).contiguous().view(bs * depth_2d, c, h, w)
        targets = targets.view(bs * depth_2d, h, w)
        class_masks_list = [(targets == class_ind)[:, None] for class_ind in range(self.num_classes)]
        # Prepare input
        with torch.no_grad():
            if self.use_div_convs:
                conved_images_list = []
                conved_images = self.conv(images).detach() * class_masks_list[0]  # bg
                for class_masks in class_masks_list[1 :]:
                    conved_images = conved_images + self.conv(images).detach() * class_masks
                conved_images_list.append(conved_images)
                for _ in range(self.conv_times - 1):
                    conved_images_list.append(self.conv(images).detach())
            else:
                conved_images_list = [
                    self.conv(images).detach()
                    for _ in range(self.conv_times)
                ]
            # for i in range(3):
            #     torch.save(conved_images_list[i].cpu(), f'conved_images_old_{i}.pth')
            if self.use_mask:
                # masks = get_masks(images.shape, self.mask_ratio).to(images.device) * self.mask_alpha
                for start_ind in range(0, (self.conv_times - 1) // 2 * 2, 2):
                    # start_ind = 0
                    masks = get_masks(images.shape, self.mask_ratio).to(images.device) * self.mask_alpha
                    inv_masks = 1 - masks
                    conved_images_list[start_ind : start_ind + 2] = (
                        conved_images_list[start_ind] * masks + conved_images_list[start_ind + 1] * inv_masks,
                        conved_images_list[start_ind] * inv_masks + conved_images_list[start_ind + 1] * masks,
                    )
        # torch.save(targets.cpu(), 'targets.pth')
        # torch.save(images.cpu(), 'ori_images.pth')
        # for i in range(3):
        #     torch.save(conved_images_list[i].cpu(), f'conved_images_{i}.pth')
        # torch.save(masks.cpu(), 'masks.pth')
        # exit(0)
        # Forward
        feats_list = []
        preds_list = []
        for conved_images in conved_images_list:
            feats, preds = self.unet_body(conved_images, ret_last_feat=True)
            feats_list.append(feats[None])
            preds_list.append(preds[None])
        feats_list = torch.cat(feats_list, dim=0)  # (conv_times, bs * depth_2d, num_channels, h, w)
        num_channels = feats_list.shape[2]
        preds_list = torch.cat(preds_list, dim=0)  # (conv_times, bs * depth_2d, num_classes, h, w)
        with torch.no_grad():
            prob_preds_mean = torch.mean(preds_list.softmax(dim=2), dim=0).detach()
            entropy_preds_mean = -prob_preds_mean * (prob_preds_mean + 1e-8).log()
        # Consist loss
        consist_loss = 0.
        for preds in preds_list:
            if 'KL' in self.consist_kind:
                consist_loss = consist_loss + F.kl_div(
                    F.log_softmax(preds, dim=1), prob_preds_mean,
                )
            if 'MSEE' in self.consist_kind:
                consist_loss = consist_loss + F.mse_loss(
                    -preds.softmax(dim=1) * F.log_softmax(preds, dim=1),
                    entropy_preds_mean
                )
            if self.consist_kind == '':
                consist_loss = torch.zeros(1).to(device)
        # Contra loss
        contra_loss = torch.zeros(1).to(device)
        if 'LocContra' in self.consist_kind:
            # Add a non-linear predictor
            proj_feats_list = self.loc_proj(feats_list.view(-1, *feats_list.shape[2 :])).view(self.conv_times, -1, *feats_list.shape[2 :])
            loc_contra_loss = ((self.conv_times ** 2) / 2) * torch.mean((proj_feats_list[None] - proj_feats_list[:, None]) ** 2)
            # consist_loss = consist_loss + loc_contra_loss
            contra_loss = contra_loss + loc_contra_loss
        if 'GlobContra' in self.consist_kind:
            # Add a non-linear predictor
            # print('feats_list', feats_list.shape)
            proj_feats_list = self.glob_proj(feats_list.view(-1, *feats_list.shape[2 :])).view(self.conv_times, -1, *feats_list.shape[2 :])
            # print('proj_feats_list', proj_feats_list.shape)
            num_feats_per_sample = self.num_classes * self.conv_times
            num_samples = bs * depth_2d
            # print('num_feats_per_sample, num_samples', num_feats_per_sample, num_samples)
            contra_mask = (
                torch.arange(self.num_classes * self.conv_times).view(-1, 1)
                != torch.arange(self.num_classes * self.conv_times).view(1, -1)
            ).to(device)  # (#feats_per_sample, #feats_per_sample)
            # print('contra_mask', contra_mask)
            # average pooling
            glob_feats_matrix = []
            valid_sample_mask = torch.ones(num_samples, dtype=bool).to(device)
            for class_masks in class_masks_list:
                class_masks = F.max_pool2d(class_masks.float(), kernel_size=2)
                # class_masks.shape = (#samples, 1, h // 2, w // 2)
                class_area = torch.sum(class_masks, dim=(-1, -2))[None]  # (1, #samples, 1)
                valid_sample_mask = valid_sample_mask & (class_area > 0)[0, :, 0]
                # print('class_area', class_area)
                glob_feats_list = (
                    torch.sum(proj_feats_list * class_masks[None], dim=(-1, -2))
                    / (class_area + 1e-6)
                )  # (conv_times, #samples, #channels)
                glob_feats_matrix.append(glob_feats_list[None])
                # print('glob_feats_list', glob_feats_list.shape)
            # print('valid_sample_mask', valid_sample_mask)
            glob_feats_matrix = torch.cat(glob_feats_matrix, dim=0).permute(2, 0, 1, 3)  # (#samples, #classes, conv_times, #channels)
            # print('glob_feats_matrix', glob_feats_matrix.shape)
            # contrastive learning
            # glob_feats_matrix = F.normalize(glob_feats_matrix, p=2, dim=-1)
            # logits = (
            #     glob_feats_matrix.view(num_samples, num_feats_per_sample, -1)
            #     @ glob_feats_matrix.view(num_samples, num_feats_per_sample, -1).permute(0, 2, 1)
            # )  # (num_samples, num_feats_per_sample, num_feats_per_sample)
            logits = F.cosine_similarity(
                glob_feats_matrix.view(num_samples, num_feats_per_sample, 1, -1),
                glob_feats_matrix.view(num_samples, 1, num_feats_per_sample, -1),
                dim=-1
            )  # (num_samples, num_feats_per_sample, num_feats_per_sample)
            # print('logits', logits.shape)
            # print('logits[0]', logits[0])
            # glob_feats_matrix = glob_feats_matrix.contiguous()
            # logits = -torch.cdist(
            #     glob_feats_matrix.view(num_samples, num_feats_per_sample, -1),
            #     glob_feats_matrix.view(num_samples, num_feats_per_sample, -1)
            # )  # (num_samples, num_feats_per_sample, num_feats_per_sample)
            logits = logits[:, contra_mask].view(num_samples, num_feats_per_sample, num_feats_per_sample - 1)
            pos_mask = (torch.arange(self.num_classes).view(-1, 1) + torch.zeros(1, self.conv_times))
            pos_mask = (pos_mask.view(-1, 1) == pos_mask.view(1, -1)).to(device)  # (#feats_per_sample, #feats_per_sample)
            pos_mask = pos_mask[contra_mask].view(1, num_feats_per_sample, num_feats_per_sample - 1)
            # print('pos_mask[0]', pos_mask[0])
            # print('pos_nume', torch.sum((pos_mask * logits)[valid_sample_mask]))
            # print('pos_deno', torch.sum(pos_mask) * torch.sum(valid_sample_mask) + 1e-6)
            # print('neg_nume', torch.sum((~pos_mask * logits)[valid_sample_mask]))
            # print('neg_deno', torch.sum(~pos_mask) * torch.sum(valid_sample_mask) + 1e-6)
            glob_contra_loss = (
                1 - (torch.sum((pos_mask * logits)[valid_sample_mask]) / (torch.sum(pos_mask) * torch.sum(valid_sample_mask) + 1e-6))
                + 1 + (torch.sum((~pos_mask * logits)[valid_sample_mask]) / (torch.sum(~pos_mask) * torch.sum(valid_sample_mask) + 1e-6))
            ) / 2
            # print('contra_loss', contra_loss)
            # exit(0)
            # consist_loss = consist_loss + contra_loss
            contra_loss = contra_loss + glob_contra_loss
        if 'BoundContra' in self.consist_kind:
            # Add a non-linear predictor
            # proj_feats_list.shape = (conv_times, #samples, num_channels, h // 2, w // 2)
            proj_feats_list = self.bound_proj(feats_list.view(-1, *feats_list.shape[2 :])).view(self.conv_times, -1, *feats_list.shape[2 :])
            num_feats_per_sample = self.num_classes * self.conv_times
            num_samples = bs * depth_2d
            contra_mask = (
                torch.arange(self.conv_times).view(-1, 1)
                != torch.arange(self.conv_times).view(1, -1)
            ).to(device)  # (#feats_per_sample, #feats_per_sample)
            # average pooling
            bound_contra_loss = 0.
            for class_masks in class_masks_list[1 :]:
                # class_masks.shape = (#samples, 1, h // 2, w // 2)
                class_masks = F.max_pool2d(class_masks.float(), kernel_size=2)
                class_area = torch.sum(class_masks, dim=(-1, -2))[None]  # (1, #samples, 1)
                valid_sample_mask = (class_area > 0)[0, :, 0]  # (#samples, )
                mid_class_masks = F.max_pool2d(class_masks, kernel_size=3, stride=1, padding=1)
                dilate_class_masks = mid_class_masks
                for _ in range(self.bound_size):
                    dilate_class_masks = F.max_pool2d(dilate_class_masks, kernel_size=3, stride=1, padding=1)
                neg_masks = (dilate_class_masks - mid_class_masks)
                #
                if self.contra_kind == 'PixelInfoNCE':
                    bound_contra_loss = 0
                    for sample_ind in range(proj_feats_list.shape[1]):
                        if class_area[0, sample_ind, 0] <= 0:
                            continue
                        # pos_feats.shape = (conv_times, num_channels, #pos_pixels)
                        pos_feats = proj_feats_list[:, sample_ind, :, class_masks[sample_ind, 0].bool()]
                        pos_feats = pos_feats.contiguous().permute(0, 2, 1).contiguous().view(-1, num_channels).contiguous()
                        max_num_feats = 300
                        if pos_feats.shape[0] > max_num_feats:
                            pos_feats = pos_feats[torch.LongTensor(random.sample(range(pos_feats.shape[0]), max_num_feats))].contiguous()
                        # neg_feats.shape = (conv_times, num_channels, #neg_pixels)
                        neg_feats = proj_feats_list[:, sample_ind, :, neg_masks[sample_ind, 0].bool()]
                        neg_feats = neg_feats.contiguous().permute(0, 2, 1).contiguous().view(-1, num_channels).contiguous()
                        if neg_feats.shape[0] > max_num_feats:
                            neg_feats = neg_feats[torch.LongTensor(random.sample(range(neg_feats.shape[0]), max_num_feats))].contiguous()
                        #
                        contra_mask = (
                            torch.arange(pos_feats.shape[0]).view(-1, 1)
                            != torch.arange(pos_feats.shape[0]).view(1, -1)
                        ).to(device)  # (conv_times * #pos_pixels, conv_times * #pos_pixels)
                        pos_logits = pos_feats @ pos_feats.permute(1, 0)
                        pos_logits = pos_logits[contra_mask].view(pos_feats.shape[0], pos_feats.shape[0] - 1)
                        neg_logits = pos_feats @ neg_feats.permute(1, 0)
                        #
                        logits = torch.cat([pos_logits[..., None], neg_logits[..., None, :].expand(-1, pos_feats.shape[0] - 1, -1)], dim=-1)
                        logits = logits.view(-1, neg_feats.shape[0] + 1)
                        bound_contra_loss = bound_contra_loss + F.cross_entropy(logits, torch.zeros(logits.shape[0]).long().to(device)) 
                    num_valid_samples = torch.sum(valid_sample_mask)
                    if num_valid_samples == 0:
                        bound_contra_loss = torch.zeros(1).to(device)
                    else:
                        bound_contra_loss = bound_contra_loss / num_valid_samples
                else:
                    # average pooling
                    pos_feats_list = (
                        torch.sum(proj_feats_list * class_masks[None], dim=(-1, -2))
                        / (class_area + 1e-6)
                    )  # (conv_times, #samples, #channels)
                    neg_feats_list = (
                        torch.sum(proj_feats_list * neg_masks[None], dim=(-1, -2))
                        / (torch.sum(neg_masks, dim=(-1, -2))[None] + 1e-6)
                    )  # (conv_times, #samples, #channels)
                    # pos_logits = F.cosine_similarity(
                    #     pos_feats_list[:, None], pos_feats_list[None], dim=-1
                    # )  # (conv_times, conv_times, #samples)
                    # # pos_logits.shape = (conv_times, conv_times - 1, #samples)
                    # pos_logits = pos_logits[contra_mask].view(self.conv_times, self.conv_times - 1, -1)
                    # neg_logits = F.cosine_similarity(
                    #     pos_feats_list[:, None], neg_feats_list[None], dim=-1
                    # )  # (conv_times, conv_times, #samples)
                    #
                    # bound_contra_loss = (
                    #     bound_contra_loss
                    #     - torch.sum(pos_logits[..., valid_sample_mask]) / (self.conv_times * torch.sum(valid_sample_mask) + 1e-6)
                    #     + torch.sum(neg_logits[..., valid_sample_mask]) / (self.conv_times * torch.sum(valid_sample_mask) + 1e-6)
                    # ) + self.conv_times ** 2 - self.conv_times
                    #
                    pos_logits = pos_feats_list.permute(1, 0, 2) @ pos_feats_list.permute(1, 2, 0)  # (#samples, conv_times, conv_times)
                    pos_logits = pos_logits[:, contra_mask].view(-1, self.conv_times, self.conv_times - 1)  # (#samples, conv_times, conv_times - 1)
                    neg_logits = pos_feats_list.permute(1, 0, 2) @ neg_feats_list.permute(1, 2, 0)  # (#samples, conv_times, conv_times)
                    #
                    if self.contra_kind == 'MultiPos':
                        bound_contra_loss = (
                            (
                                torch.sum(
                                    torch.logsumexp(
                                        torch.cat(
                                            [
                                                -pos_logits / self.contra_tau,
                                                torch.zeros(*pos_logits.shape[: 2], 1, device=device)
                                            ], dim=-1
                                        ),
                                        dim=-1
                                    )[valid_sample_mask]
                                )
                                + torch.sum(
                                    torch.logsumexp(
                                        torch.cat(
                                            [
                                                neg_logits / self.contra_tau,
                                                torch.zeros(*pos_logits.shape[: 2], 1, device=device)
                                            ],
                                            dim=-1
                                        ),
                                        dim=-1
                                    )[valid_sample_mask]
                                )
                            )
                            / (self.conv_times * torch.sum(valid_sample_mask) + 1e-6)
                        )
                    elif self.contra_kind == 'InfoNCE':
                        logits = torch.cat([pos_logits[..., None], neg_logits[..., None, :].expand(-1, -1, self.conv_times - 1, -1)], dim=-1)
                        logits = logits[valid_sample_mask].contiguous().view(-1, self.conv_times + 1)
                        bound_contra_loss = F.cross_entropy(logits, torch.zeros(logits.shape[0]).long().to(device))
                contra_loss = contra_loss + bound_contra_loss
        # Segmentation loss
        if self.sup_on_all:
            dice_loss = 0.
            ce_loss = 0.
            for preds in preds_list:
                dice_loss = dice_loss + one_minus_dice(preds, targets, self.dice_weight.to(images.device))
                ce_loss = ce_loss + F.cross_entropy(preds, targets.long(), self.ce_weight.to(images.device))
        else:
            dice_loss = one_minus_dice(preds_list[0], targets, self.dice_weight.to(images.device))
            ce_loss = F.cross_entropy(preds_list[0], targets.long(), self.ce_weight.to(images.device))
        w_contra = self.beta * (1 + np.sin(self.cur_iter / self.contra_rampup * np.pi - np.pi / 2)) / 2
        total_loss = dice_loss + ce_loss + self.lamb * consist_loss + w_contra * contra_loss
        if self.training:
            self.cur_iter = min(self.cur_iter + 1, self.contra_rampup)
        return (
            ['DiceLoss', 'CELoss', 'Consist', 'W_Contra', 'Contra', 'Total'],
            [
                round(dice_loss.item(), 3), round(ce_loss.item(), 3), round(consist_loss.item(), 3),
                round(w_contra, 3), round(contra_loss.item(), 3),
                total_loss.item()],
            total_loss
        )

    def infer(self, data_dict):
        if self.conv_pre_norm:
            data_dict['images'] = data_dict['images'] - 0.5
        return super().infer(data_dict)
