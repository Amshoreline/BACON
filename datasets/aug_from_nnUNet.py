import numpy as np
from skimage import measure
from copy import deepcopy
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.local_transforms import BrightnessGradientAdditiveTransform, LocalGammaTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, ContrastAugmentationTransform, BrightnessTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform, MedianFilterTransform, BlankRectangleTransform, SharpeningTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.augmentations.utils import rotate_coords_3d, rotate_coords_2d


# Default spatial augmentation parameters
default_3D_spatial_aug_params = {
    'dummy_2D': False,
    'border_mode_data': 'constant',
    #
    'do_elastic': False,
    'elastic_deform_alpha': (0., 900.),
    'elastic_deform_sigma': (9., 13.),
    'p_eldef': 0.2,
    #
    'do_scaling': True,
    'scale_range': (0.7, 1.4),
    'independent_scale_factor_for_each_axis': False,
    'p_independent_scale_per_axis': 1,
    'p_scale': 0.2,
    #
    'do_rotation': True,
    'rotation_x': (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    'rotation_y': (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    'rotation_z': (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    'rotation_p_per_axis': 1,
    'p_rot': 0.2,
    #
    'do_mirror': False,
    'mirror_axes': (0, 1, 2),
}
default_2D_spatial_aug_params = deepcopy(default_3D_spatial_aug_params)
default_2D_spatial_aug_params['dummy_2D'] = True
default_2D_spatial_aug_params['elastic_deform_alpha'] = (0., 200.)
default_2D_spatial_aug_params['elastic_deform_sigma'] = (9., 13.)
default_2D_spatial_aug_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)

# Default voxel augmentation parameters
default_3D_voxel_aug_params = {
    'noise_var': (0, 0.1),
    'p_noise': 0.1,
    #
    'blur_sigma': (0.5, 1.),
    'p_blur': 0.2,
    'p_blur_per_channel': 0.5,
    #
    'lowres_range': (0.5, 1),
    'p_lowres': 0.25,
    'p_lowres_per_channel': 0.5,
    'lowres_ignore_axes': None,
    #
    'gamma_retain_stats': True,
    'gamma_range': (0.7, 1.5),
    'p_gamma': 0.3,
    #
    'bright_range': (0.75, 1.25),
    'p_bright': 0.15,
    #
    'do_add_bright': False,
    'add_bright_p_per_sample': 0.15,
    'add_bright_p_per_channel': 0.5,
    'add_bright_mu': 0.0,
    'add_bright_sigma': 0.1,
}
default_2D_voxel_aug_params = deepcopy(default_3D_voxel_aug_params)
default_2D_voxel_aug_params['lowres_ignore_axes'] = (0, )


def convert_3d_to_2d_generator(data_dict):
    shp = data_dict['data'].shape
    data_dict['data'] = data_dict['data'].reshape((shp[0], shp[1] * shp[2], shp[3], shp[4]))
    data_dict['orig_shape_data'] = shp
    shp = data_dict['seg'].shape
    data_dict['seg'] = data_dict['seg'].reshape((shp[0], shp[1] * shp[2], shp[3], shp[4]))
    data_dict['orig_shape_seg'] = shp
    return data_dict


def convert_2d_to_3d_generator(data_dict):
    shp = data_dict['orig_shape_data']
    current_shape = data_dict['data'].shape
    data_dict['data'] = data_dict['data'].reshape((shp[0], shp[1], shp[2], current_shape[-2], current_shape[-1]))
    shp = data_dict['orig_shape_seg']
    current_shape_seg = data_dict['seg'].shape
    data_dict['seg'] = data_dict['seg'].reshape((shp[0], shp[1], shp[2], current_shape_seg[-2], current_shape_seg[-1]))
    return data_dict


class AbstractTransform(object):
    
    def __call__(self, **data_dict):
        raise NotImplementedError("Abstract, so implement")

    def __repr__(self):
        ret_str = str(type(self).__name__) + "( " + ", ".join(
            [key + " = " + repr(val) for key, val in self.__dict__.items()]) + " )"
        return ret_str


class Convert3DTo2DTransform(AbstractTransform):

    def __init__(self):
        pass

    def __call__(self, **data_dict):
        return convert_3d_to_2d_generator(data_dict)


class Convert2DTo3DTransform(AbstractTransform):

    def __init__(self):
        pass

    def __call__(self, **data_dict):
        return convert_2d_to_3d_generator(data_dict)


def get_spatial_trans(extra_params):
    '''
    Params:
        patch_size: (d, h, w)
        params: dict
    '''
    if extra_params['threeD']:
        params = default_3D_spatial_aug_params
    else:
        params = default_2D_spatial_aug_params
    params.update(extra_params)
    patch_size = params['patch_size']
    #
    transforms = []
    if params.get('dummy_2D'):
        print('Use dummy_2D augmentation')
        transforms.append(Convert3DTo2DTransform())
        if len(patch_size) == 3:
            patch_size = patch_size[1 :]
    transforms.append(
        SpatialTransform(
            patch_size,
            p_el_per_sample=params.get('p_eldef'), do_elastic_deform=params.get('do_elastic'), alpha=params.get('elastic_deform_alpha'), sigma=params.get('elastic_deform_sigma'),
            p_rot_per_sample=params.get('p_rot'), do_rotation=params.get('do_rotation'), angle_x=params.get('rotation_x'), angle_y=params.get('rotation_y'), angle_z=params.get('rotation_z'), p_rot_per_axis=params.get('rotation_p_per_axis'),
            p_scale_per_sample=params.get('p_scale'), do_scale=params.get('do_scaling'), scale=params.get('scale_range'), independent_scale_for_each_axis=params.get('independent_scale_factor_for_each_axis'),
            border_mode_data=params.get('border_mode_data'), border_cval_data=0, order_data=3, border_mode_seg='constant', border_cval_seg=-1, order_seg=1,
            random_crop=False, patch_center_dist_from_border=None, # We implement random_crop by ourselves
        )
    )
    if params.get('dummy_2D'):
        transforms.append(Convert2DTo3DTransform())
    if params.get('do_mirror'):
        transforms.append(MirrorTransform(params.get('mirror_axes')))
    return Compose(transforms), params


def get_voxel_trans(extra_params):
    if extra_params['threeD']:
        params = default_3D_voxel_aug_params
    else:
        params = default_2D_voxel_aug_params
    params.update(extra_params)
    #
    transforms = []
    # we need to put the color augmentations after the dummy 2d part (if applicable). Otherwise the overloaded color channel gets in the way
    transforms.append(GaussianNoiseTransform(params['noise_var'], p_per_sample=params['p_noise']))
    transforms.append(GaussianBlurTransform(params['blur_sigma'], different_sigma_per_channel=True, p_per_sample=params['p_blur'], p_per_channel=params['p_blur_per_channel']))
    if params.get('do_median'):
        transforms.append(
            MedianFilterTransform(
                (2, 8),
                same_for_each_channel=False,
                p_per_sample=0.2,
                p_per_channel=0.5
            )
        )
    transforms.append(BrightnessMultiplicativeTransform(params['bright_range'], p_per_sample=params['p_bright']))
    if params.get('do_add_bright'):
        transforms.append(
            BrightnessTransform(
                params.get('add_bright_mu'),
                params.get('add_bright_sigma'),
                True,
                p_per_sample=params.get('add_bright_p_per_sample'),
                p_per_channel=params.get('add_bright_p_per_channel')
            )
        )
    transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
    transforms.append(
        SimulateLowResolutionTransform(
            zoom_range=params['lowres_range'], per_channel=True, p_per_channel=params['p_lowres_per_channel'],
            order_downsample=0, order_upsample=3, p_per_sample=params['p_lowres'],
            ignore_axes=params['lowres_ignore_axes']
        )
    )
    transforms.append(
        GammaTransform(
            params.get('gamma_range'), invert_image=True, per_channel=True, retain_stats=params.get('gamma_retain_stats'),
            p_per_sample=0.1
        )
    )
    transforms.append(
        GammaTransform(
            params.get('gamma_range'), invert_image=False, per_channel=True, retain_stats=params.get('gamma_retain_stats'),
            p_per_sample=params['p_gamma']
        )
    )
    return Compose(transforms), params
