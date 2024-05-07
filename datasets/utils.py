import numpy as np
import SimpleITK as sitk
from skimage import measure
from skimage.transform import resize
from batchgenerators.augmentations.utils import resize_segmentation
try:
    from multiprocessing import shared_memory
except:
    print('Shared memory is unavailable in this python version')


class MemoryReader:

    def __init__(self, meta_file):
        '''
        A reader for shared memory. To load a dataset in the shared memory, please refer to '/home/ics/Documents/dataset_server'
        Params:
            meta_file: str, path to the dataset file in the shared memory
        '''
        meta_info = np.load(meta_file, allow_pickle=True).tolist()
        self.images_shm = shared_memory.SharedMemory(name=meta_info['images_shm_name'])
        self.images = np.ndarray(meta_info['images_nelmts'], dtype=meta_info['images_dtype'], buffer=self.images_shm.buf)
        self.labels_shm = shared_memory.SharedMemory(name=meta_info['labels_shm_name'])
        self.labels = np.ndarray(meta_info['labels_nelmts'], dtype=meta_info['labels_dtype'], buffer=self.labels_shm.buf)
        #
        image_paths = meta_info['image_paths']
        self.path2index = dict([(image_path, index) for index, image_path in enumerate(image_paths)])
        self.mem_indexs = meta_info['indexs']  # indexs.shape = (#images + 1, )
        self.shapes = meta_info['shapes']  # shapes.shape = (#images, 3)
        del meta_info

    def __getitem__(self, index):
        left_mem_index = self.mem_indexs[index]
        right_mem_index = self.mem_indexs[index + 1]
        shape = self.shapes[index]
        image = self.images[left_mem_index : right_mem_index].copy().reshape(shape)
        label = self.labels[left_mem_index : right_mem_index].copy().reshape(shape)
        return image, label

    def get_data_by_path(self, path):
        return self.__getitem__(self.path2index[path])

    def close(self, ):
        self.images_shm.close()
        self.labels_shm.close()


# TODO: normalize on masked area only
def normalize(origin_image, data_type):
    eps = 1e-5
    if data_type == 'Array':
        return origin_image
    elif data_type == 'CT':
        assert (np.min(image) > -1000 - eps) and (np.max(image) < 3000 + eps)
        image = np.clip(origin_image, -1000, 3000)
        image = (image + 1000.) / 4000.
    elif data_type == 'MR':
        min_value = np.percentile(origin_image, 1)
        max_value = np.percentile(origin_image, 99)
        image = (np.clip(origin_image, min_value, max_value) - min_value) / (max_value - min_value)
    else:
        raise Exception(f'Unknown data_type: {data_type}')
    assert (np.min(image) >= -eps) and (np.max(image) <= (1 + eps)), f'{data_type} [{np.min(origin_image)}, {np.max(origin_image)}], [{np.min(image)}, {np.max(image)}]'
    return image


def pad_if_need(image, min_size, pad_value=None):
    '''
    image.shape = (c, d, h, w)
    min_size = (d', h', w') or (h', w')
    '''
    *_, h, w = image.shape
    *_, min_h, min_w = min_size
    left_pad_h = max((min_h - h + 1) // 2, 0)
    right_pad_h = max(min_h - h - left_pad_h, 0)
    left_pad_w = max((min_w - w + 1) // 2, 0)
    right_pad_w = max(min_w - w - left_pad_w, 0)
    if len(min_size) == 3:  # 3D
        d = image.shape[1]
        min_d = min_size[0]
        left_pad_d = max((min_d - d + 1) // 2, 0)
        right_pad_d = max(min_d - d - left_pad_d, 0)
    else:
        left_pad_d = 0
        right_pad_d = 0
    pad_params = [
        (0, 0),
        (left_pad_d, right_pad_d),
        (left_pad_h, right_pad_h),
        (left_pad_w, right_pad_w)
    ]
    #
    if pad_value is None:
        pad_value = np.min(image)
    image = np.pad(
        image, pad_params,
        mode='constant', constant_values=pad_value,
    )
    return image


def get_mask_from_label(label, mask_type):
    '''
    Params:
        label.shape = (d, h, w), unique_values = [0, 1, ...]
        mask_type in ['all', 'valid_slice', 'valid_bbox']
    Return:
        mask.shape = (d, h, w)
    '''
    if mask_type == 'all':
        mask = np.ones_like(label, dtype=label.dtype)
    elif mask_type == 'valid_slice':
        slice_mask = (np.sum(label, axis=(1, 2)) > 0)
        mask = np.zeros_like(label, dtype=label.dtype)
        mask[slice_mask] = 1
    elif mask_type == 'valid_bbox':
        mask = np.zeros_like(label, dtype=label.dtype)
        num_classes = np.max(label)
        for class_ind in range(1, num_classes + 1):
            regions = measure.regionprops(measure.label(label == class_ind))
            for region in regions:
                zmin, ymin, xmin, zmax, ymax, xmax = region.bbox
                mask[zmin : zmax, ymin : ymax, xmin : xmax] = 1
    else:
        raise Exception('Unknown mask_type:', mask_type)
    return mask


def resample_image(itk_image, out_spacing=None, out_size=None, is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    if (
        ((out_spacing is None) and (out_size is None))
        or ((out_spacing is not None) and (tuple(out_spacing) == original_spacing))
        or ((out_size is not None) and (tuple(out_size) == original_size))
    ):
        return itk_image
    if out_size is None:
        for axis in range(3):
            if out_spacing[axis] == -1:
                out_spacing[axis] = original_spacing[axis]
        out_size = [
            round(original_size[0] * (original_spacing[0] / out_spacing[0])),
            round(original_size[1] * (original_spacing[1] / out_spacing[1])),
            round(original_size[2] * (original_spacing[2] / out_spacing[2]))
        ]
    else:
        assert out_spacing is None
        out_spacing = [
            original_size[0] * original_spacing[0] / out_size[0],
            original_size[1] * original_spacing[1] / out_size[1],
            original_size[2] * original_spacing[2] / out_size[2],
        ]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    if is_label:
        itk_image = sitk.Cast(itk_image, sitk.sitkUInt8)
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
        resample.SetDefaultPixelValue(0)
    else:
        itk_image = sitk.Cast(itk_image, sitk.sitkFloat32)
        resample.SetInterpolator(sitk.sitkLinear)
        resample.SetDefaultPixelValue(float(np.min(sitk.GetArrayFromImage(itk_image))))
    return resample.Execute(itk_image)  # Not an inplace operation


def pad_for_crop(image, label, random_crop_shape, center_crop_shape):
    '''
    Params:
        image: np.array(c,  d, h, w)
        label: np.array(c', d, h, w)  # commonly, c' == 2 for 'segmentation' and 'mask'
        random_crop_shape: (r_crop_d, r_crop_h, r_crop_w)
        center_crop_shape: (c_crop_d, c_crop_h, c_crop_w)
    Return:
        image: np.array(c,  d', h', w')
        label: np.array(c', d', h', w')
    '''
    pad_shape = np.max(
        np.concatenate(
            [
                np.array(random_crop_shape)[None],
                (
                    np.array(image.shape)[1 :]
                    + np.array(random_crop_shape)
                    - np.array(center_crop_shape)
                )[None]
            ], axis=0
        ), axis=0
    )
    image = pad_if_need(image, pad_shape)
    seg = pad_if_need(label[: 1], pad_shape, pad_value=0)
    mask = pad_if_need(label[1 : 2], pad_shape, pad_value=1)
    fg_mask = pad_if_need(label[2 :], pad_shape, pad_value=0)
    label = np.concatenate([seg, mask, fg_mask], axis=0)
    return image, label


def random_crop(image, label, crop_shape):
    '''
    Params:
        image: np.array(c, d, h, w)
        label: np.array(c', d, h, w)  # commonly, c' == 2 for 'segmentation' and 'mask'
        crop_shape: (crop_d, crop_h, crop_w) or (crop_h, crop_w)
    Return:
        croped_image: np.array(crop_d, crop_h, crop_w)
        croped_label: np.array(crop_d, crop_h, crop_w)
    '''
    _, d, h, w = image.shape
    crop_d, crop_h, crop_w = crop_shape
    offset_z = np.random.randint(0, d - crop_d + 1)
    offset_y = np.random.randint(0, h - crop_h + 1)
    offset_x = np.random.randint(0, w - crop_w + 1)
    #
    image = image[
        :,
        offset_z : offset_z + crop_d,
        offset_y : offset_y + crop_h,
        offset_x : offset_x + crop_w,
    ].copy()
    label = label[
        :,
        offset_z : offset_z + crop_d,
        offset_y : offset_y + crop_h,
        offset_x : offset_x + crop_w,
    ].copy()
    return image, label


def center_crop(image, label, crop_shape):
    '''
    Params:
        image: np.array(c, d, h, w)
        label: np.array(c', d, h, w)  # commonly, c' == 2 for 'segmentation' and 'mask'
        crop_shape: (crop_d, crop_h, crop_w)
    Return:
        croped_image: np.array(crop_d, crop_h, crop_w)
        croped_label: np.array(crop_d, crop_h, crop_w)
    '''
    _, d, h, w = image.shape
    offset_z = (d - crop_shape[0]) // 2
    offset_y = (h - crop_shape[1]) // 2
    offset_x = (w - crop_shape[2]) // 2
    image = image[:, offset_z : d - offset_z, offset_y : h - offset_y, offset_x : w - offset_x]
    image = np.ascontiguousarray(image)
    label = label[:, offset_z : d - offset_z, offset_y : h - offset_y, offset_x : w - offset_x]
    label = np.ascontiguousarray(label)
    return image, label