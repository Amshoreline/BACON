import os
import torch
import numpy as np
import SimpleITK as sitk
from tqdm import trange

from .utils import (
    normalize, get_mask_from_label, resample_image,
    pad_for_crop, random_crop, center_crop
)
from .aug_from_nnUNet import (
    get_spatial_trans as get_spatial_trans_nnUNet,
    get_voxel_trans as get_voxel_trans_nnUNet
)
from .aug_from_Jo import (
    get_spatial_trans as get_spatial_trans_Jo,
    get_voxel_trans as get_voxel_trans_Jo
)


class VanillaDataset(torch.utils.data.Dataset):

    def __init__(
            self, txt_dir, txt_files, data_type, phase,
            dataset_size=-1, data_in_memory=True,
            out_spacing=None, out_shape=(64, 64, 64), num_modals=1,
            spatial_aug_kind='nnUNet', spatial_aug_params={},
            voxel_aug_kind='nnUNet', voxel_aug_params={},
            mask_type='all', sample_by_slice=False,
        ):
        '''
        Parmas:
            txt_dir: directory of the text files
            txt_files: [[weight(int), filename], ...]
            data_type: 'MRI' or 'CT'
            phase: 'train' or 'test'
            #
            dataset_size: -1, 1, 2, 3, ...
            data_in_memory: True or False, whether to keep data in the memory
            #
            out_spacing: None or (*w'*, *h'*, *d'*), physical size of a voxel
            out_shape: (d, h, w) or (h, w)
            num_modals: number of modalities
            #
            dummy_2D_aug: whether to use dummy 2D augmentation,
                this param is for 3d out_shape only
            data_aug_params: a dict for DIY data augmentations
            mask_type: 'all' or 'valid_slice', 'valid_slice' will indicate which slice is labeled
            #
            sample_by_slice: whether to sample by slice
                this param is for 2d out_shape and 'train' phase only
        '''
        # Get data paths
        data_paths = []
        for times, filename in txt_files:
            with open(txt_dir + '/' + filename, 'r') as reader:
                data_paths.extend(times * reader.read().strip().split('\n'))
        if dataset_size < 0:
            dataset_size = len(data_paths)
        self.dataset_size = dataset_size
        self.data_paths = [item.split(',') for item in data_paths]
        #
        self.data_type = data_type
        # Attributes configuration
        assert phase in ['train', 'eval']
        self.phase = phase
        self.is_train = (phase == 'train')
        #
        assert data_in_memory in [True, False]
        self.data_in_memory = data_in_memory
        if data_in_memory:
            print('Load data into memory at runtime')
            self.path2data = {}
        else:
            print('No extra operation between for memory')
        #
        self.out_spacing = out_spacing
        self.index2ori_itk_label = None
        self.out_shape = np.array(out_shape).astype(np.int)
        self.threeD = (len(out_shape) == 3)
        self.num_modals = num_modals
        # Augmentations (spatial)
        if spatial_aug_kind == 'nnUNet':
            spatial_aug_params['patch_size'] = self.out_shape
            spatial_aug_params['threeD'] = self.threeD
            spatial_trans, spatial_aug_params = get_spatial_trans_nnUNet(spatial_aug_params)
            scale_range = spatial_aug_params['scale_range']
        else:
            spatial_trans, spatial_aug_params = get_spatial_trans_Jo(spatial_aug_params)
            scale_range = spatial_aug_params['affine']['scale']
        self.spatial_trans = spatial_trans
        print('Spatial augmentations:', spatial_aug_kind, spatial_aug_params)
        # Augmentations (voxel)
        if voxel_aug_kind == 'nnUNet':
            voxel_aug_params['threeD'] = self.threeD
            voxel_trans, voxel_aug_params = get_voxel_trans_nnUNet(voxel_aug_params)
        else:
            voxel_trans, voxel_aug_params = get_voxel_trans_Jo(voxel_aug_params)
        self.voxel_trans = voxel_trans
        print('Voxel augmentations:', voxel_aug_kind, voxel_aug_params)
        # Get random crop shape (crop_shape) and center crop shape (out_shape)
        self.crop_shape = np.round(self.out_shape * max(scale_range)).astype(int)
        if not self.threeD:
            self.out_shape = (1, *self.out_shape)
            self.crop_shape = (1, *self.crop_shape)
        else:
            self.crop_shape[0] = self.out_shape[0]
        print(f'Crop shape is {self.crop_shape}, Output shape is {self.out_shape}')
        # Mask
        assert mask_type in ['disk', 'all', 'valid_slice']
        self.mask_type = mask_type
        # Sample by slice or image
        assert sample_by_slice in [True, False]
        self.sample_by_slice = sample_by_slice
        if not self.threeD and self.is_train and self.sample_by_slice:
            print('Enumerating the whole dataset')
            image_indexs = []
            slice_indexs = []
            for index in trange(self.dataset_size):
                image, *_ = self._read_image_label_once(index)
                image_indexs.extend([index] * image.shape[1])
                slice_indexs.extend(list(range(image.shape[1])))
            self.image_indexs = image_indexs
            self.slice_indexs = slice_indexs
            self.slice_dataset_size = len(self.image_indexs)

    def _read_image_label_once(self, index):
        '''
        Desc:
            Read image and label
        Return:
            image: (c, d, h, w)
            label: (2, d, h, w), 2->seg & mask
            name: str
        '''
        image_path = self.data_paths[index][0]
        name = os.path.basename(image_path).replace('.nii.gz', '')
        if self.data_in_memory and image_path in self.path2data:
            # Case 1: from memory
            image, label = self.path2data[image_path]
        else:
            # Case 2: from disk
            # image
            image = []
            for modal_index in range(self.num_modals):
                itk_image = sitk.ReadImage(self.data_paths[index][modal_index])
                itk_image = resample_image(itk_image, out_spacing=self.out_spacing, is_label=False)
                arr_image = sitk.GetArrayFromImage(itk_image).astype(np.float32)
                arr_image = normalize(arr_image, self.data_type)
                image.append(arr_image[None])
            image = np.concatenate(image, axis=0)  # (c, d, h, w)
            # label
            if len(self.data_paths[index]) > self.num_modals:
                # From disk
                itk_label = sitk.ReadImage(self.data_paths[index][self.num_modals])
                itk_label = resample_image(itk_label, out_spacing=self.out_spacing, is_label=True)
                arr_label = sitk.GetArrayFromImage(itk_label).astype(np.uint8)  # (d, h, w)
            else:
                # There is no label
                arr_label = np.zeros_like(image[0], dtype=np.uint8)  # (d, h, w)
            # mask
            if self.mask_type == 'disk':
                # From disk
                assert len(self.data_paths[index]) == (self.num_modals + 2)
                itk_mask = sitk.ReadImage(self.data_paths[index][self.num_modals + 1])
                itk_mask = resample_image(itk_mask, out_spacing=self.out_spacing, is_label=True)
                arr_mask = sitk.GetArrayFromImage(itk_mask).astype(np.uint8)  # (d, h, w)
            else:
                # From real-time computation
                arr_mask = get_mask_from_label(arr_label, self.mask_type)  # (d, h, w)
            # # fg_mask
            # fg_mask = (image > 0)  # (c, d, h, w)
            # label += mask
            # label = np.concatenate([arr_label[None], arr_mask[None], fg_mask], axis=0)
            label = np.concatenate([arr_label[None], arr_mask[None]], axis=0)
            # Restore image, label and mask into the memory
            if self.data_in_memory:
                self.path2data[image_path] = (image, label)
        return image, label, name

    def _read_image_label(self, index):
        if self.is_train:
            if (not self.threeD) and self.sample_by_slice:
                image, label, name = self._read_image_label_once(self.image_indexs[index])
                offset_z = self.slice_indexs[index]
                image = image[:, offset_z : offset_z + 1]
                label = label[:, offset_z : offset_z + 1]
            else:
                image, label, name = self._read_image_label_once(index)
            image, label = pad_for_crop(image, label, self.crop_shape, self.out_shape)
            image, label = random_crop(image, label, self.crop_shape)
            spatial_aug_sample = self.spatial_trans(data=image[None], seg=label[None])
            voxel_aug_sample = self.voxel_trans(data=spatial_aug_sample['data'], seg=spatial_aug_sample['seg'])
            image, label = voxel_aug_sample['data'][0], voxel_aug_sample['seg'][0]
            image, label = center_crop(image, label, self.out_shape)
        else:
            # test
            image, label, name = self._read_image_label_once(index)
        return image, label, name

    def __len__(self):
        if (not self.threeD) and self.is_train and self.sample_by_slice:
            return self.slice_dataset_size  # = len(image_0) + len(image_1) + ...
        else:
            return self.dataset_size

    def __getitem__(self, index):
        image, label, name = self._read_image_label(index)
        return {
            'images': image,     # (c, d, h, w)
            'indexs': index,
            'labels': label[0],  # (d, h, w)
            'masks': label[1][None], # (1, d, h, w)
            # 'fg_masks': label[2 :],  # (c, d, h, w)
            'names': name,
        }

    def restore_spacing(self, pred, index):
        if self.index2ori_itk_label is None:
            index2ori_itk_label = {}
            for ind, data_path in enumerate(self.data_paths):
                if len(data_path) > self.num_modals:
                    itk_label = sitk.ReadImage(data_path[self.num_modals])
                else:
                    itk_image = sitk.ReadImage(data_path[0])
                    itk_label = sitk.GetImageFromArray(
                        np.zeros_like(sitk.GetArrayFromImage(itk_image), dtype=np.uint8)
                    )
                    itk_label.CopyInformation(itk_image)
                index2ori_itk_label[ind] = itk_label
                print(
                    f'Getting original itk label [{ind}/{len(self.data_paths)}] '
                    f'with size {itk_label.GetSize()} spacing {itk_label.GetSpacing()}'
                )
            self.index2ori_itk_label = index2ori_itk_label
        #
        ori_itk_label = self.index2ori_itk_label[index]
        ori_size = ori_itk_label.GetSize()
        ori_spacing = ori_itk_label.GetSpacing()
        #
        itk_pred = sitk.GetImageFromArray(pred)
        itk_pred = resample_image(itk_pred, out_size=ori_size, is_label=True)
        itk_pred.SetSpacing(ori_spacing)
        return itk_pred