import numpy as np
from skimage import measure
from medpy import metric


def post_process(pred_mask):
    fg_mask = (pred_mask > 0)
    pred_label = measure.label(fg_mask)
    regions = measure.regionprops(pred_label)
    areas = [region.area for region in regions]
    pred_mask[pred_label != (np.argmax(areas) + 1)] = 0
    return pred_mask


class Metric:

    def __init__(self, num_fg_classes):
        self.num_fg_classes = num_fg_classes

    def __call__(self, pred_mask, gt_mask):
        # pred_mask: (d, h, w)
        # gt_mask: (d, h, w)
        if self.num_fg_classes == 1:
            res = {'dice': -1, 'jc': -1, 'hd95': -1, 'asd': -1}
            pred_mask = pred_mask.astype(bool)
            # pred_mask = post_process(pred_mask)
            gt_mask = gt_mask.astype(bool)
            try:
                res['dice'] = metric.binary.dc(pred_mask, gt_mask)
                # res['jc'] = metric.binary.jc(pred_mask, gt_mask)
                # res['hd95'] = metric.binary.hd95(pred_mask, gt_mask)
                # res['asd'] = metric.binary.asd(pred_mask, gt_mask)
            except:
                pass
            return res
        else:
            dice_dict = {}
            for class_index in range(self.num_fg_classes):
                class_id = class_index + 1
                cur_pred = (pred_mask == class_id)
                cur_gt = (gt_mask == class_id)
                if np.sum(cur_gt) == 0:
                    continue
                dice = metric.binary.dc(cur_pred, cur_gt)
                dice_dict[class_id] = dice
            return dice_dict


if __name__ == '__main__':
    import SimpleITK as sitk
    from skimage import measure
    metric = Metric(1)
    with open('/home/ics/Documents/Datasets/zcb_data/Atrial/Atrial3D/txts/val.txt', 'r') as reader:
        lines = reader.readlines()
    scores = []
    for ind, line in enumerate(lines):
        gt_mask = sitk.GetArrayFromImage(sitk.ReadImage(line.strip().split(',')[1]))
        pred_mask = sitk.GetArrayFromImage(sitk.ReadImage(f'output/unet_atrial_16_part_test/infer_results/pred_{ind}.nii.gz'))
        pred_label = measure.label(pred_mask)
        regions = measure.regionprops(pred_label)
        areas = [region.area for region in regions]
        print('Ind', ind, 'Region areas: ', areas)
        pred_mask = (pred_label == (np.argmax(areas) + 1))
        score = metric(pred_mask, gt_mask)
        print(score)
        print()
        scores.append(score)
    keys = scores[0].keys()
    avg_score = {key: np.mean([score[key] for score in scores]) for key in keys}
    print(avg_score)
