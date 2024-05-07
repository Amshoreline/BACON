import os, time
import argparse
import json
from addict import Dict
from PIL import Image
import SimpleITK as sitk
import numpy as np
import torch
import torch.distributed as dist    # DDP
from torch.cuda.amp import autocast, GradScaler # Mixed precision
from torch.backends import cudnn    # Reproducibility
from torch.utils.tensorboard import SummaryWriter

import models
import datasets
import metrics
import utils


def run_epoch(
        phase, epoch, configs, data_loader, model, optimizer, scaler,
        lr_schedule, metric, logger, logger_all, recorder, 
    ):
    '''
    Params:
        phase: 'train' or 'eval'
        epoch: int
        configs: Dict
        data_loader: train loader or test loader
        model: UNet, VNet, etc.
        optimizer: Adam, SGD, etc.
        scaler: used for mixed precision
        lr_schedule: [lr_0, lr_1, ..., lr_{n-1}]
        metric: Dice
        logger: a wrapper of print, which can print the local time
        logger_all: similar to the logger
        recorder: tensorboard summary writer
    Return:
        total_scores: {'Avg': avg_score, ...}
    '''
    start_time = time.time()
    total_scores = {}   # Loss infomation (in the training phase) or scores (in the evaluation phase)
    logger.info(f'Epoch {epoch} with {len(data_loader)} iterations')
    for b_ind, data_dict in enumerate(data_loader):
        # data_dict: {'images': image_tensors, 'labels': label_tensors, ...}
        iter_scores = {}
        # Save input demo
        if (phase == 'train') and (epoch == 0) and (b_ind < 2):
            images = data_dict['images'].cpu().numpy()  # (bs, 1, d, h, w)
            images = images.reshape(-1, *images.shape[-2 :])  # (bs * d, h, w)
            images = (images - np.percentile(images, 1)) / (np.percentile(images, 99) - np.percentile(images, 1))
            rgb_images = (np.clip(np.concatenate([images[..., None]] * 3, axis=3), 1e-4, 1 - 1e-4) * 255).astype(np.uint8)
            labels = data_dict['labels'].cpu().numpy()
            labels = labels.reshape(-1, *labels.shape[-2 :])
            rgb_labels = np.concatenate([labels[..., None]] * 3, axis=3).astype(np.uint8)
            rgb_labels[rgb_labels > 0] = 255
            masks = data_dict['masks'].cpu().numpy()
            masks = masks.reshape(-1, *masks.shape[-2 :])
            rgb_masks = np.concatenate([masks[..., None]] * 3, axis=3).astype(np.uint8)
            rgb_masks[rgb_masks > 0] = 255
            for image_ind, (rgb_image, rgb_label, rgb_mask) in enumerate(zip(rgb_images, rgb_labels, rgb_masks)):
                Image.fromarray(np.concatenate([rgb_image, rgb_label, rgb_mask], axis=1)).save(f'{configs.demo_dir}/{b_ind}_{image_ind}.jpg')
        # else:
        #     exit(0)
        # Put the tensor data into gpu
        for key in data_dict.keys():
            if isinstance(data_dict[key], torch.Tensor):
                data_dict[key] = data_dict[key].to(configs.device)
        if phase == 'train':
            # Training phase
            # Update the learning rate
            iteration = epoch * len(data_loader) + b_ind
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_schedule[iteration]
            # Forward and backward
            optimizer.zero_grad()
            if configs.use_fp16:
                with autocast():
                    loss_names, loss_items, total_loss = model(data_dict)
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_names, loss_items, total_loss = model(data_dict)
                total_loss.backward()
                optimizer.step()
            # Post process, e.g., update the teacher model in the mean teacher arch
            if configs.world_size > 1:
                model.module.update()
            else:
                model.update()
            # Record losses
            for loss_name, loss_item in zip(loss_names, loss_items):
                iter_scores[loss_name] = loss_item
        else:
            # Evaluation phase
            # Forward
            with torch.no_grad():
                if configs.world_size > 1:
                    pred_coefs, pred_masks = model.module.infer(data_dict)
                else:
                    pred_coefs, pred_masks = model.infer(data_dict)
            # We expect the batch size to be 1
            assert pred_coefs.shape[0] == 1
            pred_coef = pred_coefs.cpu().numpy()[0].astype(np.float32)  # (d, h, w) or (h, w)
            pred_mask = pred_masks.cpu().numpy()[0].astype(np.uint8)
            index = data_dict['indexs'].item()
            name = data_dict['names'][0]
            '''
            # DEBUG for abdonimal datasets
            gt_mask = data_dict['labels'].cpu().squeeze().numpy()
            valid_slices = (np.sum(gt_mask, axis=(1, 2)) > 0)
            pred_mask, gt_mask = pred_mask[valid_slices], gt_mask[valid_slices]
            iter_scores = metric(pred_mask, gt_mask)
            iter_scores['indexs'] = index
            '''
            # Restore the spacing
            itk_pred_mask = data_loader.dataset.restore_spacing(pred_mask, index)
            # Calculate the scores
            pred_mask = sitk.GetArrayFromImage(itk_pred_mask)
            gt_mask = sitk.GetArrayFromImage(data_loader.dataset.index2ori_itk_label[index])
            iter_scores = metric(pred_mask, gt_mask)
            iter_scores['indexs'] = index
            # Save the prediction
            sitk.WriteImage(itk_pred_mask, f'{configs.infer_dir}/{name}.nii.gz')
        # Merge iter_scores into total_scores
        for key, value in iter_scores.items():
            if not key in total_scores:
                total_scores[key] = []
            total_scores[key].append(value)
        # Check how much cuda memory is allocated
        if configs.device.type == 'cuda':
            memory = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 4)
        else:
            memory = 0.
        # Compute how much time we have cost
        time_cost = time.time() - start_time
        # Print the information of the current iteration
        if ((b_ind + 1) % 10 == 0) or (b_ind == (len(data_loader) - 1)) or (phase != 'train'):
            logger_all.info(
                f'{configs.exp_tag} {phase} Rank {configs.rank}/{configs.world_size} '
                f'Epoch {epoch}/{configs.max_epoch} '
                f'Batch {b_ind}/{len(data_loader)} '
                f'Time {time_cost: .4f} Mem {memory}MB '
                f'LR {optimizer.param_groups[0]["lr"]:.4e} '
                f'{iter_scores}'
            )
    # Collect results among multiple processes and iterations
    if phase == 'train':
        reduce_matrix = torch.zeros(configs.world_size, len(total_scores), len(data_loader)).cuda()
        keys = list(total_scores.keys())
        keys.sort()
        for key_ind, key in enumerate(keys):
            reduce_matrix[configs.rank, key_ind] = torch.tensor(total_scores[key])
        if configs.world_size > 1:
            dist.all_reduce(reduce_matrix)
        reduce_matrix = reduce_matrix.cpu().numpy()
        scores = np.mean(reduce_matrix, axis=(0, 2))  # (#keys, )
        total_scores = list(zip(keys, scores))
        for key, score in total_scores:
            recorder.add_scalar(f'{phase}/{key}(Epoch)', score, epoch)
    else:
        with open(f'{configs.metric_dir}/eval_score_rank_{configs.rank}', 'w') as writer:
            writer.write(str(total_scores))
        if configs.world_size > 1:
            dist.barrier()  # Make sure the write operation in each process is done
        total_scores = {}
        for rank in range(configs.world_size):
            with open(f'{configs.metric_dir}/eval_score_rank_{rank}', 'r') as reader:
                rank_scores = eval(reader.read())
            for key, value in rank_scores.items():
                if not key in total_scores:
                    total_scores[key] = value
                else:
                    total_scores[key].extend(value)
        # Build unique_mask
        data_indexs = total_scores['indexs']
        unique_mask = np.zeros(len(data_indexs), dtype=bool)
        used_data_indexs = set()
        for ind, data_index in enumerate(data_indexs):
            if not data_index in used_data_indexs:
                used_data_indexs.add(data_index)
                unique_mask[ind] = 1
        assert np.all(np.array(sorted(list(used_data_indexs))) == np.arange(len(used_data_indexs)))
        logger_all.info(
            f'Rank {configs.rank}/{configs.world_size} '
            f'Epoch {epoch}/{configs.max_epoch} '
            f'collect {len(used_data_indexs)} unique indexs'
        )
        #
        total_scores = [(key, np.mean(np.array(value)[unique_mask])) for key, value in total_scores.items() if not key == 'indexs']
        total_scores.sort(key=lambda x : x[0])
        total_scores.insert(0, ('Avg', np.mean([item[1] for item in total_scores])))
        for key, value in total_scores:
            recorder.add_scalar(f'Metric/{key}', value, epoch)
    total_scores = dict([(key, round(value, 4)) for key, value in total_scores])
    return total_scores


def main():
    # Get parser
    parser = argparse.ArgumentParser(description='zcb 3d segmentation')
    parser.add_argument('--config_file', default='configs/base.yaml', type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--local_rank', type=int)
    # Get configs
    args = parser.parse_args()
    out_dir = args.config_file.replace('configs', 'output').replace('.yaml', '')
    exp_tag = args.config_file.split('/')[-1].split('.')[0]
    configs = Dict(utils.clear_configs(utils.build_configs(args.config_file, [])))
    configs.is_test = args.test
    # Basic configuration
    if args.local_rank is not None:
        configs.local_rank = utils.dist_init()
        configs.rank = dist.get_rank()
        configs.world_size = dist.get_world_size()
    else:
        configs.local_rank = 0
        configs.rank = 0
        configs.world_size = 1
    # Set seed
    utils.seed_all(configs.seed)
    print(f'Seed {configs.seed}')
    print('Torch version:', torch.__version__)
    # Avoid nondeterministic algorithms
    if configs.determine:
        print('Use deterministic algorithms')    
        cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    # Get logger and logger_all
    logger = utils.Logger()
    logger_all = utils.Logger()
    if configs.rank == 0:
        logger.set_level('Info')
        record_dir = 'summaries'
        if not os.path.exists(record_dir):
            os.mkdir(record_dir)
        record_path = record_dir + '/' + exp_tag
        if args.test:
            recorder = utils.FakeRecorder()
        else:
            recorder = SummaryWriter(log_dir=record_path)
    else:
        # For the other processes, we mute the logger and recorder
        logger.set_level('None')
        recorder = utils.FakeRecorder()
    logger_all.set_level('Info')
    logger.info(f'Recorder is {recorder}')
    # Make saving directories
    configs.out_dir = out_dir
    configs.ckpt_dir = out_dir + '/checkpoints'
    configs.infer_dir = out_dir + '/infer_results'
    configs.metric_dir = out_dir + '/metric'
    configs.demo_dir = out_dir + '/input_demo'
    configs.exp_tag = exp_tag
    if configs.rank == 0:
        os.makedirs(configs.ckpt_dir, exist_ok=True)
        os.makedirs(configs.infer_dir, exist_ok=True)
        os.makedirs(configs.metric_dir, exist_ok=True)
        os.makedirs(configs.demo_dir, exist_ok=True)
    logger.info(f'configs\n{json.dumps(configs, indent=2, ensure_ascii=False)}')
    assert torch.cuda.is_available()
    configs.device = torch.device('cuda')
    logger.info(f'Device is {configs.device}')
    # Get model
    logger.info(f'Creating model: {configs.model.kind}')
    model = models.__dict__[configs.model.kind](**configs.model.kwargs)
    assert isinstance(configs.model.sync_bn, bool) # Synchronize BatchNorm layers
    if configs.world_size > 1 and configs.model.sync_bn:
        logger.info('Use SyncBatchNorm')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda()
    logger.info(f'Use model:\n{model}')
    # Get optimizer
    optimizer = torch.optim.__dict__[configs.trainer.optimizer.kind](
        model.parameters(), **configs.trainer.optimizer.kwargs
    )
    logger.info(f'Use optimizer:\n{optimizer}')
    # Distribute model
    if configs.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[configs.local_rank],
            find_unused_parameters=True
        )
    # Epoch setting
    configs.start_epoch = 0
    # Resume model
    if configs.model.resume and configs.rank == 0:
        logger.info('Resuming model from', configs.model.resume)
        if isinstance(configs.model.resume, list):
            state_dict = dict()
            for ind, resume_path in enumerate(configs.model.resume):
                assert os.path.isfile(resume_path), f'Can not found checkpoint: {resume_path}'
                checkpoint = torch.load(resume_path, map_location='cpu')
                sub_state_dict = checkpoint['state_dict']
                sub_state_dict = dict([(key.replace('unet_body', f'unet_bodies.{ind}'), value) for key, value in sub_state_dict.items()])
                state_dict.update(sub_state_dict)
        else:
            resume_path = configs.model.resume
            assert os.path.isfile(resume_path), f'Can not found checkpoint: {resume_path}'
            checkpoint = torch.load(resume_path, map_location='cpu')
            state_dict = checkpoint['state_dict']
        if configs.world_size > 1:
            state_dict = dict([('module.' + key, value) for key, value in state_dict.items()])
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        #
        # if 'epoch' in checkpoint:
        #     configs.start_epoch = checkpoint['epoch'] + 1
        #     logger.info('Resume epoch')
        # else:
        #     logger.info('WARNING: start_epoch is not resumed')
        #
        # if 'optimizer' in checkpoint:
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        #     logger.info('Resume optimizer')
        # else:
        #     logger.info('WARNING: Optimizer is not resumed')
        #
        logger.info(
            f'Finish resuming from {configs.model.resume}\n\n'
            f'Missing keys {missing_keys}\n\nUnexpected keys {unexpected_keys}'
        )
    if configs.world_size > 1:
        dist.barrier()
    # Mixed precision
    if configs.use_fp16:
        scaler = GradScaler()
        logger.info('Use mixed precision')
    else:
        scaler = None
    # Get metric function
    metric = metrics.Metric(**configs.metric)
    # Build datasets
    if not configs.is_test:
        train_dataset = datasets.__dict__[configs.dataset.kind](
            phase='train', txt_dir=configs.dataset.txt_dir,
            txt_files=configs.dataset.train_txts,
            **configs.dataset.kwargs
        )
    else:
        train_dataset = None
    eval_dataset = datasets.__dict__[configs.dataset.kind](
        phase='eval', txt_dir=configs.dataset.txt_dir,
        txt_files=configs.dataset.eval_txts,
        **configs.dataset.kwargs
    )
    # Build data loaders
    if not configs.is_test:
        if configs.world_size > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        else:
            train_sampler = torch.utils.data.RandomSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, sampler=train_sampler,
            batch_size=configs.dataset.train_batch_size,
            num_workers=configs.dataset.num_workers,
            pin_memory=False, drop_last=True,
            worker_init_fn=utils.seed_worker
        )
        logger_all.info(f'Rank {configs.rank}/{configs.world_size} {len(train_loader)} train iters')
    else:
        train_loader = None
    if configs.world_size > 1:
        eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, shuffle=False)
    else:
        eval_sampler = torch.utils.data.SequentialSampler(eval_dataset)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, sampler=eval_sampler,
        batch_size=configs.dataset.eval_batch_size,
        num_workers=0,
        pin_memory=False, drop_last=False
    )
    logger_all.info(f'Rank {configs.rank}/{configs.world_size} {len(eval_loader)} eval iters')
    # Get lr_scheduler
    if configs.is_test:
        configs.max_epoch = configs.start_epoch + 1
    else:
        lr_configs = configs.trainer.lr_schedule
        lr_schedule = utils.get_lr_schedule(
            len(train_loader) * lr_configs.warmup_epochs,
            len(train_loader) * lr_configs.cosine_epochs,
            lr_configs.cosine_times,
            lr_configs.start_lr, lr_configs.peak_lr, lr_configs.end_lr
        )
        configs.max_epoch = lr_configs.warmup_epochs + lr_configs.cosine_epochs * lr_configs.cosine_times
    # Train/Eval
    for epoch in range(configs.start_epoch, configs.max_epoch):
        if configs.world_size > 1:
            dist.barrier()
        # Train
        if not configs.is_test:
            if configs.world_size > 1:
                train_loader.sampler.set_epoch(epoch)
            model.train()
            train_scores = run_epoch(
                'train', epoch=epoch, configs=configs, data_loader=train_loader,
                model=model, optimizer=optimizer,
                lr_schedule=lr_schedule, metric=metric,
                logger=logger, logger_all=logger_all, recorder=recorder, scaler=scaler
            )
            logger_all.info(f'Train Rank {configs.rank}/{configs.world_size} Epoch {epoch} {train_scores}')
            # Save checkpoint
            if (
                (configs.rank == 0)
                and (
                    ((epoch + 1) % configs.trainer.save_freq == 0)
                    or (epoch == configs.max_epoch - 1)
                )
            ):
                if configs.world_size > 1:
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()
                checkpoint = {
                    # 'epoch': epoch,
                    # 'model_name': configs.model.kind,
                    'state_dict': state_dict,
                    # 'optimizer': optimizer.state_dict(),
                }
                ckpt_path = f'{configs.ckpt_dir}/epoch_{epoch}.pth'
                torch.save(checkpoint, ckpt_path)
                os.system(f'cp {ckpt_path} {configs.ckpt_dir}/latest.pth')
                logger.info(f'Save checkpoint in epoch {epoch}')
        # Evaluate
        if (
            ((epoch + 1) % configs.trainer.test_freq == 0)
            or (epoch == configs.max_epoch - 1)
        ):
            model.eval()
            eval_scores = run_epoch(
                'eval', epoch=epoch, configs=configs, data_loader=eval_loader,
                model=model, optimizer=optimizer,
                lr_schedule=None, metric=metric,
                logger=logger, logger_all=logger_all, recorder=recorder, scaler=None
            )
            logger_all.info(
                f'Eval Rank {configs.rank}/{configs.world_size} Epoch {epoch} {eval_scores}\n'
                f'Latex format: | {" | ".join(np.array(list(eval_scores.values()), dtype=str))} |'
            )
        # To avoid deadlock
        time.sleep(2.33)


if __name__ == '__main__':
    main()
