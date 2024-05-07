import sys
import os
import random


def main(cfg, gpu_id):
    num_gpus = len(gpu_id.split(','))
    print(f'Use config {cfg}, GPU {gpu_id}')
    # Prepare command
    cfg_file = os.path.basename(cfg)
    tag = cfg_file.replace('.yaml', '')
    if cfg.endswith('test.yaml'):
        test = '--test'
    else:
        test = ''
    os.makedirs('log', exist_ok=True)
    #
    if num_gpus == 1:
        module = ' -u '
    else:
        port = random.randint(20000, 30000)
        module = (
            '-m torch.distributed.launch '
            f'--nproc_per_node={num_gpus} --master_port={port} '
        )
    command = (
        'now=$(date +"%Y%m%d_%H%M%S")\n'
        f'CUDA_VISIBLE_DEVICES={gpu_id} '
        'python '
        f' {module} '
        f'train_eval.py --config_file {cfg} {test} '
        f'2>&1 | tee log/{cfg_file[: -5]}.log-$now'
    )
    print(command)
    os.system(command)


if __name__ == '__main__':
    # User defined arguments
    gpu_id = sys.argv[1]
    cfgs = sys.argv[2 :]
    for cfg in cfgs:
        main(cfg, gpu_id)