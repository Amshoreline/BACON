import os


def main():
    exps = os.listdir('output')
    for exp in exps:
        if os.path.exists(f'output/{exp}/checkpoints/'):
            ckpt_names = os.listdir(f'output/{exp}/checkpoints/')
            for ckpt_name in ckpt_names:
                if not ckpt_name == 'latest.pth':
                    os.system(f'rm output/{exp}/checkpoints/{ckpt_name}')
                else:
                    print(exp, 'has latest.pth')


if __name__ == '__main__':
    main()
