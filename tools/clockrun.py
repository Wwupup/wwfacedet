import argparse
import os
from time import sleep


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert wwfacedet models to libfacedetect dnn data')
    parser.add_argument('cur_config', help='current config file path')
    parser.add_argument('target_config', help='target config file path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    root = 'work_dirs/'
    assert os.path.exists(root)
    target_dir = os.path.join(
        root,
        os.path.basename(args.cur_config).replace('.py', ''))
    assert os.path.exists(target_dir)
    target = os.path.join(target_dir, 'epoch_640.pth')
    print(f'Waiting for {target} ...')
    while (not os.path.exists(target)):
        sleep(3600)
    print('OK!')
    os.system(f'bash ./tools/dist_train.sh {args.target_config} 2 12347')
