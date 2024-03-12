# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

from mmengine import Config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert benchmark test model list to script')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--port', type=int, default=28171, help='dist port')
    parser.add_argument(
        '--work-dir',
        default='work_dirs/benchmark_evaluation',
        help='the dir to save metric')
    parser.add_argument(
        '--out',
        type=str,
        default='.dev_scripts/benchmark_evaluation.sh',
        help='path to save model benchmark script')

    args = parser.parse_args()
    return args


def process_model_info(model_info, work_dir):
    config = model_info['config'].strip()
    fname, _ = osp.splitext(osp.basename(config))
    job_name = fname
    checkpoint = model_info['checkpoint'].strip()
    work_dir = osp.join(work_dir, fname)
    if not isinstance(model_info['eval'], list):
        evals = [model_info['eval']]
    else:
        evals = model_info['eval']
    eval = ' '.join(evals)
    return dict(
        config=config,
        job_name=job_name,
        checkpoint=checkpoint,
        work_dir=work_dir,
        eval=eval)


def create_test_bash_info(commands, model_test_dict, port, script_name,
                          partition):
    config = model_test_dict['config']
    job_name = model_test_dict['job_name']
    checkpoint = model_test_dict['checkpoint']
    work_dir = model_test_dict['work_dir']
    eval = model_test_dict['eval']

    echo_info = f'\necho \'{config}\' &'
    commands.append(echo_info)
    commands.append('\n')

    command_info = f'GPUS=4  GPUS_PER_NODE=4  ' \
                   f'CPUS_PER_TASK=2 {script_name} '

    command_info += f'{partition} '
    command_info += f'{job_name} '
    command_info += f'{config} '
    command_info += f'$CHECKPOINT_DIR/{checkpoint} '

    command_info += f'--eval {eval} '
    command_info += f'--work-dir {work_dir} '
    command_info += f'--cfg-options dist_params.port={port} '
    command_info += '&'

    commands.append(command_info)


def main():
    args = parse_args()
    if args.out:
        out_suffix = args.out.split('.')[-1]
        assert args.out.endswith('.sh'), \
            f'Expected out file path suffix is .sh, but get .{out_suffix}'

    commands = []
    partition_name = 'PARTITION=$1'
    commands.append(partition_name)
    commands.append('\n')

    checkpoint_root = 'CHECKPOINT_DIR=$2'
    commands.append(checkpoint_root)
    commands.append('\n')

    script_name = osp.join('tools', 'slurm_test.sh')
    port = args.port
    work_dir = args.work_dir

    cfg = Config.fromfile(args.config)

    for model_key in cfg:
        model_infos = cfg[model_key]
        if not isinstance(model_infos, list):
            model_infos = [model_infos]
        for model_info in model_infos:
            print('processing: ', model_info['config'])
            model_test_dict = process_model_info(model_info, work_dir)
            create_test_bash_info(commands, model_test_dict, port, script_name,
                                  '$PARTITION')
            port += 1

    command_str = ''.join(commands)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(command_str + '\n')


if __name__ == '__main__':
    main()
