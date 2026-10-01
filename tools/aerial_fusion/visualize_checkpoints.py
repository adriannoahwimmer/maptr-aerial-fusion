"""Run tools/maptr/vis_pred.py for several checkpoints and score thresholds.

Visualizes every n-th epoch and the last checkpoint of a training run. The
results are written to <checkpoint_dir>/vis_pred/epoch_<n>_visuals.

Use a config whose test set only contains the scenes you want to look at,
e.g. an info file created with tools/aerial_fusion/subset_infos.py.

Example:
    python tools/aerial_fusion/visualize_checkpoints.py \
        work_dirs/maptr_tiny_r50_24e_aerial_cwl \
        projects/configs/maptr_aerial/maptr_tiny_r50_24e_aerial_cwl.py \
        --every 4 --thresholds 0.0 0.2 0.3 0.4
"""
import argparse
import os
import subprocess
import sys

VIS_PRED = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'maptr', 'vis_pred.py')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('checkpoint_dir', help='work dir with epoch_<n>.pth files')
    parser.add_argument('config', help='config used for the visualization')
    parser.add_argument('--every', type=int, default=24, help='visualize every n-th epoch')
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.0, 0.2, 0.3, 0.4],
                        help='score thresholds passed to vis_pred.py')
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoints = [f for f in os.listdir(args.checkpoint_dir)
                   if f.startswith('epoch_') and f.endswith('.pth')]
    checkpoints.sort(key=lambda f: int(f[len('epoch_'):-len('.pth')]))

    for idx, checkpoint in enumerate(checkpoints):
        epoch = int(checkpoint[len('epoch_'):-len('.pth')])
        if epoch % args.every != 0 and idx != len(checkpoints) - 1:
            continue
        show_dir = os.path.join(args.checkpoint_dir, 'vis_pred', f'epoch_{epoch}_visuals')
        os.makedirs(show_dir, exist_ok=True)
        for thresh in args.thresholds:
            cmd = [sys.executable, VIS_PRED, args.config,
                   os.path.join(args.checkpoint_dir, checkpoint),
                   '--score-thresh', str(thresh), '--show-dir', show_dir]
            print('Running:', ' '.join(cmd))
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f'vis_pred.py failed for epoch {epoch}, threshold {thresh}: {e}')


if __name__ == '__main__':
    main()
