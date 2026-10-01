"""Visualize the learned weight tensor of a CWLFuser checkpoint as a heatmap.

The weights are averaged over the channels, so the heatmap shows the mean
influence of the BEV features (0 = only aerial, 1 = only BEV) per BEV cell
(thesis, Abb. 6.1).

Example:
    python tools/aerial_fusion/visualize_fusion_weights.py \
        work_dirs/maptr_tiny_r50_24e_aerial_cwl/epoch_24.pth
"""
import argparse
import os

import matplotlib.pyplot as plt
import torch

WEIGHT_KEY = 'pts_bbox_head.transformer.fuser.fusion_weight_tensor'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('checkpoint', help='checkpoint file (.pth)')
    parser.add_argument('--out', help='output image, default: <run>_<epoch>_heatmap.png')
    parser.add_argument('--vmax', type=float, default=0.6, help='upper limit of the color scale')
    parser.add_argument('--save-tensors', action='store_true',
                        help='also save the full and the averaged weights as .pt files')
    parser.add_argument('--show', action='store_true', help='show the plot')
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    if WEIGHT_KEY not in state_dict:
        raise SystemExit(f'{WEIGHT_KEY} not found, is this a CWLFuser checkpoint?')

    weights = state_dict[WEIGHT_KEY]  # (channels, width, length)
    fusion_weight_avg = weights.mean(dim=0)
    print(f'fusion weights: {tuple(weights.shape)}, '
          f'mean {fusion_weight_avg.mean():.4f}, '
          f'min {fusion_weight_avg.min():.4f}, max {fusion_weight_avg.max():.4f}')

    run_name = os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint)))
    epoch = os.path.splitext(os.path.basename(args.checkpoint))[0]
    out = args.out or f'{run_name}_{epoch}_heatmap.png'

    if args.save_tensors:
        torch.save(weights, os.path.splitext(out)[0] + '_weights.pt')
        torch.save(fusion_weight_avg, os.path.splitext(out)[0] + '_weights_avg.pt')

    plt.figure(figsize=(10, 20))
    heatmap = plt.imshow(fusion_weight_avg.numpy(), cmap='viridis', vmin=0, vmax=args.vmax)
    cbar = plt.colorbar(heatmap, ticks=[0, args.vmax / 2, args.vmax], label='Weight Value')
    cbar.ax.tick_params(labelsize=32)
    plt.xlabel('Width', fontsize=24)
    plt.ylabel('Height', fontsize=24)
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f'saved heatmap to {out}')
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
