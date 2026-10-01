"""Remove a path prefix from the camera and sweep paths of an info file (.pkl).

Useful when the info files were created on another machine, e.g. with the
nuScenes data under /datasets/nuScenes instead of data/nuscenes.

Example:
    python tools/aerial_fusion/strip_info_paths.py \
        data/nuscenes/nuscenes_infos_temporal_train.pkl \
        --old /datasets/nuScenes/ --new data/nuscenes/
"""
import argparse
import os
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('src', help='info file to read')
    parser.add_argument('--old', required=True, help='path prefix to replace')
    parser.add_argument('--new', default='', help='new path prefix')
    parser.add_argument('--out', help='output file, default: <src>_paths.pkl')
    return parser.parse_args()


def replace_prefix(path, old, new):
    return new + path[len(old):] if path.startswith(old) else path


def main():
    args = parse_args()
    with open(args.src, 'rb') as f:
        data = pickle.load(f)

    for info in data.get('infos', []):
        for key in ('lidar_path',):
            if key in info:
                info[key] = replace_prefix(info[key], args.old, args.new)
        for sweep in info.get('sweeps', []):
            sweep['data_path'] = replace_prefix(sweep['data_path'], args.old, args.new)
        for cam in info.get('cams', {}).values():
            cam['data_path'] = replace_prefix(cam['data_path'], args.old, args.new)

    out = args.out or os.path.splitext(args.src)[0] + '_paths.pkl'
    with open(out, 'wb') as f:
        pickle.dump(data, f)
    print(f'saved to {out}')


if __name__ == '__main__':
    main()
