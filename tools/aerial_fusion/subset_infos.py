"""Write an info file (.pkl) that only contains selected samples.

Used to visualize a few scenes of the validation set (thesis, Abb. 6.2 - 6.6).
Without --indices the samples are listed and asked for interactively.

Example:
    python tools/aerial_fusion/subset_infos.py \
        data/nuscenes/nuscenes_infos_temporal_val.pkl \
        data/nuscenes/nuscenes_infos_temporal_val_subset.pkl \
        --indices 150,204,320,337,499
"""
import argparse
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('src', help='info file to read')
    parser.add_argument('dst', help='info file to write')
    parser.add_argument('--indices', help='comma separated sample indices to keep')
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.src, 'rb') as f:
        data = pickle.load(f)
    infos = data.get('infos', [])
    if not infos:
        raise SystemExit('no infos found')

    indices = args.indices
    if indices is None:
        for idx, info in enumerate(infos):
            print(f"{idx}: {info.get('lidar_path', 'unknown')}")
        indices = input('Indices of the samples to keep (comma separated): ')
    keep = [int(i) for i in indices.split(',') if i.strip().isdigit()]

    data['infos'] = [infos[i] for i in keep if i < len(infos)]
    with open(args.dst, 'wb') as f:
        pickle.dump(data, f)
    print(f"kept {len(data['infos'])} of {len(infos)} samples, saved to {args.dst}")


if __name__ == '__main__':
    main()
