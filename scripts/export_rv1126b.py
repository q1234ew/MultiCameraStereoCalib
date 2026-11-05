"""Export an isolated RV1126B calibration candidate; no network or device activation."""

import argparse
from pathlib import Path

from multicamera.io.rv1126b import export_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--device-mac')
    for key in ('pair', 'aux', 'device-id', 'endpoint'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    result = export_bundle(args.source, args.output, pair_name=args.pair, aux_id=args.aux,
                           device_id=args.device_id, endpoint=args.endpoint,
                           device_mac=args.device_mac,
                           adapter_dir=Path(__file__).resolve().parents[1] / 'cpp' / 'rv1126b')
    print(result)


if __name__ == '__main__':
    main()
