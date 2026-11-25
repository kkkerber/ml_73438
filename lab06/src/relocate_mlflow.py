"""Relink local MLflow FileStore artifacts after moving this checkout.

Stop the MLflow UI/server first. This only updates artifact-location metadata;
metrics, params, run IDs and artifacts remain unchanged.
"""
import argparse
from pathlib import Path
import yaml

REPO = Path(__file__).resolve().parents[2]


def relocate(store, dry_run=False):
    store = Path(store).resolve()
    changed = 0
    for path in sorted(store.glob('*/meta.yaml')) + sorted(store.glob('*/*/meta.yaml')):
        data = yaml.safe_load(path.read_text())
        if 'artifact_uri' in data:
            key, target = 'artifact_uri', (path.parent / 'artifacts').as_uri()
        elif 'artifact_location' in data:
            key, target = 'artifact_location', path.parent.as_uri()
        else:
            continue
        if data[key] == target:
            continue
        changed += 1
        print(path.relative_to(store), key, '->', target)
        if not dry_run:
            data[key] = target
            temp = path.with_suffix('.tmp')
            temp.write_text(yaml.safe_dump(data, sort_keys=False))
            temp.replace(path)
    print(f'{changed} metadata files would change.' if dry_run else f'{changed} metadata files updated.')
    return changed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store', type=Path, default=REPO / 'lab06/results/mlruns')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    relocate(args.store, args.dry_run)
