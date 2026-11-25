"""Launch every study in fresh processes; retain resolved YAML and raw evidence."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import yaml
from train import REPO, load_config, write_csv

RESULTS = REPO / 'lab06/results'


def jobs(base):
    plan = []
    def add(name, **updates):
        c = copy.deepcopy(base)
        c['training'].update(updates)
        plan.append((name, c))
    for workers in [0, 1, 2, 4, 8]:
        add(f'workers_{workers}', num_workers=workers, epochs=2)
    for batch in [32, 64, 128, 256, 512]:
        add(f'batch_{batch}', batch_size=batch)
    for optimizer, lr in [('SGD', .01), ('Adam', .001)]:
        for scheduler in ['StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau']:
            add(f'{optimizer}_{scheduler}', optimizer=optimizer, lr=lr, scheduler=scheduler, epochs=10)
    add('accumulation_64x4', batch_size=64, accumulation=4)
    add('recovery_reference', epochs=5)
    add('recovery_resumed', epochs=5)
    add('profile', epochs=1)
    return plan


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--only', nargs='*')
    args = p.parse_args()
    base = load_config(REPO / 'lab06/config.yaml')
    for name, c in jobs(base):
        if args.only and name not in args.only:
            continue
        config = REPO / f'lab06/configs/{name}.yaml'
        config.write_text(yaml.safe_dump(c, sort_keys=False))
        out = RESULTS / 'runs' / name
        if (out / 'summary.json').exists():
            print('Already complete:', name, flush=True)
            continue
        cmd = [sys.executable, str(REPO / 'lab06/src/train.py'), '--config', str(config), '--name', name]
        print('START', name, flush=True)
        with (RESULTS / f'{name}.log').open('a') as log:
            if name == 'recovery_resumed':
                # The first process actually terminates after epoch 3; a new process restores state.
                if not (out / 'last.pt').exists():
                    subprocess.run(cmd + ['--stop-after', '3'], cwd=REPO, stdout=log, stderr=log, check=True)
                cmd += ['--resume', str(out / 'last.pt')]
            if name == 'profile':
                cmd += ['--profile']
            subprocess.run(cmd, cwd=REPO, stdout=log, stderr=log, check=True)
        print('DONE', name, flush=True)
    rows = []
    for path in sorted((RESULTS / 'runs').glob('*/summary.json')):
        rows.append(json.loads(path.read_text()))
    write_csv(RESULTS / 'study_summary.csv', rows)


if __name__ == '__main__':
    main()
