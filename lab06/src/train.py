"""Reproducible CPU training, with epoch-boundary recovery and local MLflow."""
import argparse
import copy
import csv
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import random
import threading
import time

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/lab06-matplotlib')
import mlflow
import numpy as np
import psutil
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import yaml

REPO = Path(__file__).resolve().parents[2]


class CNN(nn.Sequential):
    """Exact best Lab 3 architecture; Lab 3 did not persist trained weights."""
    def __init__(self):
        super().__init__(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(32 * 7 * 7, 10))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_config(path):
    with open(path) as f:
        c = yaml.safe_load(f)
    if c['device'] != 'cpu' or c['model'] != 'lab03_cnn':
        raise ValueError('This measured lab implements lab03_cnn on CPU only.')
    t = c['training']
    if min(t['epochs'], t['batch_size'], t['accumulation']) < 1 or t['num_workers'] < 0:
        raise ValueError('Invalid training sizes')
    return c


def make_data(c):
    mean, std = c['data']['normalization']
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean,), (std,))])
    full = datasets.FashionMNIST(REPO / c['data']['root'], train=True, download=False, transform=transform)
    train, val = random_split(full, [c['data']['train_size'], c['data']['val_size']],
                              generator=torch.Generator().manual_seed(c['seed']))
    test = datasets.FashionMNIST(REPO / c['data']['root'], train=False, download=False, transform=transform)
    return train, val, test


def make_loader(dataset, batch, workers, generator=None, shuffle=False):
    return DataLoader(dataset, batch_size=batch, num_workers=workers, shuffle=shuffle,
                      generator=generator, worker_init_fn=seed_worker,
                      persistent_workers=workers > 0)


def make_optimizer(model, t):
    if t['optimizer'] == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=t['lr'])
    if t['optimizer'] == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=t['lr'], momentum=t['momentum'])
    raise ValueError(t['optimizer'])


def make_scheduler(optimizer, t):
    kind = t['scheduler']
    if kind == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, t['step_size'], gamma=t['gamma'])
    if kind == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t['epochs'])
    if kind == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=t['gamma'], patience=t['plateau_patience'])
    raise ValueError(kind)


class PeakRSS:
    """Sample process tree RSS every 10 ms; CPU memory, not CUDA allocations."""
    def __enter__(self):
        self.stop = threading.Event()
        self.peak = 0
        self.process = psutil.Process()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        while not self.stop.is_set():
            total = 0
            for p in [self.process, *[psutil.Process(child.pid) for child in multiprocessing.active_children()]]:
                try:
                    total += p.memory_info().rss
                except psutil.Error:
                    pass
            self.peak = max(self.peak, total)
            self.stop.wait(.01)

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    loss, correct = 0., 0
    for x, y in loader:
        logits = model(x)
        loss += nn.functional.cross_entropy(logits, y, reduction='sum').item()
        correct += (logits.argmax(1) == y).sum().item()
    return loss / len(loader.dataset), correct / len(loader.dataset)


def train_epoch(model, loader, optimizer, accumulation=1, profiler=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, correct, seen = 0., 0, 0
    timings = dict.fromkeys(['data_loading', 'forward', 'backward', 'optimizer'], 0.)
    iterator = iter(loader)
    start = time.perf_counter()
    window_samples = 0
    for i in range(len(loader)):
        if profiler:
            batch_prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])
            batch_prof.__enter__()
        tick = time.perf_counter()
        with torch.profiler.record_function('phase/data_loading'):
            x, y = next(iterator)
        timings['data_loading'] += time.perf_counter() - tick
        tick = time.perf_counter()
        with torch.profiler.record_function('phase/forward'):
            logits = model(x)
            loss_sum = nn.functional.cross_entropy(logits, y, reduction='sum')
        timings['forward'] += time.perf_counter() - tick
        tick = time.perf_counter()
        with torch.profiler.record_function('phase/backward'):
            loss_sum.backward()
        timings['backward'] += time.perf_counter() - tick
        window_samples += len(y)
        tick = time.perf_counter()
        with torch.profiler.record_function('phase/optimizer'):
            if (i + 1) % accumulation == 0 or i + 1 == len(loader):
                # Normalize by actual window size, including a short final window.
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.div_(window_samples)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                window_samples = 0
        timings['optimizer'] += time.perf_counter() - tick
        total_loss += loss_sum.item()
        correct += (logits.detach().argmax(1) == y).sum().item()
        seen += len(y)
        if profiler:
            batch_prof.__exit__(None, None, None)
            profiler(batch_prof)
    seconds = time.perf_counter() - start
    return {'train_loss': total_loss / seen, 'train_accuracy': correct / seen,
            'train_seconds': seconds, 'throughput': seen / seconds}, timings


def checkpoint(path, model, optimizer, scheduler, generator, epoch, history, c, run_id):
    state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                 generator=generator.get_state(), torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                 python_rng=random.getstate(), epoch=epoch, history=history, config=c, run_id=run_id)
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    temp.replace(path)


def flatten(c, prefix=''):
    result = {}
    for k, v in c.items():
        key = f'{prefix}.{k}' if prefix else k
        if isinstance(v, dict):
            result.update(flatten(v, key))
        else:
            result[key] = str(v)
    return result


def run(c, name, resume=None, stop_after=None, profile=False):
    seed_all(c['seed'])
    torch.set_num_threads(c['threads'])
    t = c['training']
    out = REPO / c['output'] / name
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
    train, val, test = make_data(c)
    generator = torch.Generator().manual_seed(c['seed'])
    loader = make_loader(train, t['batch_size'], t['num_workers'], generator, True)
    val_loader = make_loader(val, 512, 0)
    model = CNN()
    optimizer = make_optimizer(model, t)
    scheduler = make_scheduler(optimizer, t)
    history, start_epoch, saved = [], 0, None
    if resume:
        # Only load locally generated, trusted checkpoints (pickle).
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        if saved['config'] != c:
            raise ValueError('Resume requires identical resolved configuration.')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        generator.set_state(saved['generator'])
        torch.set_rng_state(saved['torch_rng'])
        np.random.set_state(saved['numpy_rng'])
        random.setstate(saved['python_rng'])
        history, start_epoch = saved['history'], saved['epoch']
    mlflow.set_tracking_uri((REPO / c['tracking']['store']).as_uri())
    mlflow.set_experiment(c['tracking']['experiment'])
    with mlflow.start_run(run_id=saved['run_id'] if saved else None, run_name=name) as active:
        mlflow.log_params(flatten(c))
        mlflow.set_tags({'architecture_origin': 'lab03 CNN; retrained from scratch', 'memory_protocol': 'sampled process-tree RSS, 10ms, CPU'})
        env = {'python': platform.python_version(), 'platform': platform.platform(), 'torch': torch.__version__,
               'threads': torch.get_num_threads(), 'device': 'cpu', 'pid': os.getpid(),
               'data_sha256': hashlib.sha256(train.dataset.data.numpy().tobytes()).hexdigest()}
        (out / 'environment.json').write_text(json.dumps(env, indent=2))
        with PeakRSS() as memory:
            for epoch in range(start_epoch + 1, t['epochs'] + 1):
                lr = optimizer.param_groups[0]['lr']
                if profile and epoch == 1:
                    totals = {}
                    def collect(prof):
                        for event in prof.key_averages():
                            row = totals.setdefault(event.key, {'cpu_us': 0., 'self_cpu_us': 0., 'calls': 0})
                            row['cpu_us'] += event.cpu_time_total
                            row['self_cpu_us'] += event.self_cpu_time_total
                            row['calls'] += event.count
                    metrics, timings = train_epoch(model, loader, optimizer, t['accumulation'], collect)
                    phase_total = sum(totals['phase/' + k]['cpu_us'] for k in timings)
                    rows = [{'phase': k, 'cpu_time_ms': totals['phase/' + k]['cpu_us'] / 1000,
                             'share_of_four_phases_percent': 100 * totals['phase/' + k]['cpu_us'] / phase_total,
                             'wall_seconds': timings[k]} for k in timings]
                    write_csv(out / 'profile.csv', rows)
                    write_csv(out / 'operators.csv', [{'operator': key, **value} for key, value in
                              sorted(totals.items(), key=lambda item: -item[1]['self_cpu_us'])])
                else:
                    metrics, timings = train_epoch(model, loader, optimizer, t['accumulation'])
                val_loss, val_acc = evaluate(model, val_loader)
                if t['scheduler'] == 'ReduceLROnPlateau':
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
                row = {'epoch': epoch, 'lr': lr, **metrics, 'val_loss': val_loss, 'val_accuracy': val_acc,
                       'peak_tree_rss_mb': memory.peak / 1e6}
                history.append(row)
                write_csv(out / 'history.csv', history)
                mlflow.log_metrics({k: v for k, v in row.items() if k != 'epoch'}, step=epoch)
                checkpoint(out / 'last.pt', model, optimizer, scheduler, generator, epoch, history, c, active.info.run_id)
                print(name, json.dumps(row), flush=True)
                if stop_after and epoch >= stop_after:
                    mlflow.set_tag('recovery_status', f'interrupted_after_epoch_{epoch}')
                    mlflow.log_artifacts(str(out), artifact_path=name)
                    return None
        test_loss, test_acc = evaluate(model, make_loader(test, 512, 0))
        summary = {'name': name, 'epochs': len(history), 'batch_size': t['batch_size'],
                   'num_workers': t['num_workers'], 'accumulation': t['accumulation'],
                   'optimizer': t['optimizer'], 'scheduler': t['scheduler'],
                   'throughput': sum(len(train) for _ in history) / sum(r['train_seconds'] for r in history),
                   'peak_tree_rss_mb': max(r['peak_tree_rss_mb'] for r in history),
                   'final_val_accuracy': history[-1]['val_accuracy'], 'test_accuracy': test_acc,
                   'test_loss': test_loss, 'run_id': active.info.run_id}
        (out / 'summary.json').write_text(json.dumps(summary, indent=2))
        mlflow.log_metrics({'test_accuracy': test_acc, 'test_loss': test_loss})
        if resume:
            mlflow.set_tag('recovery_status', 'resumed_and_completed')
        mlflow.log_artifacts(str(out), artifact_path=name)
        return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='lab06/config.yaml')
    p.add_argument('--name', default='baseline')
    p.add_argument('--resume')
    p.add_argument('--stop-after', type=int)
    p.add_argument('--profile', action='store_true')
    args = p.parse_args()
    run(load_config(args.config), args.name, args.resume, args.stop_after, args.profile)


if __name__ == '__main__':
    main()
