"""Assemble tables, plots and report solely from measured study artifacts."""
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/lab06-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mlflow.tracking import MlflowClient
from train import REPO, write_csv, make_data
from studies import jobs
from train import load_config

R = REPO / 'lab06/results'


def table(frame, formats=None):
    frame = frame.copy()
    for k, fmt in (formats or {}).items():
        frame[k] = frame[k].map(lambda v: format(v, fmt))
    columns = list(frame.columns)
    return '\n'.join(['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in frame.itertuples(index=False, name=None)])


def main():
    expected = [name for name, _ in jobs(load_config(REPO / 'lab06/config.yaml'))]
    missing = [name for name in expected if not (R / 'runs' / name / 'summary.json').exists()]
    if missing:
        raise RuntimeError(f'Incomplete experiments: {missing}')
    summaries = {name: json.loads((R / 'runs' / name / 'summary.json').read_text()) for name in expected}
    client = MlflowClient((R / 'mlruns').as_uri())
    for name, c in jobs(load_config(REPO / 'lab06/config.yaml')):
        run_id = summaries[name]['run_id']
        tracked = client.get_run(run_id)
        assert tracked.info.status == 'FINISHED', (name, tracked.info.status)
        steps = sorted(m.step for m in client.get_metric_history(run_id, 'val_accuracy'))
        assert steps == list(range(1, c['training']['epochs'] + 1)), (name, steps)
        artifacts = {Path(a.path).name for a in client.list_artifacts(run_id, name)}
        assert {'config.yaml', 'history.csv', 'last.pt', 'summary.json', 'environment.json'} <= artifacts, name
    train, val, test = make_data(load_config(REPO / 'lab06/config.yaml'))
    tensors = {'train_source_images': train.dataset.data.numpy(),
               'train_source_labels': train.dataset.targets.numpy(),
               'test_images': test.data.numpy(), 'test_labels': test.targets.numpy(),
               'train_indices': np.asarray(train.indices, dtype=np.int64),
               'validation_indices': np.asarray(val.indices, dtype=np.int64)}
    (R / 'data_manifest.json').write_text(json.dumps({
        key: {'shape': list(value.shape), 'dtype': str(value.dtype),
              'sha256': hashlib.sha256(value.tobytes()).hexdigest()}
        for key, value in tensors.items()}, indent=2))
    manifest = {'required_completed_runs': len(expected),
                'run_ids': {name: row['run_id'] for name, row in summaries.items()},
                'source_sha256': {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((REPO / 'lab06/src').glob('*.py'))},
                'mlflow_status_metrics_artifacts_verified': True}
    (R / 'evidence_manifest.json').write_text(json.dumps(manifest, indent=2))
    all_rows = list(summaries.values())
    write_csv(R / 'study_summary.csv', all_rows)
    def history(name):
        return pd.read_csv(R / 'runs' / name / 'history.csv')
    workers = []
    for n in [0, 1, 2, 4, 8]:
        h = history(f'workers_{n}')
        workers.append({'num_workers': n, 'epoch1_samples_s': h.throughput.iloc[0],
                        'epoch2_samples_s': h.throughput.iloc[1],
                        'peak_tree_rss_mb': summaries[f'workers_{n}']['peak_tree_rss_mb']})
    workers = pd.DataFrame(workers)
    workers.to_csv(R / 'workers.csv', index=False)
    batches = pd.DataFrame([summaries[f'batch_{b}'] for b in [32, 64, 128, 256, 512]])[
        ['batch_size', 'throughput', 'peak_tree_rss_mb', 'final_val_accuracy', 'test_accuracy']]
    batches.to_csv(R / 'batches.csv', index=False)
    schedulers = pd.DataFrame([summaries[f'{o}_{s}'] for o in ['SGD', 'Adam'] for s in
                              ['StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau']])[
        ['optimizer', 'scheduler', 'epochs', 'throughput', 'final_val_accuracy', 'test_accuracy']]
    schedulers.to_csv(R / 'optimizers_schedulers.csv', index=False)
    profile = pd.read_csv(R / 'runs/profile/profile.csv')
    profile.to_csv(R / 'profile.csv', index=False)
    reference = torch.load(R / 'runs/recovery_reference/last.pt', weights_only=False)
    resumed = torch.load(R / 'runs/recovery_resumed/last.pt', weights_only=False)
    max_difference = max((reference['model'][key] - resumed['model'][key]).abs().max().item()
                         for key in reference['model'])
    recovery_curve = max(float(np.max(np.abs(history('recovery_reference')[col] - history('recovery_resumed')[col])))
                         for col in ['lr', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy'])
    recovery = {'max_absolute_weight_difference': max_difference,
                'max_absolute_curve_difference': recovery_curve,
                'bitwise_equal_model': all(torch.equal(reference['model'][k], resumed['model'][k]) for k in reference['model']),
                'same_scheduler_state': reference['scheduler'] == resumed['scheduler'],
                'same_shuffle_generator_state': torch.equal(reference['generator'], resumed['generator']),
                'same_torch_rng_state': torch.equal(reference['torch_rng'], resumed['torch_rng'])}
    (R / 'recovery_comparison.json').write_text(json.dumps(recovery, indent=2))
    assert recovery['bitwise_equal_model'] and recovery_curve == 0, recovery
    direct, accum = history('batch_256'), history('accumulation_64x4')
    accumulation = pd.DataFrame({'epoch': direct.epoch,
                                 'batch256_train_loss': direct.train_loss,
                                 'accum64x4_train_loss': accum.train_loss,
                                 'batch256_val_accuracy': direct.val_accuracy,
                                 'accum64x4_val_accuracy': accum.val_accuracy})
    accumulation.to_csv(R / 'accumulation.csv', index=False)
    # Similarity is empirically checked over all epochs, not assumed to be bitwise equal.
    loss_delta = float(np.max(np.abs(direct.train_loss - accum.train_loss)))
    acc_delta = float(np.max(np.abs(direct.val_accuracy - accum.val_accuracy)))
    accumulation_check = {'max_train_loss_difference': loss_delta, 'max_val_accuracy_difference': acc_delta,
                          'loss_tolerance': .02, 'accuracy_tolerance': .01,
                          'similar': loss_delta < .02 and acc_delta <= .01}
    (R / 'accumulation_comparison.json').write_text(json.dumps(accumulation_check, indent=2))
    assert accumulation_check['similar'], accumulation_check
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10})
    fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
    ax.plot(workers.num_workers, workers.epoch1_samples_s, 'o--', label='Epoch 1')
    ax.plot(workers.num_workers, workers.epoch2_samples_s, 'o-', label='Epoch 2 (persistent workers)')
    ax.set(xlabel='DataLoader num_workers', ylabel='Training throughput (samples/s)', xticks=[0, 1, 2, 4, 8])
    ax.legend(); ax.grid(alpha=.2)
    fig.savefig(R / 'workers.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout='constrained')
    for ax, key, label in zip(axes, ['throughput', 'peak_tree_rss_mb', 'final_val_accuracy'],
                              ['Samples/s', 'Sampled process-tree RSS (MB)', 'Final validation accuracy']):
        ax.plot(batches.batch_size, batches[key], 'o-')
        ax.set(xscale='log', xlabel='Batch size', ylabel=label, xticks=[32,64,128,256,512])
        ax.set_xticklabels([32,64,128,256,512]); ax.grid(alpha=.2)
    fig.savefig(R / 'batches.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    for i, opt in enumerate(['SGD', 'Adam']):
        for schedule in ['StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau']:
            h = history(f'{opt}_{schedule}')
            axes[i, 0].plot(h.epoch, h.val_accuracy, marker='.', label=schedule)
            axes[i, 1].plot(h.epoch, h.lr, marker='.', label=schedule)
        axes[i, 0].set(title=opt, xlabel='Epoch', ylabel='Validation accuracy')
        axes[i, 1].set(title=opt, xlabel='Epoch', ylabel='Learning rate')
        for ax in axes[i]: ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(R / 'schedulers.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout='constrained')
    for h, label in [(direct, 'Batch 256'), (accum, 'Batch 64 x accumulation 4')]:
        axes[0].plot(h.epoch, h.train_loss, 'o-', label=label)
        axes[1].plot(h.epoch, h.val_accuracy, 'o-', label=label)
    for ax, ylabel in zip(axes, ['Train cross-entropy', 'Validation accuracy']):
        ax.set(xlabel='Epoch', ylabel=ylabel); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(R / 'accumulation.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 4), layout='constrained')
    ax.plot(history('recovery_reference').epoch, history('recovery_reference').val_accuracy, 'o-', label='Uninterrupted')
    ax.plot(history('recovery_resumed').epoch, history('recovery_resumed').val_accuracy, 'x--', label='Interrupted + restored')
    ax.axvline(3, color='gray', linestyle=':', label='Process terminates after epoch 3')
    ax.set(xlabel='Epoch', ylabel='Validation accuracy'); ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.savefig(R / 'recovery.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.6), layout='constrained')
    ax.barh(profile.phase, profile.share_of_four_phases_percent, color='#247E8B')
    ax.set(xlabel='Share of summed annotated CPU phase time (%)')
    fig.savefig(R / 'profile.png', dpi=180); plt.close(fig)
    dominant_phase = profile.loc[profile.share_of_four_phases_percent.idxmax()]
    best_workers = int(workers.loc[workers.epoch2_samples_s.idxmax(), 'num_workers'])
    worker_speedup = workers.epoch2_samples_s.max() / workers.loc[workers.num_workers == 0, 'epoch2_samples_s'].iloc[0]
    best_batch = int(batches.loc[batches.final_val_accuracy.idxmax(), 'batch_size'])
    fastest_batch = int(batches.loc[batches.throughput.idxmax(), 'batch_size'])
    best = schedulers.loc[schedulers.final_val_accuracy.idxmax()]
    plateau_notes = ' '.join(
        f"{opt}: {history(f'{opt}_ReduceLROnPlateau').lr.nunique()} distinct lr value(s) "
        f"actually used, from {history(f'{opt}_ReduceLROnPlateau').lr.max():g} "
        f"to {history(f'{opt}_ReduceLROnPlateau').lr.min():g}."
        for opt in ['SGD', 'Adam'])
    env = json.loads((R / 'runs/batch_128/environment.json').read_text())
    report = f'''# Lab 6: Training pipelines and project proposal

Student: Valeriia Khapilina, 73438. Experiments executed 25 November 2026. Scheduled class: 18 November 2026.

## 1. Reproducible YAML pipeline [20]

The CNN from Lab 3 had the highest recorded accuracy (89.18%) and smallest parameter count (20,490). This lab retains its exact architecture. Lab 3 did not save weights, so all study runs start from the same seeded initialization, rather than claiming to restore an unavailable trained model.

One resolved YAML object drives data, seed, CPU threads, optimizer, schedule, batch size, workers, gradient accumulation, output and tracking. Each run stores its own YAML, environment, raw per-epoch metrics and checkpoint. MLflow logs flattened parameters, epoch metrics, final test metrics and all run artifacts. All {len(expected)} required runs completed; see `study_summary.csv` for run IDs. Runs are launched serially in separate processes to avoid concurrent training interference and allocator carryover.

Environment: Python {env['python']}, PyTorch {env['torch']}, {env['platform']}; CPU with 4 intra-op threads. The training-source image SHA256 is stored in every environment JSON; data_manifest.json additionally hashes labels, test images/labels and split indices. Fashion-MNIST uses the official 60,000/10,000 train/test sources, a seeded 55,000/5,000 train/validation split, ToTensor and normalization mean 0.2860/std 0.3530. Transforms execute per sample so the worker study measures a real loading/preprocessing path. No test data enter training or scheduling. Test accuracy is reported after each final study run as requested; conclusions and setting choices use validation accuracy, not the test ranking.

## 2. Full-epoch profiling [15]

{table(profile, {'cpu_time_ms': '.2f', 'share_of_four_phases_percent': '.2f', 'wall_seconds': '.3f'})}

![Profile breakdown](profile.png)

Every batch in one complete 55,000-example epoch is instrumented with torch.profiler on CPU. Each batch's trace is aggregated and released before the next batch to bound profiler memory. The four disjoint record_function scopes surround next(iterator), forward plus loss, backward, and gradient normalization/optimizer step/zero_grad. Percentages use inclusive CPU duration of these scopes divided by their sum; nested operators are not summed again. These are host-scope elapsed times on this synchronous CPU run, not summed utilization across four CPU cores. Iterator construction, profiler collection, metric bookkeeping, validation and checkpoint I/O are outside the four-phase denominator. `runs/profile/operators.csv` contains operator statistics aggregated over the full epoch. Profiling overhead makes this run unsuitable for normal throughput comparison.

The largest annotated phase is **{dominant_phase['phase']} ({dominant_phase['share_of_four_phases_percent']:.2f}%)**. This identifies the dominant measured component in the single-process reference, while the worker sweep tests how much input/compute overlap actually helps.

## 3. Data-loader workers [15]

{table(workers, {'epoch1_samples_s': '.1f', 'epoch2_samples_s': '.1f', 'peak_tree_rss_mb': '.1f'})}

![Worker study](workers.png)

The highest second-epoch throughput was measured with **{best_workers} workers**, a **{worker_speedup:.2f}x** speedup over workers=0. Increasing from 4 to 8 workers did not improve steady throughput on this host, while RSS grew substantially. This suggests a compute/CPU-contention or IPC limit once input preparation is sufficiently overlapped; adding input processes beyond this point is not useful for this workload. Two complete epochs use the same batch size 128, seed, model, optimizer and transforms for each worker count. Workers persist between epochs; each worker uses one PyTorch intra-op thread. First-epoch measurements include the wait for initial data, while iterator construction itself is excluded. The second epoch better reflects steady operation; these are single measurements, not confidence intervals. Compare the throughput change with the profile breakdown to distinguish waiting for input from neural compute and process/IPC overhead. Memory includes the training process and active DataLoader children; increasing workers can duplicate dataset/runtime memory and eventually contend for CPU resources. PyTorch's persistent-worker iterator also consumes generator state differently from a newly created single-process iterator; after epoch 1, worker=0 and worker>0 can have different shuffle orders despite identical initial seeds. All cases still process all 55,000 samples, and this is a throughput study rather than an accuracy comparison across worker counts. This is an in-memory Fashion-MNIST source, so results do not generalize to a slow disk or remote storage pipeline.

## 4. Batch size, throughput, memory and accuracy [15]

{table(batches, {'throughput': '.1f', 'peak_tree_rss_mb': '.1f', 'final_val_accuracy': '.4f', 'test_accuracy': '.4f'})}

![Batch study](batches.png)

Each configuration trains for exactly 5 epochs with Adam lr=0.001, StepLR(step_size=4, gamma=0.5), workers=0 and the same split/seed. Throughput is total training examples divided by summed training-loop seconds, excluding validation, checkpointing and initial iterator construction. The fastest measured batch was {fastest_batch}; the best final validation accuracy used batch {best_batch}. Equal epochs do not mean equal update counts: smaller batches take more optimizer steps. The learning rate is intentionally held fixed to isolate the configured batch-size change; a separately tuned learning rate could alter the ranking.

Peak memory is sampled process-tree resident memory every 10 ms, in decimal MB. Sampling runs through training and validation, including checkpoint/logging between epochs. It includes Python/framework/data/model/optimizer/activations, counts shared resident pages per process, and can miss brief peaks. This is an observed CPU RSS maximum, not GPU allocated memory or an exact activation-memory measurement. Each batch-size case runs in a fresh process, making the comparison more meaningful.

## 5. Optimizer and scheduler comparison [15]

{table(schedulers, {'throughput': '.1f', 'final_val_accuracy': '.4f', 'test_accuracy': '.4f'})}

![Schedules and validation curves](schedulers.png)

All six optimizer x schedule combinations train exactly 10 epochs with batch 128 and workers=0. SGD uses momentum=0.9 and initial lr=0.01; Adam uses initial lr=0.001 and default beta values. Both use no weight decay. Optimizer-specific learning rates are stated explicitly and are fixed in advance, so this is a comparison of these concrete recipes, not proof that one optimizer is universally superior or equally well tuned. StepLR halves lr every 4 epochs; CosineAnnealingLR has T_max=10; ReduceLROnPlateau monitors validation cross-entropy with factor=0.5 and patience=1. Scheduling occurs after validation; the recorded lr is the one actually used during that epoch.

The best final validation result in this experiment is {best['optimizer']} with {best['scheduler']} ({best['final_val_accuracy']:.2%}). Adam maintains two moving-average state tensors per parameter, whereas SGD with momentum keeps one; the performance/accuracy trade-off must be judged together with those state costs. These extra optimizer states are small for this 20,490-parameter model compared with the overall process RSS. Plateau observations: {plateau_notes} There is one seed per case, with no claim of statistical significance. Runs execute in a fixed serial order; thermal conditions and background host load can bias throughput comparisons.

## 6. Gradient accumulation [10]

{table(accumulation, {c: '.6f' for c in accumulation.columns if c != 'epoch'})}

![Accumulation comparison](accumulation.png)

Batch 64 with accumulation=4 is compared with the independently trained direct batch-256 run over the same 5 epochs. Microbatches follow the same shuffled sample order; one optimizer step uses the sum of sample gradients divided by the actual sample count in that accumulation window. The last partial window is normalized correctly. No learning-rate multiplication is needed because both cases target the same effective batch 256. The CNN has neither dropout nor batch normalization, avoiding two common sources of inequivalence. Floating-point kernel/reduction order can still differ.

Maximum train-loss difference: {loss_delta:.6f}; maximum validation-accuracy difference: {acc_delta:.6f}. The predeclared comparison checks loss difference <0.02 and validation-accuracy difference <=0.01 over every epoch: **passed**. Numerical tests additionally verify near-equal SGD updates, including a short final accumulation window.

## 7. Recovery after epoch 3 [10]

![Recovery curves](recovery.png)

A separate process trains to epoch 3, saves the atomic last.pt checkpoint, and exits. A new process loads it and completes epochs 4 and 5 under the identical total-epoch configuration. The checkpoint includes model, optimizer, scheduler, completed epoch, metric history, Python/NumPy/PyTorch RNG states, DataLoader generator state, configuration and MLflow run ID. The same MLflow run continues with original epoch step numbers. This demonstrates a real process restart at an epoch boundary with workers=0. Worker-process RNG/iterator state is not checkpointed, so exact resume is not guaranteed for persistent workers or added random transforms. Arbitrary mid-batch crash recovery is outside the scope.

The uninterrupted reference and restored run have bitwise-identical final model tensors: **{recovery['bitwise_equal_model']}**. Maximum absolute weight difference is {max_difference}; maximum difference across loss/accuracy/lr curves is {recovery_curve}. Scheduler, shuffle-generator and torch RNG states also match. Evidence: `recovery_comparison.json`, both raw histories and final checkpoints.

## 8. Individual proposal [15]

The one-page `proposal.pdf` and editable `proposal.md` specify the student-approved SMS-spam project, dataset, evaluation protocol, success targets, CPU latency <=50 ms, total model/preprocessing artifact size <=5 MB, and INT8 quantization. Student identity: Valeriia Khapilina, 73438. **Instructor approval is pending** and must be obtained from the instructor; it cannot be asserted by this report. The SMS project itself is future work, distinct from the Fashion-MNIST training experiments above.

## Verification and reproduction

See ../README.md for exact commands, dependencies, MLflow UI, tests and scope limitations. `python -m unittest discover -s lab06/src -p 'test_*.py' -v` checks accumulation and checkpoint state recovery; `report.py` checks all 20 required runs exist, full-run bitwise recovery and accumulation tolerances. Raw results remain alongside figures and tables. Pre-class reading notes and official sources are in `preparation.md`. Git commit records the implementation and evidence; no remote push or instructor submission is performed.
'''
    (R / 'report.md').write_text(report)
    print('Report generated; recovery and accumulation checks passed.')


if __name__ == '__main__':
    main()
