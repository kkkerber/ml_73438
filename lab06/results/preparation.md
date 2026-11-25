# Lab 6 preparation and oral-defense notes

Student: Valeriia Khapilina, 73438. Scheduled lab: Wednesday, 18 November 2026, 16:50-18:45, room 134, Stoklosy 3, Warsaw. This work was prepared on 25 November 2026.

## Reading map

The prescribed sections are Reddi, Chapter 8 (training architecture, loading, optimization and profiling), including 8.7 Optimization Techniques; Chapter 5.5 Model Development; and Chapter 13.3 MLOps Key Components. The online book is revised over time, so titles and subsection numbers in the accessible development edition differ from the manual. The following official book pages cover the corresponding topics; they are not claimed to be a verified identical numbered edition.

- [AI Training](https://harvard-edge.github.io/cs249r_book_dev/contents/core/training/training.html): separate data, forward, backward and parameter-update stages. Optimize the limiting stage after measuring it. Prefetching overlaps input preparation; batch size trades activation memory and update frequency against processing efficiency. Accumulation combines sample gradients across microbatches before an update. Distinguish activation checkpointing (recompute activations to save memory) from recovery checkpoints (save persistent training state).
- [AI Workflow: Model Development and Training](https://harvard-edge.github.io/cs249r_book_dev/contents/core/workflow/workflow.html#model-development-training-stage): define success and deployment constraints before tuning; keep validation and final test roles separate; use controlled experiments and track artifacts.
- [ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/core/ops/ops.html): operational components include repeatable data preparation, pipeline automation, versioning, experiment tracking, deployment and monitoring. Configuration, data lineage, metrics and model artifacts together make results traceable.

## Concepts applied in this implementation

| Concept | What to explain during the lab |
|---|---|
| YAML configuration | One object specifies data, seed, optimizer, schedule, batch, worker count, accumulation and output. Each run saves its resolved object. |
| MLflow | Parameters describe the recipe; metrics are numerical observations indexed by epoch; artifacts are files such as configs, curves and checkpoints. Tracking is local here. |
| torch.profiler | Annotate disjoint training phases; inspect phase durations and operator statistics. Do not sum nested inclusive operator times as if independent. |
| num_workers | Worker processes prepare inputs in parallel. Too few can starve compute; too many add startup, IPC, memory pressure and CPU contention. Persistent workers amortize startup. |
| Throughput | Samples divided by measured training-loop seconds. State whether preprocessing, warmup, validation and checkpoint I/O are included. Batch latency and samples/s are different quantities. |
| Peak memory | This CPU run samples resident memory every 10 ms. It includes framework and dataset memory, not just tensors, and may miss short peaks. |
| StepLR | Multiply lr by gamma every fixed number of completed epochs. |
| CosineAnnealingLR | Smooth cosine decay over the configured total budget T_max. |
| ReduceLROnPlateau | Reduce lr when the monitored validation loss stops improving according to patience/threshold; use validation loss, not test accuracy. |
| SGD + momentum | Momentum smooths updates; it adds one state tensor per parameter. Learning rate often needs task-specific tuning. |
| Adam | Adaptive updates use first/second moment estimates; two state tensors per parameter. Faster convergence is possible, not guaranteed. |
| Gradient accumulation | Effective batch = microbatch x accumulation. Normalize by the actual number of samples, including the short final window; step once per window. Same effective batch does not require an lr change. |
| Recovery | Restore weights, optimizer, scheduler, epoch, RNG and shuffle state. This implementation resumes only at epoch boundaries. |
| Reproducibility | Fix initialization/split/shuffle seeds and deterministic algorithms; record environment and data checksum. Timing remains noisy even when model tensors reproduce exactly. |
| Deployment | Specify target CPU, batch 1, latency percentile, preprocessing scope and size units. Quantization must be evaluated for both quality loss and measured resource benefit. |

## Practical checks and evidence

- Required training packages are present: torch, torchvision, MLflow, PyYAML, NumPy, pandas and Matplotlib. Exact installed versions are frozen in `environment_packages.txt`.
- Best Lab 3 architecture: CNN. No saved weights were present, so this lab retrains the architecture rather than pretending to load them.
- Data are already downloaded under `data/FashionMNIST`; no external tracking account is used.
- The original lab PDF remains available at the Downloads location supplied by the student. Bring that digital copy or print it.
- Proposal: student-selected and student-approved SMS-spam classification with INT8 quantization; instructor approval remains to be obtained.

## Short answers in Russian

**Почему нужны seed и checkpoint одновременно?** Seed задаёт воспроизводимый старт, а checkpoint сохраняет конкретное состояние уже идущего обучения, включая optimizer и scheduler.

**Почему больше workers не всегда быстрее?** Подготовка данных может перестать быть узким местом; после этого дополнительные процессы увеличивают накладные расходы и расход памяти.

**Почему batch 64 x 4 похож на batch 256?** Градиенты отдельных примеров складываются. При одинаковом порядке данных и нормировке optimizer получает почти тот же средний градиент. Различия возможны из-за округления, dropout и BatchNorm.

**Почему нельзя выбирать настройки по test accuracy?** Тогда test set становится частью настройки и перестаёт давать независимую оценку. Здесь выбор делается по validation.

**Что ещё надо сделать лично?** Прочитать эти объяснения и исходную методичку, разобраться в измеренных графиках и получить утверждение proposal у преподавателя.
