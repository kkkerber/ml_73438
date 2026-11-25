# Execution notes

- An initial unrestricted full-epoch torch.profiler trace retained too many events and grew to approximately 3 GB before postprocessing completed. That attempt was terminated and its MLflow run marked KILLED with an explanatory tag. The completed profiling study aggregates every batch's profiler events over the entire epoch, releasing per-batch events immediately. No timing values from the killed attempt enter the final tables.
- An initial sandboxed study launch failed before training with OpenMP shared-memory error #179. The experiments were then run with the local process/shared-memory permissions required by OpenMP and multiprocessing. The failed launch output remains at the beginning of workers_0.log; all final tabulated metrics come from its subsequent completed run.
- Peak RSS is obtained from the training process and its known multiprocessing child PIDs. It does not scan unrelated system processes.
- The source PDF and existing Lab 3 files are unchanged. A pre-existing modification to lab03/results/environment.json was left out of this lab's changes and commit.
- Instructor approval of the individual proposal is an external action and remains pending. No email, message, submission or remote Git push was performed.

- Verified the MLflow relocation command on a temporary copy: experiment/run artifact URIs were relinked, metrics preserved and checkpoint artifacts accessible; the original store was unchanged. Temporary paths on macOS were compared after resolving the /var to /private/var alias.
