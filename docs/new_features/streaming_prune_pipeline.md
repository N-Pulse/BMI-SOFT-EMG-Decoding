# Streaming the prune pipeline

Status: proposed. Nothing here is implemented yet.

## Problem

`DatasetPruner.run` loads every file into memory before the first step runs, so peak RAM grows with the size of the dataset. On large sources the process gets killed.

Three things stack up:

1. `Loader.load` (`src/bmiemg/prune/adapters/loader.py`) returns a `list[DataActor]` holding every loaded actor at once.
2. `DatasetPruner.apply` runs `actors = step.apply(actors)`. While a step runs, the input list and the output list are both alive, and most signal steps allocate new arrays, so the peak is about twice the dataset.
3. On export, `DatasetPartition.mne_dataset` builds one `mne.io.RawArray` per trial and concatenates the whole partition. The test trials exist twice (`client.test` with hidden markers, `server.test` with real ones), and `_hide_markers` allocates a fresh zero array per trial.

## Goal

Bound peak RAM to roughly "one file times the pipeline's working space, times the number of workers", with a configurable maximum number of parallel workers, while producing the same output as today: same trials, same split for the same seed, same files on disk.

## Why plain batching is not enough

Running the full pipeline on chunks of N files would change the result: `TTVSplitStep` splits over all trials with one seeded RNG, so per-chunk splits differ from a single run.

The way out is that only `ttv_split` needs the whole dataset, and it only needs metadata (subject, dominant movement), never the signal arrays.

## Design

```
files ─► ProcessPoolExecutor(max_workers=N)
           each worker: load_actors → actor steps → write trials to disk → return metadata
      ─► global step (ttv_split) on metadata only → partition assignment
      ─► writer: reads trials by assignment, writes client/server sets
```

### Actor vs global steps

`PruneStep` gets one ClassVar:

```python
scope: ClassVar[Literal["actor", "global"]] = "actor"
```

Only `TTVSplitStep` sets `"global"`. No existing step changes. `DatasetPruner` splits the protocol at its global steps into segments: the actor steps of a segment run per file in the workers, a global step runs once after all workers are done (collecting `executor.map` results is the barrier). See "Several global steps" for how segments chain.

An explicit flag rather than "the last step is global" keeps the rule visible and allows global steps in the middle of a protocol (e.g. normalisation with train-set statistics).

### Phase 1: per-file workers

1. `Loader` gets `iter_files(source_names)` yielding paths in today's order (sources, then suffix groups, then `rglob` order). `load()` stays for the in-memory path and the tests.
2. Each worker loads one file (`load_actors`), runs the actor steps with `step.apply(actors)` on that file's actors, writes every resulting `TrialData` to a run cache folder (`np.save` of `signal_ch` and `markers`, dtype kept so output is bit-identical), and returns one metadata record per trial: subject, dominant movement, cache path, plus the fields the writer needs (channel names, sfreq, file name).
3. Fan-out is free: a file that becomes N trials returns N records; a filtered file (`remove_session`) returns none.
4. Workers return records, never arrays. Each worker handles exactly one file at a time, with every actor that file yields. `max_workers` is a `DatasetPruner` parameter and the only concurrency knob: files in memory at once = `max_workers`. Size it as `ram_budget / peak_per_file` rounded down, where `peak_per_file` is the peak RAM one worker uses on the largest file (its actors plus the copies the steps make). Example: 16 GB budget, 3 GB peak → 5 workers.

The steps never deal with metadata bookkeeping: the record is derived once from the final `TrialData` of the chain.

If a protocol has no global step, workers write their trials straight to the output and there is no cache.

### Phase 2: the global step

`TTVSplitStep` runs on the metadata records. Its logic already only reads the subject and the dominant movement, so `_strata`, `_split_groups` and `_split_stratum` stay; the records only need to expose those two values (store the dominant movement at record time).

- `group_by: subject` sorts subject keys, so worker order does not matter. The stratified mode shuffles in list order, so records must come back in file order (`executor.map` preserves it).
- `_hide_markers` moves out of the step: the split returns assignments, and the writer zeroes markers for `client.test` while writing.

### Several global steps

A global step is not a plain `apply` over everything. It has three parts:

1. `summarise(actor) -> summary`, in the worker, per actor: a small record returned with the metadata.
2. `reduce(summaries) -> params`, once, after the barrier.
3. `apply_params(actor, params) -> actor`, per actor, at the start of the next segment's workers.

`ttv_split` fits it: summarise = subject and dominant movement, reduce = partition assignment, apply = tag each trial with its partition (the writer reads the tag). A dataset-wide normalisation: summarise = per-channel sum, sum of squares and count, reduce = mean and std, apply = normalise the trial.

A protocol with k global steps runs k + 1 segments:

```
segment 1: workers read source files → actor steps → summarise → write cache → return records
global 1:  reduce
segment 2: workers read cache → apply global 1 → actor steps → summarise → write cache → return records
...
last:      writer reads cache → output files
```

Constraints:

- Each global step costs one extra read and write of the cache.
- Statistics computed before `ttv_split` include the test set (leakage). A step that should use train-set statistics only must come after the split; its summarise reads the partition tag and skips non-train trials.
- Summaries must combine exactly from per-actor parts (sums, counts, min/max, sets). Medians or percentiles would need histograms or another pass.

Scope of the first implementation: the segment loop is written for any number of global steps, but only `ttv_split` exists, so in practice there is one barrier and it is the last step.

### Phase 3: export

`DatasetWriter` reads each partition's trials from the cache by assignment and must stop building one big `Raw` in memory: write each trial to a temporary `*_raw.fif`, open them with `mne.io.read_raw_fif(path, preload=False)`, `mne.concatenate_raws` and `.save()`. MNE then streams the data in chunks. Same files on disk as today.

The cache folder is removed in a `finally` block (or kept behind a `keep_cache` flag for debugging).

## Interface changes

- `PruneStep`: `scope` ClassVar, default `"actor"`; `TTVSplitStep.scope = "global"`. Global steps implement `summarise`, `reduce` and `apply_params`.
- `Loader`: `iter_files`.
- `DatasetPruner`: `max_workers` parameter; a streaming `run_and_export(protocol_name)` (a streaming run cannot return every actor); helpers to split steps at the global one and to run one file. One run timestamp shared by the cache and the writers.
- `TTVSplitStep`: accepts metadata records; no marker hiding.
- `DatasetWriter`: per-trial writes, lazy concatenation, marker hiding for `client.test`.
- `class.puml`: `scope` on `PruneStep`, `iter_files` on `Loader`, the new `DatasetPruner` members.

## Known risks

- A step marked `"actor"` that secretly needs the whole list gives a different result silently. Covered by the scope test below.
- `ProtocolDefinerStep` mutates actors in place. Harmless per worker, but it should use `copy_with`.
- Steps and loaders must be picklable for the workers; they are plain dataclasses today.
- Disk space: the cache needs roughly the size of the processed signals.

## Testing

1. Equivalence: on a small fixture dataset, the in-memory path and the streaming path with the same protocol and seed export identical FIF data, channel names, sfreq and partition membership. Main acceptance test.
2. Scope: for every `"actor"` step, `step.apply(xs) == [y for x in xs for y in step.apply([x])]` on a few actors.
3. Segments: a protocol with a fake global step in the middle (e.g. a counting step) gives the same result streamed as in memory, and its summaries reach `reduce` from every actor.
4. Ordering: with several workers, records reach the global step in file order.
5. Cleanup: the cache folder is gone after a successful run and after one that raises.
6. Memory: peak RSS on synthetic large files stays under a bound. Marked slow, never run on a full dataset locally.

## Suggested order of work

1. Add `scope` to `PruneStep`, mark `TTVSplitStep` global and split it into `summarise`/`reduce`/`apply_params`, add the scope test.
2. Add `Loader.iter_files` and the per-file worker with the cache, sequential first.
3. Make `TTVSplitStep` work on metadata records and move marker hiding to the writer.
4. Rewrite `DatasetWriter` to write per trial and concatenate lazily; pass the equivalence test.
5. Add `ProcessPoolExecutor` with `max_workers`.
6. Update `class.puml`, the prune README and the prune scripts.
