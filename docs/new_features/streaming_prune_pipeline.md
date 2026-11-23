# Streaming the prune pipeline

Status: proposed. Nothing here is implemented yet.

## Problem

`DatasetPruner.run` loads every file into memory before the first step runs, so peak RAM grows with the size of the dataset. On large sources the process gets killed.

Three things stack up:

1. `Loader.load` (`src/bmiemg/prune/adapters/loader.py`) returns a `list[DataActor]` holding every `XdfData` at once.
2. `DatasetPruner.apply` runs `actors = step.apply(actors)`. While a step runs, the input list and the output list are both alive. `signal_split`, `signal_unwarp`, `analogue_filter` and `marker_unification` all allocate new arrays, so the peak is about twice the dataset.
3. On export, `DatasetPartition.mne_dataset` calls `np.vstack` and `mne.io.RawArray` for every trial and then `mne.concatenate_raws` over the whole partition. The test trials exist twice (`client.test` with hidden markers, `server.test` with real ones), and `_hide_markers` allocates a fresh zero array per trial.

## Goal

Bound peak RAM to roughly "one file times the pipeline's working space" (times the number of worker processes, if parallel), while producing exactly the same output as today: same trials, same split for the same seed, same files on disk.

## Why plain batching is not enough

Running the full pipeline on chunks of N files would change the result. `TTVSplitStep` stratifies by subject and dominant movement and shuffles with one seeded RNG over all trials. Run per batch, every batch gets its own strata and its own shuffle, so the train/val/test membership differs from a single run.

The way out is that only one step needs the whole dataset, and that step only needs metadata.

## Step classification

| Step | YAML key | What it needs | Kind |
|---|---|---|---|
| `RemoveSessionStep` | `remove_session` | the file name | per actor, filter (1 to 0 or 1) |
| `SignalSplitStep` | `signal_split` | one `XdfData` | per actor |
| `ProtocolDefinerStep` | `protocol_definer` | one actor's `date` | per actor |
| `SignalUnwarpStep` | `signal_unwarp` | one `XdfData` | per actor |
| `AnalogueFilterStep` | `analogue_filter` | one `TrialData` | per actor |
| `MarkerUnificationStep` | `marker_unification` | one `TrialData` | per actor |
| `BuildEpochs` | `build_epochs` | one `TrialData` (on a `Dataset` it just loops over the trials) | per actor, 1 to N |
| `TTVSplitStep` | `ttv_split` | every trial's `subject_number` and dominant movement | global (barrier) |

The barrier never reads the signal arrays. That is what makes streaming safe.

## Design

The pipeline becomes three phases.

```
files ──► [per-actor steps] ──► spill to disk ──► handles ──► [global step] ──► [per-actor steps] ──► streamed writer
          one file at a time                     metadata only                   one trial at a time
```

### Phase 1: stream and spill

1. `Loader` gets an `iter_load(source_names) -> Iterator[DataActor]` that yields one actor per file. Keep `load()` as `list(self.iter_load(...))` so existing callers and tests keep working. File discovery (`rglob`, the readable summary) stays as is. Only the `load_file` calls become lazy.
2. `DatasetPruner` splits the protocol's steps at the first global step: `head` (per-actor), `barrier`, `tail` (whatever follows).
3. For each yielded actor, run the `head` steps on a one-element list (or via `apply_one`, see below). A step may return zero actors (`remove_session`) or several (`build_epochs`).
4. Each resulting `TrialData` is spilled: `signal_ch` and `markers` are saved with `np.save` into a cache folder, for example `out_dir(protocol_name) / ".cache" / f"{run_timestamp}"`. The in-memory object is then replaced by a copy whose arrays come from `np.load(path, mmap_mode="r")`.
5. Store the dominant movement in the handle at spill time, so the barrier never touches the arrays. Either add an optional field to `TrialData` or keep a side dict keyed by the cache path.

A memory-mapped `TrialData` is the handle. `copy_with`, slicing in `BuildEpochs._segment_trial`, and `np.asarray` all work on `np.memmap`, so most code doesn't need to know it is lazy.

If there is no global step in the protocol, skip the spill and write each actor directly. That case is pure streaming.

### Phase 2: the barrier

Run `TTVSplitStep.apply` on the full list of handles. Its logic doesn't change: `_strata`, `_stratum_key` and `_split_stratum` only touch metadata and list positions, and the seeded RNG sees the same list in the same order, so the split is identical.

Two details:

- Ordering. The list order must match today's loader order (sources in order, then suffix groups, then `rglob` order) or the shuffle changes. If Phase 1 runs in parallel, sort results back into input order before the barrier.
- `_hide_markers`. `np.zeros_like` on a memmap allocates a full array per test trial. Replace the eager copy with a lazy flag, for example a `hide_markers: bool = False` field on `DatasetPartition` that the writer applies per trial while writing. `server.test` and `client.test` then share the same handles.

### Phase 3: tail steps and export

Steps after the barrier (for example `build_epochs` on a `Dataset`) are per-actor in practice, so apply them per trial at write time rather than to the whole `Dataset` up front. The simplest form: the writer iterates a partition's handles, runs the tail steps on each trial, and writes the result.

The writer must stop building one big `Raw`. There are two options:

- Same output as today (one FIF per partition). Write each trial to a temporary `*_raw.fif` with `RawArray(...).save()`, then open them all with `mne.io.read_raw_fif(path, preload=False)`, `mne.concatenate_raws` the lazy raws, and `.save()` the result. MNE reads and writes non-preloaded data in buffer-sized chunks, so the partition is never fully in RAM. Delete the temporaries afterwards.
- Simpler output (one FIF per trial, or per subject) inside the existing zip. It's cheaper, but downstream readers must accept a folder of files instead of one.

Prefer the first option unless the consumers are already being changed.

Finally, remove the cache folder in a `finally` block so a crash doesn't leave gigabytes behind (or keep it behind a `keep_cache` flag for debugging).

## Interface changes

### `PruneStep`

```python
@dataclass
class PruneStep[InT: DataActor, OutT: DataActor](ABC):
    name: ClassVar[str]
    scope: ClassVar[Literal["actor", "global"]] = "actor"

    def apply_one(self, actor: InT) -> list[OutT]:
        return self.apply([actor])

    @abstractmethod
    def apply(self, actors: list[InT]) -> list[OutT]: ...
```

- Default `scope` is `"actor"`. Only `TTVSplitStep` sets `scope = "global"`.
- `apply_one` defaults to calling `apply` with one element, so no existing step has to change to work. Steps can override it later for clarity.
- `apply(list)` stays, so `custom_steps` and the current tests keep working.
- A step that is wrongly marked `"actor"` but really needs the whole list will silently give a different result. Add a test per step (see Testing) rather than trusting the flag.

### `DatasetPruner`

- `run` keeps its signature for the in-memory path. Add a streaming path, for example `run_and_export(protocol_name)`, since a streaming run can't return every actor in RAM anyway.
- New private helpers: `_split_steps(steps) -> (head, barrier, tail)`, `_stream_head(...) -> list[TrialData]` (handles), `_spill(trial) -> TrialData`.
- One run timestamp shared by the cache folder and the writers (today it is created in `export`).

### `Loader`

- Add `iter_load`; `load` wraps it.

### Exporter and writers

- `Exporter.export` should accept an iterable, not only a list, and must not materialise it. Today it filters `actors` per writer, which consumes an iterator on the first writer. Either group by `actor.name` in one pass or require that there is one `Dataset` actor (true today).
- `DatasetWriter._write_partition` writes per trial and concatenates lazily, as described above.
- `DatasetPartition.mne_dataset` can stay for small in-memory use, but the writer should not call it.

### Class diagram

Update `docs/diagrams/class.puml`: add `scope` and `apply_one` on `PruneStep`, `iter_load` on `Loader`, the new `DatasetPruner` methods, and `hide_markers` on `DatasetPartition`.

## Using the available RAM

Phase 1 is embarrassingly parallel across files. Run it with `concurrent.futures.ProcessPoolExecutor`, with `max_workers = floor(ram_budget / peak_per_file)`. Measure `peak_per_file` on the largest `.xdf` (for example with `tracemalloc` or `/usr/bin/time -l`) and leave some headroom. Expose the budget or the worker count as a parameter instead of hardcoding it.

Points to watch:

- Workers must return handles (paths and metadata), never arrays, or the pickling cost and memory come back.
- Steps and loaders must be picklable. They are plain dataclasses today, which is fine.
- Preserve input order (`executor.map` does this) because the barrier depends on it.

## Known risks

- Step order in the protocol matters. If `build_epochs` runs before `ttv_split`, each epoch becomes its own unit in the split. The design handles that (more handles, still small), but it is a different split from splitting whole trials. Check which one the protocol intends.
- `ProtocolDefinerStep` mutates actors in place. It works with streaming, but it should use `copy_with` like the other steps.
- Memory-mapped arrays are read-only. Any step that writes into an input array in place will fail with an error, which is the desired outcome, but check the head steps that run after spilling (none today, since the spill happens after the head).
- Disk space. The cache needs roughly the size of the processed signals. Check free space up front or document it.
- Float precision. `np.save` keeps dtype exactly, so the output is bit-identical. Don't convert dtypes when spilling.

## Testing

1. Equivalence. On a small fixture dataset, run the old in-memory path and the new streaming path with the same protocol and seed, and compare the exported FIF data (`get_data()`), channel names, sfreq and partition membership. This is the main acceptance test.
2. Per-step scope check. For every step marked `"actor"`, assert `step.apply(xs) == [y for x in xs for y in step.apply_one(x)]` on a few actors. This catches a step that secretly needs the whole list.
3. Memory. A test or script that runs the pipeline on synthetic large files and asserts peak RSS stays under a bound (`resource.getrusage` or `tracemalloc`). Mark it slow.
4. Cleanup. The cache folder is gone after both a successful run and a run that raises.
5. Ordering. With parallel workers, the barrier receives handles in input order.

## Suggested order of work

1. Add `scope` and `apply_one` to `PruneStep`, mark `TTVSplitStep` as global, and add the per-step scope test.
2. Add `Loader.iter_load`.
3. Add the streaming path in `DatasetPruner` (head, spill, barrier) without changing export, and pass the equivalence test on `run` output.
4. Make `hide_markers` lazy and rewrite `DatasetWriter` to write per trial and concatenate lazily. Pass the equivalence test on exported files.
5. Add parallel Phase 1 with a RAM budget parameter.
6. Update `class.puml` and `scripts/01_prune_dataset/prune_archive_raw.py`.
