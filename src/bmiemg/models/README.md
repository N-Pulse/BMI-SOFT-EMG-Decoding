# Models (proposal)
> **Status: proposal, nothing here is implemented.** This README describes the target architecture for refactoring the legacy `bmiemg.models` code. Classes are drawn as `<<planned>>` in [`docs/diagrams/class.puml`](../../../docs/diagrams/class.puml).

Trains and evaluates EMG decoders on the dataset exported by [`bmiemg.prune`](../prune/README.md). It is agnostic to the model family (scikit-learn classical models, the PyTorch Temporal CNN of Atzori et al. 2016, future ones) and to the evaluation method (metrics, grouping schemes, report formats). Everything is driven by YAML configs and plugged in through registries, like prune steps.

Out of scope:
- Signal processing (filtering, notch, normalisation, resampling): prune protocols own it. A preprocessing variant is a prune protocol, and a model trains on that protocol's exported dataset.
- Embedded reduction (quantisation, channel silencing, window-size reduction): a separate sub-package that takes the model artifact saved here as its input.

---

## 1. Package split
**Recommendation: one `bmiemg.models` sub-package with two services, `ModelTrainer` and `ModelEvaluator`, sharing one domain layer.**

| Option | For | Against |
|---|---|---|
| One `models` package, two services (recommended) | The two circuits share almost every domain object: `Model`, `WindowConfig`, `ModelArtifact`, `WindowSet`, `Predictions`. Inference must window the data exactly as training did, so they must live in one place. One README, one class-diagram package, one registry per family. | The package is larger than `prune`. Evaluation-only users still install the training dependencies (torch), which they need anyway to load a TCNN. |
| `train` + `evaluation` packages | Smaller packages; evaluation could in principle run without training code. | `evaluation` would import `train.domain` for `Model`, `WindowConfig` and `ModelArtifact`, or a third shared package would be needed. Either way the boundary is artificial and every new model touches two packages. |

Dependency direction is one way: `models` reads files written by `prune` (it does not import prune services or adapters) and reuses only prune's `DataActor` base class. Nothing depends on `models` except entry scripts and the future reduction package, which reads `ModelArtifact`.

Cluster batch jobs need a single command per run with no interactive input: `scripts/03_model/train.py --config <name>` and `scripts/04_evaluation/evaluate.py --config <name>`. The two circuits only meet through the artifact folder on disk, so training and evaluation can run as separate jobs, on separate machines, at different times. The package split therefore does not need to follow the job split.

---

## 2. Architecture

| Layer | Folder | Contents |
|---|---|---|
| Service | `service/` | `ModelTrainer` (train circuit) and `ModelEvaluator` (evaluation circuit). They wire the adapters and domain together, with no logic of their own. |
| Domain | `domain/` | Actors (`PartitionData`, `WindowSet`, `Predictions`), interfaces (`Model`, `Metric`, `EvaluationScheme`), their registries and implementations (`SklearnModel`, `TorchModel` and their subclasses), `WindowConfig` and the pure windowing function, `TrainProtocol`, `EvalProtocol`, `DatasetRef`, `ModelArtifact`, `TrainingReport`, `EvaluationReport`. No I/O. |
| Adapter | `adapters/` | `DatasetReader` (prune output), `ConfigLoader` (YAML), `ArtifactStore` (writes and reads artifact folders), `ReportWriter`s. |

Model implementations sit in `domain/`, like prune steps next to `PruneStep`: `fit` and `predict` are computation, even when they wrap scikit-learn or torch. A model never touches disk. It serialises to and from bytes in memory (`to_bytes`, `from_bytes`), and `ArtifactStore` is the only class that writes or reads the files.

### Data actors
All actors subclass prune's `DataActor` (dataclass, `name: ClassVar[str]`, `copy_with`).

| Actor | Produced by | Contents |
|---|---|---|
| `PartitionData` | `DatasetReader` | One partition (`group`: `client`/`server`, `partition_name`: `train`/`val`/`test`): per-trial `signals` `(channels, time)` and `markers`, `subjects`, `channel_names`, `sfreq`. |
| `WindowSet` | Windowing (`WindowConfig`) | Raw windows `x` `(n, channels, time)`, labels `y` (`None` for client test), per-window `subjects`, `class_labels`. Feature extraction for classical models happens inside the model, not here. |

### Windowing
The only transform between prune's dataset and a model. `WindowConfig` is a plain dataclass (YAML `window:` block), not a step family:

| Field | Default | Meaning |
|---|---|---|
| `size_ms` | `200` | Window length |
| `step_ms` | `100` | Hop between window starts (50% overlap) |
| `max_per_trial` | `10` | Evenly spaced subsample when a trial yields more; `null` keeps all |
| `only_labelled` | `true` | Cut windows only inside a trial's movement interval (non-zero `DD`) |
| `label_rule` | `majority` | Window label from the markers it covers: `majority` or `last` |
| `classes` | all | Canonical movements to keep; `one_vs_all` turns one of them into a binary target |

Defaults follow the team's EPN-612 CNN pipeline (`scripts/epn612/dataset.py`). The config is saved in the artifact so inference windows exactly like training.
| `Predictions` | `Model.predict` | `y_pred`, optional `y_proba`, `y_true` when labels exist, `subjects`, the partition it came from. |

### Loading prune's dataset (no TTV split here)
The train/val/test split is prune's `ttv_split`. This package never re-splits. `DatasetReader` reads `data/dataset/<protocol>/{client,server}/<partition>_<timestamp>_raw.fif`, picking the run given by `DatasetRef.timestamp` (`latest` by default). It splits the concatenated Raw back into trials on the `BAD boundary` annotations `mne.concatenate_raws` inserts, and reads markers from the `STI 014` stim channel.

| Partition | Labels | Used by |
|---|---|---|
| `client/train` | yes | train circuit, `fit` |
| `client/val` | yes | train circuit (validation, early stopping), evaluation during development |
| `client/test` | zeroed | evaluation circuit, blind predictions |
| `server/test` | yes | evaluation circuit, server-side scoring of the client test predictions |

---

## 3. The two circuits

### TRAIN: dataset → model artifact + training report
`ModelTrainer().run("<train_config>")`

1. `ConfigLoader` builds a `TrainProtocol`, resolving every key through its registry. Unknown keys or bad parameters fail here.
2. Seed python, numpy and torch with `seed`.
3. `DatasetReader` reads `client/train` and `client/val` into `PartitionData`.
4. Windowing turns each `PartitionData` into a `WindowSet` with the config's `WindowConfig`.
5. `Model.fit(train, val)` returns a `TrainingReport` (history, validation scores, fit time, parameter count, size).
6. `ArtifactStore.save` writes the artifact folder (below), taking the weights from `Model.to_bytes()`, then report writers write the training report next to it.
7. The user uploads the folder to the MAKE Archive and records its path (SDR §3.2.4). An `archive` artifact store can automate this later.

### INFERENCE / EVALUATION: artifact + data → predictions → report
`ModelEvaluator().run("<eval_config>")`

1. `ConfigLoader` builds an `EvalProtocol`.
2. `ArtifactStore.load` reads the weights file and rebuilds the `Model` with `from_bytes` (class found through `ModelRegistry` and the saved `model_key`) and the `WindowConfig` from `artifact.yaml`. The evaluation config cannot change them, which guarantees train/inference parity.
3. `DatasetReader` reads each target partition. Channel names and `sfreq` must match the artifact, otherwise `ValueError`.
4. Windowing → `WindowSet` → `Model.predict` → `Predictions`.
5. For `client/test` (labels zeroed) predictions are written blind, then joined with `server/test` labels for server-side scoring. Labelled targets are scored directly.
6. Each `EvaluationScheme` splits the predictions into groups (`pooled`, `per_subject`, `per_class`). Each `Metric` is computed per group. The result is an `EvaluationReport`, written by the configured report writers.

### Model artifact
The artifact is a self-describing folder, `artifacts/models/<train_config>_<timestamp>/`:

| File | Contents |
|---|---|
| `artifact.yaml` | `ModelArtifact`: `model_key`, `model_params`, `window` (`WindowConfig`), `class_labels`, `channel_names`, `sfreq`, `seed`, `dataset` (`protocol` + `timestamp`), `git_commit`, `created_at`. |
| `config.yaml` | The resolved train config, so the run can be reproduced exactly. |
| `weights.pt` / `model.joblib` | `Model.to_bytes()`, written by `ArtifactStore`. |
| `report/` | Training report files. |

Everything the future reduction stage needs (window and step, channel list, sampling rate, class set, weights) is explicit in `artifact.yaml`, so it can work on the artifact without importing training code beyond `ArtifactStore` and `Model`.

---

## 4. Interfaces and registries
Every family follows the prune pattern: subclass the interface as a dataclass, register it on a `bmiemg.utils.Registry` subclass under a string key, keep `name` equal to the key, import it in the package `__init__.py`. Dataclass fields are the YAML keys.

| Interface | Registry | Method(s) | Planned implementations (key) |
|---|---|---|---|
| `Model` | `ModelRegistry` | `from_config(data, seed)`, `fit(train, val) -> TrainingReport`, `predict(windows) -> Predictions`, `to_bytes() -> bytes`, `from_bytes(data) -> Model` | `svm`, `logistic_regression`, `decision_tree` (`SklearnModel`, with a `features` field: `rms`, `mav`, `wl`, `zc`, `ssc`), `tcnn` (`TorchModel`, Atzori 2016) |
| `Metric` | `MetricRegistry` | `score(predictions)` | `accuracy`, `balanced_accuracy`, `cohen_kappa`, `f1` (`average`), `confusion_matrix`, `flips_per_second` |
| `EvaluationScheme` | `SchemeRegistry` | `split(predictions) -> dict[str, Predictions]` | `pooled`, `per_subject`, `per_class` |
| `ReportWriter` | `ReportWriterRegistry` | `from_config`, `write(report, dest_dir)` | `json_metrics`, `predictions_csv`, `confusion_plot`, `training_curves` |

Cross-subject generalisation is already guaranteed by prune's participant-disjoint split, so evaluating on `server/test` with `pooled` is the cross-subject score, and `per_subject` shows its spread. A within-subject protocol would need a different prune protocol, not a new scheme here.

---

## 5. Config shape
Configs live in `configs/model/train/<name>.yaml` and `configs/model/eval/<name>.yaml`. ```yaml
# configs/model/train/tcnn_raw_150ms.yaml
name: tcnn_raw_150ms
seed: 42
dataset: {protocol: raw_archive_v1, timestamp: latest}
window: {size_ms: 150, step_ms: 25, label_rule: majority, classes: [open hand, close hand, pinch]}
model:
  tcnn: {time_kernel: 5, n_filters: [32, 32, 64, 64], epochs: 10, batch_size: 32, learning_rate: 1.0e-3, device: auto}
val_metrics: [accuracy, cohen_kappa, {f1: {average: macro}}]
reports:
  - training_curves: {}
  - json_metrics: {}
```

```yaml
# configs/model/eval/tcnn_raw_150ms.yaml
name: tcnn_raw_150ms_eval
artifact: artifacts/models/tcnn_raw_150ms_<timestamp>
dataset: {protocol: raw_archive_v1, timestamp: latest}
targets: [client/val, client/test]   # client/test is scored against server/test
metrics: [accuracy, cohen_kappa, {f1: {average: macro}}, confusion_matrix]
schemes: [pooled, per_subject]
reports:
  - json_metrics: {}
  - predictions_csv: {}
  - confusion_plot: {}
```

A classical baseline swaps `model:` for e.g. `svm: {C: 1.0, class_weight: balanced, features: [rms, mav, wl, zc, ssc]}`. The preprocessing study (SDR §3.2.5) is one prune protocol per variant (e.g. with and without a notch step); each train config points `dataset.protocol` at one of them.

---

## 6. Usage (target API)

```python
from bmiemg.models import ModelTrainer, ModelEvaluator

artifact = ModelTrainer().run("tcnn_raw_150ms")       # writes artifacts/models/...
report = ModelEvaluator().run("tcnn_raw_150ms_eval")  # writes reports/...
```

---

## 7. Migration from the legacy code

| Legacy | New home | Notes |
|---|---|---|
| `models/Trainer.py` `Trainer` | `service/model_trainer.py` `ModelTrainer` | No cross-validation inside training; validation uses `client/val`. |
| `models/Evaluator.py` `Evaluator.cross_validate` | dropped | K-fold CV replaced by prune's fixed split. Could return later as an `EvaluationScheme` if needed. |
| `models/ModelRegistry.py` `ModelRegistry` | `domain/models/model_registry.py` | Rebuilt on `bmiemg.utils.Registry` with decorator registration. |
| `models/ClassifierEstimator.py` (Protocol) | `domain/models/model.py` `Model` (ABC) | Adds `to_bytes`/`from_bytes` and actor-based `fit`/`predict`. |
| `model_factories/ModelFactory` + `DecisionTreeFactory`, `LogisticRegressionFactory`, `SVMFactory` | `domain/models/sklearn/` `DecisionTreeModel`, `LogisticRegressionModel`, `SvmModel` | Factory parameters become dataclass fields / YAML keys. |
| `models/TrainingConfig.py` | `domain/protocols/train_protocol.py` `TrainProtocol` | `random_state` → `seed`; `n_splits`, `scoring`, `shuffle` dropped. |
| `models/TrainingResults.py` | `ModelArtifact` + `TrainingReport`; `save` → `ArtifactStore` | `predict` moves to `Model`. |
| `models/evaluation.py` `ClassificationEvaluation`, `evaluate_classification` | `Metric` implementations + `EvaluationReport` | Accuracy, balanced accuracy, confusion matrix, report. |
| `target_builders/TargetBuilder`, `OneVsAllTargetBuilder` | `WindowConfig` `label_rule`, `classes`, `one_vs_all` | Labels come from prune's canonical `XXDD` markers. |
| `train.py` `collect_feature_dataset` | windowing + `SklearnModel` `features` | User filtering belongs to prune protocols. |
| `scripts/epn612/dataset.py` `WindowConfig`, `extract_trial_windows` | windowing | Source of the default window settings. |
| `train.py` `_save_results`, `_save_importance_plot` | `json_metrics`, `confusion_plot` report writers | Feature importance: possible `feature_importance` writer for sklearn models. |
| `train.py` `load_config`, `configs/train.yml`, `configs/epn612_svm.yml` | `ConfigLoader`, `configs/model/train/*.yaml` | `test_size`/`val_size` dropped (prune). `param_grids`/nested CV: open question. |
| `eval.py` (empty), `configs/eval.yaml` | `ModelEvaluator`, `configs/model/eval/*.yaml` | |
| `scripts/cnn/cnn.py` `TCNN` | `domain/models/torch/tcnn.py` | Network module behind `TcnnModel`. |
| `scripts/cnn/training.py` `fit`, `run_epoch`, `evaluate`, `save_checkpoint`, `build_data_loaders` | `TorchModel` (`fit`, `predict`, `to_bytes`); checkpoint files → `ArtifactStore` | Generic torch loop shared by future torch models. |
| `scripts/cnn/training.py` `TCNNConfig`, `TrainingConfig`; `scripts/cnn/train.py` `ExperimentConfig`, `seed_everything`, `resolve_device`, `checkpoint_path` | `TcnnModel` fields, `TrainProtocol`, `ModelTrainer` seeding, `TorchModel.device`, `ArtifactStore` | |
| `scripts/evaluation/recog_eval_metrics.py` `prediction_flips_per_second`, `raw_recognition_count` | `flips_per_second` and a `recognition_rate` metric | |
| `scripts/evaluation/recog_eval_metrics.py` `write_predictions` | `predictions_csv` report writer | |
| `scripts/evaluation/recog_eval_offline.py` `model_contract` | `ModelArtifact` fields + contract check in `ModelEvaluator` | |
| `scripts/evaluation/recog_eval_data.py` trial extraction and splitting | dropped | Covered by prune (`marker_unification`, `ttv_split`). |
| `data/epn612` loaders + adaptation | out of scope | EPN612 must go through prune (needs a file loader there) before this package can train on it. |

---

## 8. Open questions
- Prune's `.fif` export does not carry per-trial subject/session ids, which `per_subject` needs. Options: write them as annotations or a sidecar `metadata.json` in `DatasetWriter`.
- Hyperparameter search: separate sweep script over configs, or a `search:` section in the train config?
- MAKE Archive upload: manual, or an `archive` `ArtifactStore` reusing `bmiemg.download` credentials (needs WebDAV upload)?
- Where reports go: inside the artifact folder or a separate `reports/` root.
- Torch as an optional Poetry extra, with `TorchModel` imported lazily so sklearn-only environments still work.
