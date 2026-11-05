# Tests

The suite checks the **interfaces** of each layered sub-package, not the logic of each implementation. When you add a `PruneStep`, `DataActor`, `FileLoader`, `Writer` or `DownloadStrategy`, the contract tests pick it up automatically (from the registries or from `__subclasses__`) and tell you which rule it breaks.

```bash
poetry run pytest tests/contracts tests/download tests/prune   # contract suite
poetry run pytest                                              # everything, legacy tests included
```

## Layout

| Path | What |
|---|---|
| `conftest.py` | Shared fixtures. `register_temporarily(registry, key, cls)` registers a fake for one test and removes it afterwards. Being the root conftest also puts `tests/` on `sys.path`, so `contracts` imports as a package. |
| `contracts/discovery.py` | Find implementations: `registry_keys`, `registered_type`, `registry_entries` (work with `bmiemg.utils.Registry` and with registries keeping their own dict), `temporary_entry`, `all_subclasses`, `leaf_subclasses`, `generic_arg` (the type argument a class passed to a generic base, e.g. the spec of a `DownloadStrategy[Spec]`). |
| `contracts/assertions.py` | Shape checks with explicit messages: `assert_is_dataclass`, `assert_name_classvar` (optionally equal to the registry key), `assert_implements` (subclass, nothing abstract left, overridden methods keep the base's parameter names and kinds). |
| `contracts/registry_contract.py` | `RegistryContract`: register/get round trip, `register` returns the class, duplicate key raises `ValueError`, unknown key raises `ValueError` listing the known keys. |
| `contracts/samples.py` | `sample_for` (looks up a per-key sample, failing with "add a sample config for <key> to <file>"), `sentinel_instance` (a dataclass instance with placeholder fields, for structural checks), `init_field_names`. |
| `contracts/test_registry_base.py` | The shared `Registry` itself, including per-family isolation. |
| `download/`, `prune/` | One folder per sub-package that applies the contracts to its families. |
| `test_epn612_*.py`, `test_cnn_pipeline.py` | Legacy tests, outside this pattern. |

## Contracts per family

- **Registries** (`StepRegistry`, `FileLoaderRegistry`, `WriterRegistry`, download `Registry`): `RegistryContract`, and prune families do not leak entries into each other.
- **PruneStep** (each registered key): dataclass, implements `apply` with the base signature, `name` equals the key, `from_config(sample)` builds it, sample keys are dataclass fields, `apply([])` returns a list.
- **DataActor** (each leaf subclass): dataclass, `name` ClassVar, `copy_with` returns a new instance of the same type with the change applied and the original untouched; names are unique.
- **FileLoader** (each key): key is a lowercase suffix (`Loader` matches `Path.suffix.lower()`), implements `load_actors`, builds without arguments.
- **Writer** (each key): dataclass, implements `write`, `name` equals the key, `actor` names a known `DataActor.name`, `from_config(sample, timestamp)` builds it.
- **DownloadStrategy** (each key): implements `validate` and `fetch`, builds without arguments, is declared as `DownloadStrategy[Spec]` with `Spec` a dataclass whose `source` equals the key.
- **Services**: `Downloader.run` resolves the strategy from `spec.source` and calls `validate` then `fetch`, unknown sources raise. `DatasetPruner` runs `custom_steps` in order and does not mutate its input. Both use fakes, never the network or data files.

Tests state the correct behaviour only. A contract violation is fixed in the code, never marked as expected with `xfail`.

## Adding an implementation

Register it as usual. Run the contract suite. The only file you may need to edit is the sample table of its sub-package (for prune: `prune/prune_samples.py`, one YAML-shaped config per step or writer key); a missing entry fails with a message saying so.

## Plugging in a new sub-package

1. Create `tests/<subpackage>/` (no `__init__.py`; use file names unique across the suite, since pytest imports them by basename).
2. Import the sub-package at the top of the test module so its adapters register.
3. For each registry family, subclass `RegistryContract`, set `registry` and implement `make_entry` to return a fresh, no-arg constructible probe class.
4. For each interface, write one test class parametrized over `registry_keys(XRegistry)` (or `leaf_subclasses(Base)` for unregistered families) and assert the rules with `contracts.assertions`. Test only what the interface promises: signatures, identity ClassVars, construction from config, return types, immutability.
5. If construction needs data, add a `<subpackage>_samples.py` table and read it with `sample_for` so a new key fails with a clear message.
6. Test the service with fakes defined in `<subpackage>_fakes.py` or in the test module, registered via `register_temporarily`.
7. If a rule is generic enough for other sub-packages, move its helper into `contracts/`.
