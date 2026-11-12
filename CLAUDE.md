# CLAUDE.md

Guidance for Claude Code in this repository. kapture is a pivot file format and a Python toolkit for SfM and
sensor-acquired data. The format specification is `kapture_format.adoc`; contributor rules are in `CONTRIBUTING.adoc`
(this file summarizes them, the .adoc files are the reference).

## Commands

- Install (editable, with tests/lint/build deps): `make install_dev` (= `pip install -e ".[dev]"`)
- All tests: `python -m unittest discover -s tests/`
- One test file: `python -m unittest tests.test_extract` (tests are `unittest`, data in `samples/`)
- Lint, as in CI (`.github/workflows/kapture-push-main.yml`):
  - blocking: `flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics`
  - style: `flake8 . --max-complexity=16 --max-line-length=120 --extend-ignore=F401`
- `README.md` is generated from `README.adoc` (asciidoctor + pandoc): edit `README.adoc`, never `README.md`.
- Python >= 3.10 (CI: 3.10, 3.12, 3.13).

## Architecture

Layers, dependencies go downward only (`tools` -> `algo`/`converter` -> `io` -> `core`; `utils` usable everywhere):

- `kapture/core/` — in-memory data objects: `Kapture` (root), `Sensors`/`Camera`, `Rigs`, `Trajectories`,
  `Records*`, `Keypoints`/`Descriptors`/`GlobalFeatures`, `Matches`, `Observations`, `Points3d`, `PoseTransform`.
  One CamelCase module per class. Everything is re-exported by `kapture/__init__.py`: use `kapture.Trajectories`,
  not `kapture.core.Trajectories.Trajectories`. No disk access here.
- `kapture/io/` — reading/writing on disk.
  - `csv.py`: `kapture_from_dir` / `kapture_to_dir`, and per object `<object>_from_file` / `<object>_to_file`.
  - `records.py`: record data file paths (`get_record_fullpath`, `get_image_fullpath`, ...), depth/range map
    files, and `TransferAction` (defined in `binary.py`: how data files are copied/linked/moved/skipped).
  - `features.py`, `binary.py`, `tar.py`, `ply.py`, `image.py`, `structure.py`.
- `kapture/algo/` — processing on `Kapture` objects: `merge_*`, `compare`, `extract`, `pose_operations`.
- `kapture/converter/<format>/` — conversion from/to other formats: `import_<format>.py`, `export_<format>.py`,
  shared code in `<format>_commons.py`. No command line code here.
- `kapture/tools/` — command line tools, thin wrappers that parse arguments and call `algo`/`converter`.
  No business logic: put it in `algo` or `converter` so it is importable and testable.
- `kapture/utils/` — generic helpers: `cli.py` (`run_cli`, `setup_logging`, `Verbosity`, `DEFAULT_VERBOSITY`),
  `paths.py` (`safe_remove_file`, `path_secure`, ...), `logging.py`, `computation.py`, `upgrade.py`.
- `tests/test_<subject>.py` — one file per subject; build small kaptures in code or use `samples/`.
- `doc/` — `tutorial.adoc`, `datasets.adoc`, `installation.adoc`, `changelog_2.0.adoc` (breaking changes of 2.0).

### Adding a command line tool

1. `kapture/tools/kapture_<verb>_<object>.py`: a type-hinted function with a reST docstring (each `:param:` becomes
   the help of its flag), then `def main(): run_cli(<function>, aliases={...})`. Never build an argparse parser.
   Model: `kapture/tools/kapture_extract_period.py`.
2. Logic in `kapture/algo/` or `kapture/converter/<format>/`.
3. Entry point in `pyproject.toml` `[project.scripts]`: `kapture_<...>="kapture.tools.kapture_<...>:main"`.
4. Test in `tests/`; user visible or breaking change noted in `doc/` (changelog, tutorial).

## Naming rules

**Tools and modules**
- Tool file and console script: `kapture_<verb>_<object>`. Verbs in use: `import`, `export`, `extract`, `merge`,
  `print`, `upgrade`, `download`, `rename`. Dataset/format names keep their usual case
  (`kapture_import_RobotCar_Seasons`, `kapture_export_LTVL2020`).
- The tool function is named without the `kapture_` prefix (`extract_period`); the library function it calls
  keeps it (`kapture.algo.extract.kapture_extract_period`).
- Converter modules: `import_<format>.py` / `export_<format>.py`.

**Command line parameters** (= function parameters = `--config` YAML keys)
- snake_case, short, and identical across tools: `input` / `inputs`, `output`, `force`, `image_transfer`
  (or `transfer` when not only images), `verbose`, `quiet`. Reuse an existing name before inventing one.
- Short flags `-i`, `-o`, `-v`, `-q`, `-f`/`-y` come from `COMMON_ALIASES`; other short flags and legacy long flags
  go in the tool `aliases`. Booleans are flags (`--force` / `--no_force`), choices are `Literal[...]`, lists are
  space separated.
- Library functions use explicit path names instead: `input_path`, `output_path`, `kapture_dirpath`.

**Functions**
- I/O: `<object>_from_file` / `<object>_to_file`, `<object>_from_dir` / `<object>_to_dir`, `get_<x>_fullpath`.
- In-place mutation: suffix `_inplace` (`trajectory_transform_inplace`, `rigs_remove_inplace`); without the suffix,
  return a new object and leave the input untouched.
- Module private helpers and private members start with `_`.

**Variables**
- Explicit names, no acronyms (`sensor_id`, not `sid`; `kapture_data`, not `kd`).
- Paths: `*_filepath` / `*_dirpath` in new code. The codebase also has `*_file_path` / `*_dir_path`; do not
  rename existing ones just for that.
- Poses: `<target>_from_<source>` (`world_from_device`, `rig_from_device`, `query_from_refer`), composed as
  `a_from_c = PoseTransform.compose([a_from_b, b_from_c])`. Kapture trajectories store `device_from_world`
  (world to device, as in colmap). Quaternions are `[qw, qx, qy, qz]`.
- Timestamps are integers (usually microseconds); say the unit in the name or docstring when it is not obvious
  (`timestamp_us`, `time_unit`).
- Core classes: plural for containers (`Trajectories`, `RecordsCamera`, `Sensors`), singular for items
  (`Sensor`, `RecordWifi`, `PoseTransform`).

## Code conventions

- File header: `# Copyright <year>-present NAVER Corp. Under BSD 3-clause license`, preceded by
  `#!/usr/bin/env python3` for tools and tests, then a module docstring.
- PEP-8 with 120 columns; type hints on all arguments and return types; reST (sphinx) docstrings
  (`:param x:`, `:return:`).
- Logging: library modules use a named logger (`logging.getLogger('<format_or_module>')`, or `__name__`) and never
  configure handlers; tools call `setup_logging(logger, verbose, quiet)`.
- Never overwrite user data silently: check paths and honor `force` (`kapture.utils.paths.safe_remove_*`).

## Git

- Work on `dev`; features on `feat/<name>` branches merged into `dev` with a merge commit
  (`Merge branch 'feat/<name>': <summary>`).
- Commit messages: short, lowercase, imperative; prefix with the touched module or tool when relevant
  (`add kapture_extract_area: ...`, `export_colmap: do not crash when ...`).
