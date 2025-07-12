<div align="center">
  <img src="images/banner.png" alt="Banner Image" />
</div>


<p align="center">
<a href="https://github.com/DenverCoder1/readme-typing-svg"><img src="https://readme-typing-svg.herokuapp.com?font=Time+New+Roman&color=%231c88e3&size=25&center=true&vCenter=true&width=600&height=30&lines=👋+Welcome!"></a>
</p>

<p align="center">
  <a href="https://github.com/MatheusFS-dev/araras"><img src="https://img.shields.io/github/license/MatheusFS-dev/araras?style=flat-square" alt="License"/></a>
  <a href="https://github.com/MatheusFS-dev/araras/stargazers"><img src="https://img.shields.io/github/stars/MatheusFS-dev/araras?style=flat-square" alt="Stars"/></a>
  <a href="https://github.com/MatheusFS-dev/araras/network/members"><img src="https://img.shields.io/github/forks/MatheusFS-dev/araras?style=flat-square" alt="Forks"/></a>
  <a href="https://visitor-badge.laobi.icu/badge?page_id=MatheusFS-dev.araras"><img src="https://visitor-badge.laobi.icu/badge?page_id=MatheusFS-dev.araras" alt="Visitors"/></a>
</p>

<p align="center">
  <a href="#">
      <img src="https://api.visitorbadge.io/api/VisitorHit?user=MatheusFS-dev&repo=araras&countColor=%23007FFF" />
   </a>
</p>

This is a python module that provides a set of tools for working with machine learning models. It includes utilities for neural architecture search using optuna, builders and helpers for keras/tensorflow, a monitoring system for the kernel, and several other features. The module is designed to be easy to use and flexible, allowing users to customize their machine learning workflows.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Features](#features)
- [📚 API Reference](#-api-reference)
- [⚙️ Installation Instructions](#️-installation-instructions)
  - [📌 Prerequisites](#-prerequisites)
  - [🪜 Steps](#-steps)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [🤝 Collaborators](#-collaborators)

## 📚 API Reference

### `email.utils`

#### `send_email(subject: str, body: str, recipients_file: str, credentials_file: str, text_type: str = "plain", smtp_server: str = "smtp.gmail.com", smtp_port: int = 587) -> None`

Send an email to the recipients listed in ``recipients_file`` using the Gmail SMTP server.

**Parameters**
- **subject** – email subject line.
- **body** – message body in plain text or HTML.
- **recipients_file** – path to the JSON file containing target addresses.
- **credentials_file** – JSON file with ``email`` and ``password`` fields.
- **text_type** – ``"plain"`` or ``"html"`` body encoding.
- **smtp_server** – SMTP host name.
- **smtp_port** – port number for the SMTP server.

**Example**
```python
from araras.email.utils import send_email
send_email("Run finished", "All tasks completed", "recipients.json", "creds.json")
```

### `keras.analysis.estimator`

#### `model_param_distribution(build_model_fn: Callable[[optuna.Trial], tf.keras.Model], bits_per_param: int, batch_size: int = 1, n_trials: int = 1000) -> None`

Randomly samples models using ``build_model_fn`` and plots histograms of the parameter count, model size and estimated training memory.

**Parameters**
- **build_model_fn** – function that builds a model from an Optuna trial.
- **bits_per_param** – number of bits used per weight.
- **batch_size** – batch size for memory estimation.
- **n_trials** – number of random trials.

**Example**
```python
from araras.keras.analysis.estimator import model_param_distribution
model_param_distribution(make_model, bits_per_param=32)
```

### `keras.analysis.profiler`

#### `get_flops(model: tf.keras.Model, batch_size: int = 1) -> int`
Return the FLOPs required for a single forward pass.

#### `get_macs(model: tf.keras.Model, batch_size: int = 1) -> int`
Return the number of MAC operations for a forward pass (``1 MAC = 2 FLOPs``).

#### `get_memory_and_time(model: tf.keras.Model, batch_size: int = 1, device: int = 0, warmup_runs: int = 10, test_runs: int = 50, verbose: bool = True) -> Tuple[int, float]`
Measure peak memory usage and average inference time on ``device``.

### `keras.builders`

#### `cnn.build_cnn1d(trial, kparams, x, filters_range, kernel_size_range, filters_step=10, kernel_size_step=1, use_batch_norm=True, trial_kernel_reg=False, trial_bias_reg=False, trial_activity_reg=False, strides=1, dilation_rate=1, groups=1, use_bias=False, padding="same", data_format="channels_last", kernel_initializer=initializers.GlorotUniform(), bias_initializer=initializers.Zeros(), name_prefix="cnn1d") -> layers.Layer`
Create a Conv1D block with optional Optuna tuned parameters.

#### `cnn.build_dense_as_conv1d(trial, kparams, x, filters_range, filters_step=10, padding="valid", trial_kernel_reg=False, trial_bias_reg=False, trial_activity_reg=False, name_prefix="dense_as_conv1d") -> layers.Layer`
Simulate a dense layer using ``Conv1D(kernel_size=1)``.

#### `cnn.build_cnn2d(trial, kparams, x, filters_range, kernel_size_range, filters_step=10, kernel_size_step=1, use_batch_norm=True, trial_kernel_reg=False, trial_bias_reg=False, trial_activity_reg=False, strides=(1,1), dilation_rate=(1,1), groups=1, use_bias=False, padding="same", data_format="channels_last", kernel_initializer=initializers.GlorotUniform(), bias_initializer=initializers.Zeros(), name_prefix="cnn2d") -> layers.Layer`
Create a Conv2D block with optional tuning and regularisation.

#### `cnn.build_dense_as_conv2d(...) -> layers.Layer`
Dense layer emulation via ``Conv2D(kernel_size=(1,1))``.

#### `cnn.build_cnn3d(...) -> layers.Layer`
3‑D convolution block with Optuna search support.

#### `cnn.build_dense_as_conv3d(...) -> layers.Layer`
Dense layer emulation via ``Conv3D(kernel_size=(1,1,1))``.

#### `dnn.build_dnn(trial, kparams, x, units_range, dropout_rate_range, units_step=10, dropout_rate_step=0.1, kernel_initializer=initializers.GlorotUniform(), bias_initializer=initializers.Zeros(), use_bias=True, use_batch_norm=False, trial_kernel_reg=False, trial_bias_reg=False, trial_activity_reg=False, name_prefix="dnn") -> layers.Layer`
Build a fully connected block with optional batch norm and dropout.

#### `gnn.build_grid_adjacency(rows: int, cols: int) -> tf.sparse.SparseTensor`
Return a 4‑neighbour grid adjacency matrix suitable for GCN layers.

#### `gnn.build_knn_adjacency(rows: int, cols: int, k: int) -> tf.sparse.SparseTensor`
Construct a k‑nearest neighbour adjacency matrix on a 2‑D grid.

#### `gnn.build_gcn(trial, kparams, x, a_graph, units_range, dropout_rate_range, units_step=10, dropout_rate_step=0.1, ... ) -> layers.Layer`
Build a Graph Convolutional Network block.

#### `gnn.build_gat(...) -> layers.Layer`
Graph Attention Network layer with tunable heads and dropout.

#### `gnn.build_cheb(...) -> layers.Layer`
Chebyshev graph convolution layer.

#### `lstm.build_lstm(...) -> layers.Layer`
Create an LSTM block with optional regularisation and batch norm.

#### `se.build_squeeze_excite_1d(x, trial, kparams, ratio_choices, name_prefix="se_block") -> tf.keras.layers.Layer`
Apply a squeeze‑and‑excitation block on 1‑D inputs.

#### `skip.trial_skip_connections(trial, layers_list, axis_to_concat=-1, print_combinations=False, strategy="final", merge_mode="concat") -> tf.Tensor`
Conditionally merge intermediate tensors via skip connections.

#### `tcnn.build_tcnn1d(...) -> layers.Layer`
1‑D transposed convolution block with optional tuning.

#### `tcnn.build_tcnn2d(...) -> layers.Layer`
2‑D transposed convolution block with optional tuning.

#### `tcnn.build_tcnn3d(...) -> layers.Layer`
3‑D transposed convolution block with optional tuning.

### `keras.callbacks`

#### `NanLossPrunerOptuna(trial: optuna.Trial)`
Keras callback that prunes the Optuna trial if training loss becomes ``NaN``.

#### `get_callbacks_study(trial: optuna.Trial, tensorboard_logs: str | None = None, monitor: str = "val_loss") -> List[tf.keras.callbacks.Callback]`
Return common callbacks for Optuna model trials.

#### `get_callbacks_model(backup_dir: str, tensorboard_logs: str) -> List[tf.keras.callbacks.Callback]`
Return a set of callbacks for regular model training runs.

### `keras.hyperparams`

#### `class KParams`
Container defining the default search spaces for activations, regularisers, optimisers, scalers and initialisers.
Provides ``set_*`` and ``get_*`` methods to customise sampling, ``default()`` for standard presets and ``full_search_space()`` to enable every available option.

### `keras.utils`

#### `convert_to_saved_model(input_keras_path: str, output_zip_path: str) -> None`
Convert a ``.keras`` model archive into a zipped TensorFlow SavedModel.

#### `capture_model_summary(model: tf.keras.Model) -> str`
Return the summary of ``model`` as a single string.

#### `punish_model_flops(target, model, penalty_factor=1e-10, direction="minimize")`
Add a penalty proportional to FLOPs to ``target``.

#### `punish_model_params(target, model, penalty_factor=1e-9, direction="minimize")`
Add a penalty proportional to the parameter count.

#### `punish_model(target, model, type=None, flops_penalty_factor=1e-10, params_penalty_factor=1e-9, direction="minimize")`
Apply FLOPs or parameter penalties in a single call.

### `kernel.monitoring`

#### `run_auto_restart(file_path: str, success_flag_file: str = "/tmp/success.flag", title: str | None = None, max_restarts: int = 10, restart_delay: float = 3.0, recipients_file: str | None = None, credentials_file: str | None = None, restart_after_delay: float | None = None, retry_attempts: int | None = None, supress_tf_warnings: bool = False, resource_usage_log_file: str | None = None) -> None`
Execute ``file_path`` under a restart manager that emails on crashes and optionally restarts after a delay.

### `optuna.analysis.analyzer`

#### `class PlotConfig`
Dataclass storing matplotlib sizing and font parameters used by the plotting utilities.

#### `analyze_study(study: optuna.Study, table_dir: str, top_frac: float = 0.2, param_name_mapping: Dict[str, str] | None = None, create_standalone: bool = False, save_data: bool = False, plots: List[str] | None = None) -> None`
Perform a full analysis of an Optuna study and save figures and tables to ``table_dir``.

### `optuna.keras.stats`

#### `get_model_stats(trial: optuna.Trial, model: tf.keras.Model, bits_per_param: int, batch_size: int, n_trials: int = 10000, device: int = 0, verbose: bool = False) -> Dict[str, float]`
Extract model statistics and attach them as user attributes of ``trial``.

### `optuna.callbacks`

#### `ImprovementStagnation(min_n_trials: int = 5, window_size: int = 5, variance_threshold: float = 1e-10, improvement_evaluator: Optional[BaseImprovementEvaluator] = None, verbose: bool = False)`
Stop the study when recent improvement variance drops below ``variance_threshold``.

#### `StopIfKeepBeingPruned(threshold: int)`
Halt optimisation if ``threshold`` consecutive trials are pruned.

### `optuna.utils`

#### `supress_optuna_warnings() -> None`
Ignore warnings emitted by Optuna experimental features.

#### `get_remaining_trials(study: optuna.Study, num_trials: int) -> int`
Return how many trials are left to reach ``num_trials``.

#### `cleanup_non_top_trials(all_trial_ids: Set[int], top_trial_ids: Set[int], cleanup_paths: List[Tuple[str, str]]) -> None`
Remove files or directories associated with trials that are not in ``top_trial_ids``.

#### `rename_top_k_files(top_trials: List[optuna.Trial], file_configs: List[Tuple[str, str]]) -> None`
Rename trial files by ranking.

#### `save_trial_params_to_file(filepath: str, params: Dict[str, float], **kwargs: str) -> None`
Write parameter values and metadata to ``filepath``.

#### `get_top_trials(study: optuna.Study, top_k: int, rank_key: str = "value", rank_descending: bool = True) -> List[optuna.Trial]`
Return the best ``top_k`` trials ordered by ``rank_key``.

#### `save_top_k_trials(top_trials: List[optuna.Trial], args_dir: str, study: optuna.Study, extra_attrs: List[str] | None = None) -> None`
Save parameters of the top trials to text files.

#### `init_study_dirs(run_dir: str, study_name: str = "optuna_study", subdirs: List[str] | None = None) -> Tuple[str, ...]`
Create an experiment folder structure and return all paths.

### `plot.configs`

#### `config_plt(style: str = 'single-column') -> None`
Configure ``matplotlib`` for single or double column figures.

### `tensorflow.model`

#### `get_model_usage_stats(saved_model: str | tf.keras.Model, n_trials: int = 100000, device: int = -1, rapl_path: str = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj", verbose: bool = True) -> Tuple[float, float, float]`
Estimate average inference time, power draw and energy consumption for ``saved_model``.

### `utils.dir`

#### `create_run_directory(prefix: str, base_dir: str = "runs") -> str`
Create a unique run directory under ``base_dir`` and return its path.

### `utils.gpu`

#### `get_user_gpu_choice() -> str`
Prompt the user to select a GPU index interactively.

#### `get_gpu_info() -> None`
Print a detailed GPU report.

### `utils.logs`

#### `log_resources(log_dir: str, interval: int = 5, **kwargs) -> None`
Periodically log CPU, RAM, GPU and TensorFlow usage statistics to ``log_dir``.

### `utils.misc`

#### `clear()`
Clear the current terminal or notebook output.

#### `format_number(number, precision=2)`
Return a human readable string with SI prefixes.

#### `format_bytes(bytes_value, precision=2)`
Format a byte count using binary units.

#### `format_scientific(number, max_precision=2)`
Format a number using scientific notation.

#### `format_number_commas(number, precision=2)`
Return the number with comma separators.

#### `NotebookConverter.convert_notebook_to_python(notebook_path: Path) -> Path`
Convert a Jupyter notebook into a Python script file.

### `utils.terminal`

#### `SimpleTerminalLauncher(supress_tf_warnings: bool = False)`
Launch commands in a new terminal window and capture the PID.


## ⚙️ Installation Instructions

To set up the development environment, follow these steps:

### 📌 Prerequisites

- Optuna
- Tensorflow
- Keras
- pandas
- Git

### 🪜 Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/MatheusFS-dev/araras.git
   ```

2. **Import the modules as you need**

## 🤝 Contributing

Contributions are what make the open-source community amazing. To contribute:

1. Fork the project.
2. Create a feature branch (`git checkout -b feature/new-feature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Open a Pull Request.

## 📜 License

This project is licensed under the **[General Public License](LICENSE)**.

## 🤝 Collaborators

We thank the following people who contributed to this project:

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/MatheusFS-dev" title="Matheus Ferreira">
        <img src="https://avatars.githubusercontent.com/u/99222557" width="100px;" alt="Foto do Matheus Ferreira no GitHub"/><br>
        <sub>
          <b>Matheus Ferreira</b>
        </sub>
      </a>
    </td>
  </tr>
</table>

