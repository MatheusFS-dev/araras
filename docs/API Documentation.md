# API Documentation


## email

### send_email

```python
send_email(subject: str, body: str, recipients_file: str, credentials_file: str, text_type: str = "plain", smtp_server: str = "smtp.gmail.com", smtp_port: int = 587)  [source]
```
Sends an email notification with the specified subject and body to all addresses listed in a JSON file.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| subject | `str` | The subject line for the email. |
| body | `str` | Body content of the message. |
| recipients_file | `str` | Path to a JSON file containing a list of recipient addresses. |
| credentials_file | `str` | Path to a JSON file with the sender email and password. |
| text_type | `str` | Content type such as `"plain"` or `"html"`. |
| smtp_server | `str` | SMTP server hostname. |
| smtp_port | `int` | Port number used for the SMTP connection. |

**Returns**

`None` – The email is sent and no value is returned.

**Raises**

- `ValueError` – If the credential or recipients files cannot be read.

**Examples**

```python
# Example 1: basic usage
send_email("Hello", "Test", "recipients.json", "credentials.json")

# Example 2: with optional parameters
send_email("Hi", "<b>HTML</b>", "recipients.json", "credentials.json", text_type="html")
```


## keras

### callbacks.NanLossPrunerOptuna

```python
class NanLossPrunerOptuna(callbacks.Callback)
```
Stops an Optuna trial when the training loss becomes NaN.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| trial | `optuna.Trial` | Trial object used for reporting and pruning. |

**Raises**

- `optuna.exceptions.TrialPruned` – Raised when a NaN loss is detected at the end of an epoch.

**Examples**

```python
model.fit(..., callbacks=[NanLossPrunerOptuna(trial)])
```

> [!TIP]
> Combine this callback with Optuna's `KerasPruningCallback` for early stopping of bad trials.

### callbacks.get_callbacks_study

```python
get_callbacks_study(trial: optuna.Trial, tensorboard_logs: str | None = None, monitor: str = "val_loss")  [source]
```
Return a list of callbacks commonly used when running Optuna trials.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| trial | `optuna.Trial` | Current trial object. |
| tensorboard_logs | `str | None` | Directory for TensorBoard logs, or `None` to disable. |
| monitor | `str` | Metric to monitor for early stopping and learning rate schedules. |

**Returns**

`List[tf.keras.callbacks.Callback]` – Callbacks configured for a trial.

**Examples**

```python
cbs = get_callbacks_study(trial, "./logs")
model.fit(x, y, callbacks=cbs)
```

### callbacks.get_callbacks_model

```python
get_callbacks_model(backup_dir: str, tensorboard_logs: str)  [source]
```
Return a list of callbacks for regular model training.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| backup_dir | `str` | Directory for saving checkpoints. |
| tensorboard_logs | `str` | Directory for TensorBoard output. |

**Returns**

`List[tf.keras.callbacks.Callback]` – Callback list for training.

**Examples**

```python
cbs = get_callbacks_model("./backup", "./logs")
model.fit(x, y, callbacks=cbs)
```


### hyperparams.KParams

```python
class KParams
```
Container for Keras hyperparameter search spaces.

The dataclass stores mappings of possible activations, regularizers, optimizers, scalers and initializers. Each mapping can be overridden using the provided setter methods and values can be sampled using the `get_*` methods.

**Default search space**

- **Activations**: relu, gelu, silu, elu, sigmoid, tanh, none
- **Regularizers**: none, L2(1e-2)
- **Optimizers**: SGD(momentum=0.9), Adam, AdamW(weight_decay=1e-4), Lion, RMSprop
- **Scalers**: StandardScaler, MinMaxScaler(0–1), MinMaxScaler(-1–1)
- **Initializers**: GlorotUniform

**Full search space**

Includes a wide range of activations, regularizers, optimizers, scalers and initializers as returned by `KParams.full_search_space()`.

> [!WARNING]
> Extremely large search spaces may lead to long optimization times.

#### set_activation_choices(choices)

```python
set_activation_choices(choices: Sequence | Mapping)  [source]
```
Replace the available activation functions.

#### set_regularizer_choices(choices)

```python
set_regularizer_choices(choices: Sequence | Mapping)  [source]
```
Replace the available regularizers.

#### set_optimizer_choices(choices)

```python
set_optimizer_choices(choices: Sequence | Mapping)  [source]
```
Replace the available optimizers.

#### set_scaler_choices(choices)

```python
set_scaler_choices(choices: Sequence | Mapping)  [source]
```
Replace the available data scalers.

#### set_initializer_choices(choices)

```python
set_initializer_choices(choices: Sequence | Mapping)  [source]
```
Replace the available kernel initializers.

#### get_activation

```python
get_activation(trial: optuna.Trial, name: str) -> Callable | None  [source]
```
Sample an activation function for a trial.

#### get_regularizer

```python
get_regularizer(trial: optuna.Trial, name: str) -> tf.keras.regularizers.Regularizer | None  [source]
```
Sample a regularizer.

#### get_optimizer

```python
get_optimizer(trial: optuna.Trial) -> tf.keras.optimizers.Optimizer  [source]
```
Sample an optimizer, optionally sampling the learning rate when a range is provided.

#### get_scaler

```python
get_scaler(trial: optuna.Trial) -> Any  [source]
```
Sample a scikit‑learn scaler instance.

#### get_initializer

```python
get_initializer(trial: optuna.Trial, name: str) -> tf.keras.initializers.Initializer  [source]
```
Sample a kernel initializer.

#### get_default_params

```python
get_default_params() -> KParams  [source]
```
Return an instance with the default search space.

#### default

```python
default() -> KParams  [source]
```
Alias for `get_default_params()`.

#### full_search_space

```python
full_search_space() -> KParams  [source]
```
Return an instance where all choices are expanded to include nearly all Keras built‑ins.


### utils.convert_to_saved_model

```python
convert_to_saved_model(input_keras_path: str, output_zip_path: str)  [source]
```
Convert a `.keras` archive into a zipped TensorFlow SavedModel.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| input_keras_path | `str` | Path to the `.keras` file. |
| output_zip_path | `str` | Destination path for the resulting zip file. |

**Returns**

`None` – The SavedModel is written to `output_zip_path`.

**Raises**

- `Exception` – Propagated if TensorFlow fails to load or save the model.

**Examples**

```python
convert_to_saved_model("model.keras", "saved_model.zip")
```

### utils.capture_model_summary

```python
capture_model_summary(model: tf.keras.Model) -> str  [source]
```
Return the textual model summary for a given Keras model.

**Returns**

`str` – Summary string as produced by `model.summary()`.

### utils.punish_model_flops

```python
punish_model_flops(target: float | Sequence[float], model: tf.keras.Model, penalty_factor: float = 1e-10, direction: Literal["minimize", "maximize"] = "minimize")  [source]
```
Add a penalty proportional to model FLOPs to an objective value.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| target | `float | Sequence[float]` | Original objective value(s). |
| model | `tf.keras.Model` | Model used to compute FLOPs. |
| penalty_factor | `float` | Multiplicative penalty factor. |
| direction | `Literal["minimize", "maximize"]` | Whether lower or higher values are better. |

**Returns**

`float | Sequence[float]` – Penalised objective.

**Raises**

- `ValueError` – If `direction` is invalid.

### utils.punish_model_params

```python
punish_model_params(target: float | Sequence[float], model: tf.keras.Model, penalty_factor: float = 1e-9, direction: Literal["minimize", "maximize"] = "minimize")  [source]
```
Add a penalty based on the number of parameters.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| target | `float | Sequence[float]` | Original objective value(s). |
| model | `tf.keras.Model` | Model used to count parameters. |
| penalty_factor | `float` | Multiplicative penalty factor. |
| direction | `Literal["minimize", "maximize"]` | Whether lower or higher values are better. |

**Returns**

`float | Sequence[float]` – Penalised objective.

### utils.punish_model

```python
punish_model(target: float | Sequence[float], model: tf.keras.Model, type: Literal["flops", "params", None] = None, flops_penalty_factor: float = 1e-10, params_penalty_factor: float = 1e-9, direction: Literal["minimize", "maximize"] = "minimize")  [source]
```
Convenience wrapper that applies FLOPs or parameter penalties.

**Returns**

`float | Sequence[float]` – Penalised objective or the original value if `type` is `None`.


### analysis.model_param_distribution

```python
model_param_distribution(build_model_fn: Callable[[optuna.Trial], tf.keras.Model], bits_per_param: int, batch_size: int = 1, n_trials: int = 1000)  [source]
```
Sample random models from a builder function and plot distributions of parameters, model size and estimated training memory.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| build_model_fn | `Callable[[optuna.Trial], tf.keras.Model]` | Function that builds a model for a given trial. |
| bits_per_param | `int` | Number of bits used to store each parameter when computing model size. |
| batch_size | `int` | Batch size used for memory estimation. |
| n_trials | `int` | Number of random trials to sample. |

**Returns**

`None` – Displays histograms using Matplotlib.


### builders.build_cnn1d

```python
build_cnn1d(trial, kparams: KParams, x: layers.Layer, filters_range: int | tuple[int, int], kernel_size_range: int | tuple[int, int], filters_step: int = 10, kernel_size_step: int = 1, use_batch_norm: bool = True, trial_kernel_reg: bool = False, trial_bias_reg: bool = False, trial_activity_reg: bool = False, strides: int = 1, dilation_rate: int = 1, groups: int = 1, use_bias: bool = False, padding: str = "same", data_format: str = "channels_last", kernel_initializer=initializers.GlorotUniform(), bias_initializer=initializers.Zeros(), name_prefix: str = "cnn1d") -> layers.Layer  [source]
```
Build a 1‑D convolutional block with optional batch normalisation and activation sampling.

### builders.build_dense_as_conv1d

```python
build_dense_as_conv1d(...)
```
Convenience wrapper over `build_cnn1d` that emulates a dense layer using a 1‑D convolution with kernel size of 1.

### builders.build_cnn2d

```python
build_cnn2d(...)
```
Build a 2‑D convolutional block with optional hyperparameter tuning.

### builders.build_dense_as_conv2d

```python
build_dense_as_conv2d(...)
```
Dense‑like 2‑D convolution using kernel size `(1, 1)`.

### builders.build_cnn3d

```python
build_cnn3d(...)
```
Construct a 3‑D convolutional layer with optional regularisation and activation.

### builders.build_dense_as_conv3d

```python
build_dense_as_conv3d(...)
```
Dense‑like 3‑D convolution using kernel size `(1, 1, 1)`.

### builders.build_dnn

```python
build_dnn(...)
```
Create a fully connected block with optional batch normalisation and dropout.

### builders.build_grid_adjacency

```python
build_grid_adjacency(rows: int, cols: int) -> tf.sparse.SparseTensor  [source]
```
Generate a grid adjacency matrix suitable for GNN layers.

### builders.build_knn_adjacency

```python
build_knn_adjacency(rows: int, cols: int, k: int) -> tf.sparse.SparseTensor  [source]
```
Construct a k‑nearest neighbour adjacency matrix on a grid.

### builders.build_gcn

```python
build_gcn(...)
```
Build a single GCN layer with optional dropout, regularisation and batch normalisation.

### builders.build_gat

```python
build_gat(...)
```
Create a Graph Attention layer with a tunable number of heads.

### builders.build_cheb

```python
build_cheb(...)
```
Create a Chebyshev graph convolution layer.

### builders.build_lstm

```python
build_lstm(...)
```
Assemble an LSTM block with optional batch norm and custom dropout rates.

### builders.build_squeeze_excite_1d

```python
build_squeeze_excite_1d(x: layers.Layer, trial: optuna.Trial, kparams: KParams, ratio_choices: List[int], name_prefix: str = "se_block") -> layers.Layer  [source]
```
Apply a Squeeze‑and‑Excitation block to a 1‑D tensor.

### builders.trial_skip_connections

```python
trial_skip_connections(trial: optuna.Trial, layers_list: Sequence[tf.Tensor], axis_to_concat: int = -1, print_combinations: bool = False, strategy: str = "final", merge_mode: str = "concat") -> tf.Tensor  [source]
```
Optionally connect intermediate layers using concatenation or addition based on trial suggestions.

### builders.build_tcnn1d

```python
build_tcnn1d(...)
```
Create a transposed 1‑D convolution block.

### builders.build_tcnn2d

```python
build_tcnn2d(...)
```
Create a transposed 2‑D convolution block.

### builders.build_tcnn3d

```python
build_tcnn3d(...)
```
Create a transposed 3‑D convolution block.


## kernel

### monitoring.run_auto_restart

```python
run_auto_restart(file_path: str, success_flag_file: str = "/tmp/success.flag", title: str | None = None, max_restarts: int = 10, restart_delay: float = 3.0, recipients_file: str | None = None, credentials_file: str | None = None, restart_after_delay: float | None = None, retry_attempts: int | None = None, supress_tf_warnings: bool = False, resource_usage_log_file: str | None = None)  [source]
```
Execute a script or notebook with automatic restart and optional email alerts.

**Parameters** (excerpt)

| Name | Type | Description |
|------|------|-------------|
| file_path | `str` | Path to the script or notebook. |
| success_flag_file | `str` | File whose existence indicates successful completion. |
| title | `str | None` | Title shown in messages and emails. |
| max_restarts | `int` | Maximum number of restarts allowed. |
| restart_delay | `float` | Seconds to wait before restarting a crashed process. |
| recipients_file | `str | None` | JSON file with recipient emails. |
| credentials_file | `str | None` | JSON file with email credentials. |
| restart_after_delay | `float | None` | Force restarts after this delay even if no crash. |
| retry_attempts | `int | None` | Number of retry attempts before sending failure email. |
| supress_tf_warnings | `bool` | If True, suppress TensorFlow warnings in the child process. |
| resource_usage_log_file | `str | None` | Optional CSV file path to log CPU/GPU usage. |

**Returns**

`None` – Restarts are handled internally until completion or failure.

**Raises**

- `FileNotFoundError` – If `file_path` does not exist.
- `ValueError` – If the file type is unsupported.
- `ImportError` – If notebook conversion requires missing dependencies.


## optuna

### callbacks.ImprovementStagnation

```python
class ImprovementStagnation
```
Stop an Optuna study once the variance of recent expected improvements drops below a threshold.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| min_n_trials | `int` | Minimum completed trials before variance checking begins. |
| window_size | `int` | Number of recent improvement values to evaluate. |
| variance_threshold | `float` | Variance threshold triggering early stop. |
| improvement_evaluator | `optuna.terminator.BaseImprovementEvaluator | None` | Custom evaluator instance. |
| verbose | `bool` | If True, logs variance information each trial. |

### callbacks.StopIfKeepBeingPruned

```python
class StopIfKeepBeingPruned
```
Stop the optimization if a given number of consecutive trials are pruned.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| threshold | `int` | Number of consecutive pruned trials before stopping. |


### utils.supress_optuna_warnings

```python
supress_optuna_warnings()  [source]
```
Hide warnings emitted by experimental Optuna features.

### utils.get_remaining_trials

```python
get_remaining_trials(study: optuna.Study, num_trials: int) -> int  [source]
```
Return the number of trials that still need to run to reach `num_trials`.

### utils.cleanup_non_top_trials

```python
cleanup_non_top_trials(all_trial_ids: Set[int], top_trial_ids: Set[int], cleanup_paths: List[Tuple[str, str]])  [source]
```
Remove files or directories associated with trials that are not in the top‑K set.

### utils.rename_top_k_files

```python
rename_top_k_files(top_trials: List[optuna.Trial], file_configs: List[Tuple[str, str]])  [source]
```
Rename files for the top trials with ranking prefixes.

### utils.save_trial_params_to_file

```python
save_trial_params_to_file(filepath: str, params: dict[str, float], **kwargs: str)  [source]
```
Write trial parameters and metadata to a text file.

### utils.get_top_trials

```python
get_top_trials(study: optuna.Study, top_k: int, rank_key: str = "value", order: str = "descending") -> List[optuna.Trial]  [source]
```
Return the best trials in a study according to objective value or a user attribute.

### utils.save_top_k_trials

```python
save_top_k_trials(top_trials: List[optuna.Trial], args_dir: str, study: optuna.Study, extra_attrs: Optional[List[str]] = None)  [source]
```
Save a textual summary file for each top trial.

### utils.init_study_dirs

```python
init_study_dirs(run_dir: str, study_name: str = "optuna_study", subdirs: list | None = None)
```
Create the directory structure used to store study artifacts.


### keras.stats.get_model_stats

```python
get_model_stats(trial: optuna.Trial, model: tf.keras.Model, bits_per_param: int, batch_size: int, n_trials: int = 10000, device: int = 0, verbose: bool = False) -> Dict[str, float]  [source]
```
Collect FLOPs, MACs, parameter count, memory usage and power metrics for a Keras model and attach them to an Optuna trial.

**Returns**

`Dict[str, float]` – Dictionary with statistics such as `num_params`, `model_size`, `flops` and others.


### analysis.PlotConfig

```python
class PlotConfig
```
Holds global Matplotlib configuration used for Optuna study analysis. Modify attributes or call `set_plot_config_param` to adjust plot appearance.

### analysis.analyze_study

```python
analyze_study(study: optuna.Study, table_dir: str, top_frac: float = 0.2, param_name_mapping: Dict[str, str] | None = None, create_standalone: bool = False, save_data: bool = False, create_plotly: bool = False, plots: Optional[List[str]] = None)  [source]
```
Generate a suite of plots and summary tables for an Optuna study.

**Parameters** (excerpt)

| Name | Type | Description |
|------|------|-------------|
| study | `optuna.Study` | Study to analyse. |
| table_dir | `str` | Output directory for figures and tables. |
| top_frac | `float` | Fraction of trials considered best/worst. |
| param_name_mapping | `Dict[str, str] | None` | Optional mapping for nicer parameter labels. |
| create_standalone | `bool` | Also save individual figures for each plot. |
| save_data | `bool` | Export underlying data for LaTeX plotting. |
| create_plotly | `bool` | Save interactive Plotly versions when possible. |
| plots | `List[str] | None` | Subset of plot types to generate. `None` means all. |


## plot

### configs.config_plt

```python
config_plt(style: str = "single-column")  [source]
```
Configure Matplotlib for IEEE‑style figures. The `style` argument accepts `'single-column'` or `'double-column'` and adjusts figure size and font parameters accordingly.

**Raises**

- `ValueError` – If the style is not recognised.


## tensorflow

### model.get_model_usage_stats

```python
get_model_usage_stats(saved_model: str | tf.keras.Model, n_trials: int = 10000, device: int = 0, rapl_path: str = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj", verbose: bool = True) -> Tuple[float, float, float]  [source]
```
Measure inference time, average power and energy consumption for a SavedModel or Keras model. Requires access to NVIDIA NVML for GPU measurements or the RAPL interface for CPU measurements.

**Returns**

`Tuple[float, float, float]` – `(per_run_time, avg_power, avg_energy)`.

**Raises**

- `RuntimeError` – When NVML cannot be initialized.
- `ValueError` – If an unsupported device index is supplied.


## utils

### dir.create_run_directory

```python
create_run_directory(prefix: str, base_dir: str = "runs") -> str  [source]
```
Create a new directory named `<prefix><number>` inside `base_dir`, incrementing the number automatically.

### logs.log_resources

```python
log_resources(log_dir: str, interval: int = 5, **kwargs) -> None  [source]
```
Start background threads that periodically log CPU, RAM, GPU and TensorFlow resource usage to CSV files.

### progress.white_track

```python
white_track(iterable, *, description: str, total: int)
```
Iterate over `iterable` while displaying a progress bar in white style using
``tqdm``.

### cleanup.ChildProcessCleanup

```python
class ChildProcessCleanup
```
Terminate or kill child processes of the current process.

- `cleanup_children(exclude_pids=None)` – Terminate children and then force kill lingering processes.
- `add_protected_pid(pid)` – Exclude a PID from cleanup.
- `remove_protected_pid(pid)` – Remove a PID from the exclusion list.

### gpu.get_user_gpu_choice

```python
get_user_gpu_choice() -> str
```
Prompt the user for a GPU index and return it as a string.

### gpu.get_gpu_info

```python
get_gpu_info() -> None
```
Print TensorFlow and GPU configuration information similar to `nvidia-smi`.

### gpu.gpu_summary

```python
gpu_summary() -> None
```
Display a brief text summary of available GPUs.

### misc.clear

```python
clear()
```
Clear the terminal or Jupyter output.

### misc.format_number

```python
format_number(number, precision=2) -> str
```
Format a large or small number using metric suffixes.

### misc.format_bytes

```python
format_bytes(bytes_value, precision=2) -> str
```
Format a byte count using binary units (KB, MB, GB, …).

### misc.format_scientific

```python
format_scientific(number, max_precision=2) -> str
```
Return a number formatted in scientific notation.

### misc.format_number_commas

```python
format_number_commas(number, precision=2) -> str
```
Format a number with comma separators.

### misc.NotebookConverter.convert_notebook_to_python

```python
NotebookConverter.convert_notebook_to_python(notebook_path: Path) -> Path
```
Convert a Jupyter notebook to a `.py` script.

### terminal.SimpleTerminalLauncher

```python
class SimpleTerminalLauncher
```
Launch commands in a new system terminal.

- `set_supress_tf_warnings(value)` – Enable or disable filtering of TensorFlow warnings.
- `launch(command, working_dir)` – Start the command and return a `Popen` object with a `pid_file` attribute.

