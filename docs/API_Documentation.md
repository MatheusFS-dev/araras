# API Documentation


## ml.model.builders.cnn

### build_cnn1d
```python
build_cnn1d(trial: Any, kparams: KParams, x: keras.layers.Layer, filters_range: Union[int, tuple[int, int]], kernel_size_range: Union[int, tuple[int, int]], filters_step: int = 10, kernel_size_step: int = 1, use_batch_norm: bool = True, trial_kernel_reg: bool = False, trial_bias_reg: bool = False, trial_activity_reg: bool = False, strides: int = 1, dilation_rate: int = 1, groups: int = 1, use_bias: bool = False, padding: str = "same", data_format: str = "channels_last", kernel_initializer: Initializer = GlorotUniform(), bias_initializer: Initializer = Zeros(), name_prefix: str = "cnn1d")  [source]
```
Builds a configurable 1D convolution layer optionally tuned with Optuna.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| trial | `Any` | Optuna trial for suggesting hyperparameters. |
| kparams | `KParams` | Hyperparameter provider. |
| x | `keras.layers.Layer` | Input tensor. |
| filters_range | `Union[int, tuple[int, int]]` | Fixed filters or range to tune. |
| kernel_size_range | `Union[int, tuple[int, int]]` | Fixed kernel size or range to tune. |
| filters_step | `int` | Step size for filter tuning. |
| kernel_size_step | `int` | Step size for kernel size tuning. |
| use_batch_norm | `bool` | Whether to add BatchNormalization. |
| trial_kernel_reg | `bool` | Tune kernel regularizer. |
| trial_bias_reg | `bool` | Tune bias regularizer. |
| trial_activity_reg | `bool` | Tune activity regularizer. |
| strides | `int` | Convolution stride. |
| dilation_rate | `int` | Dilation rate. |
| groups | `int` | Grouped convolution groups. |
| use_bias | `bool` | Add bias term. |
| padding | `str` | Padding mode. |
| data_format | `str` | Tensor data format. |
| kernel_initializer | `Initializer` | Kernel initializer. |
| bias_initializer | `Initializer` | Bias initializer. |
| name_prefix | `str` | Prefix for generated layer names. |

**Returns**
`keras.layers.Layer` – Output layer after convolution, optional batch norm and activation.

**Raises**
- None

**Examples**
```python
# Example: basic usage
out = build_cnn1d(trial, kparams, x, 32, 3)
```

> [!NOTE]
> When `use_batch_norm` is True, `use_bias` is often unnecessary.

### build_dense_as_conv1d
```python
build_dense_as_conv1d(trial: Any, kparams: KParams, x: keras.layers.Layer, filters_range: int, filters_step: int = 10, padding: str = "valid", trial_kernel_reg: bool = False, trial_bias_reg: bool = False, trial_activity_reg: bool = False, name_prefix: str = "dense_as_conv1d")  [source]
```
Simulates a Dense layer using a 1×1 convolution.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| trial | `Any` | Optuna trial object. |
| kparams | `KParams` | Hyperparameter provider. |
| x | `keras.layers.Layer` | Input tensor. |
| filters_range | `int` | Number of output filters. |
| filters_step | `int` | Step size when tuning filters. |
| padding | `str` | Padding mode. |
| trial_kernel_reg | `bool` | Tune kernel regularizer. |
| trial_bias_reg | `bool` | Tune bias regularizer. |
| trial_activity_reg | `bool` | Tune activity regularizer. |
| name_prefix | `str` | Prefix for layer names. |

**Returns**
`keras.layers.Layer` – Layer equivalent to `Dense` on a sequence.

**Raises**
- None

**Examples**
```python
out = build_dense_as_conv1d(trial, kparams, x, 64)
```

> [!TIP]
> Reshape 2‑D tensors to `(batch, 1, features)` before applying.

### build_cnn2d
```python
build_cnn2d(trial: Any, kparams: KParams, x: keras.layers.Layer, filters_range: Union[int, tuple[int, int]], kernel_size_range: Union[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]], filters_step: int = 10, kernel_size_step: int = 1, use_batch_norm: bool = True, trial_kernel_reg: bool = False, trial_bias_reg: bool = False, trial_activity_reg: bool = False, strides: tuple[int, int] = (1, 1), dilation_rate: tuple[int, int] = (1, 1), groups: int = 1, use_bias: bool = False, padding: str = "same", data_format: str = "channels_last", kernel_initializer: Initializer = GlorotUniform(), bias_initializer: Initializer = Zeros(), name_prefix: str = "cnn2d")  [source]
```
Builds a configurable 2D convolution layer optionally tuned with Optuna.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| trial | `Any` | Optuna trial object. |
| kparams | `KParams` | Hyperparameter provider. |
| x | `keras.layers.Layer` | Input tensor. |
| filters_range | `Union[int, tuple[int, int]]` | Fixed filters or range to tune. |
| kernel_size_range | `Union[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]]` | Fixed kernel size or ranges. |
| filters_step | `int` | Step for filter tuning. |
| kernel_size_step | `int` | Step for kernel tuning. |
| use_batch_norm | `bool` | Add BatchNormalization. |
| trial_kernel_reg | `bool` | Tune kernel regularizer. |
| trial_bias_reg | `bool` | Tune bias regularizer. |
| trial_activity_reg | `bool` | Tune activity regularizer. |
| strides | `tuple[int, int]` | Convolution strides. |
| dilation_rate | `tuple[int, int]` | Dilation rate. |
| groups | `int` | Convolution groups. |
| use_bias | `bool` | Add bias term. |
| padding | `str` | Padding mode. |
| data_format | `str` | Tensor data format. |
| kernel_initializer | `Initializer` | Kernel initializer. |
| bias_initializer | `Initializer` | Bias initializer. |
| name_prefix | `str` | Prefix for layer names. |

**Returns**
`keras.layers.Layer` – Output layer after convolution, optional batch norm and activation.

**Examples**
```python
out = build_cnn2d(trial, kparams, x, (32, 64), ((3,3),(5,5)))
```

### build_dense_as_conv2d
```python
build_dense_as_conv2d(trial: Any, kparams: KParams, x: keras.layers.Layer, filters_range: int, filters_step: int = 10, padding: str = "valid", trial_kernel_reg: bool = False, trial_bias_reg: bool = False, trial_activity_reg: bool = False, name_prefix: str = "dense_as_conv2d")  [source]
```
Dense layer emulation using a 1×1×1 convolution on 2D data.

**Parameters**
| Name | Type | Description |
|------|------|-------------|
| trial | `Any` | Optuna trial. |
| kparams | `KParams` | Hyperparameters object. |
| x | `keras.layers.Layer` | Input tensor. |
| filters_range | `int` | Output filters. |
| filters_step | `int` | Step for filter tuning. |
| padding | `str` | Padding mode. |
| trial_kernel_reg | `bool` | Tune kernel regularizer. |
| trial_bias_reg | `bool` | Tune bias regularizer. |
| trial_activity_reg | `bool` | Tune activity regularizer. |
| name_prefix | `str` | Layer name prefix. |

**Returns**
`keras.layers.Layer` – Output tensor matching `Dense` behaviour.

### build_cnn3d
```python
build_cnn3d(trial: Any, kparams: KParams, x: keras.layers.Layer, filters_range: Union[int, Tuple[int, int]], kernel_size_range: Union[Tuple[int, int, int], Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]], filters_step: int = 10, kernel_size_step: int = 1, use_batch_norm: bool = True, trial_kernel_reg: bool = False, trial_bias_reg: bool = False, trial_activity_reg: bool = False, strides: Tuple[int, int, int] = (1,1,1), dilation_rate: Tuple[int, int, int] = (1,1,1), groups: int = 1, use_bias: bool = False, padding: str = "same", data_format: str = "channels_last", kernel_initializer: Initializer = GlorotUniform(), bias_initializer: Initializer = Zeros(), name_prefix: str = "cnn3d")  [source]
```
Creates a tunable 3D convolution block with optional normalization and activation.

### build_dense_as_conv3d
```python
build_dense_as_conv3d(trial: Any, kparams: KParams, x: keras.layers.Layer, filters_range: int, filters_step: int = 10, padding: str = "valid", trial_kernel_reg: bool = False, trial_bias_reg: bool = False, trial_activity_reg: bool = False, name_prefix: str = "dense_as_conv3d")  [source]
```
Simulates a Dense layer using a 1×1×1 convolution on 3D data.


### build_tcnn1d
```python
build_tcnn1d(trial: Any, kparams: KParams, x: keras.layers.Layer, filters_range: Union[int, tuple[int, int]], filters_step: int, kernel_size_range: Union[int, tuple[int, int]], kernel_size_step: int, data_format: str = "channels_last", padding: str = "same", strides: int = 1, dilation_rate: int = 1, use_bias: bool = False, kernel_initializer: Initializer = GlorotUniform(), bias_initializer: Initializer = Zeros(), use_batch_norm: bool = True, trial_kernel_reg: bool = False, trial_bias_reg: bool = False, trial_activity_reg: bool = False, name_prefix: str = "tcnn1d")  [source]
```
Creates a tunable 1D transposed convolution block.

### build_tcnn2d
```python
build_tcnn2d(trial: Any, kparams: KParams, x: keras.layers.Layer, filters_range: Union[int, tuple[int, int]], filters_step: int, kernel_size_range: Union[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]], kernel_size_step: int, data_format: str = "channels_last", padding: str = "same", strides: tuple[int, int] = (1,1), dilation_rate: tuple[int, int] = (1,1), use_bias: bool = False, kernel_initializer: Initializer = GlorotUniform(), bias_initializer: Initializer = Zeros(), use_batch_norm: bool = True, trial_kernel_reg: bool = False, trial_bias_reg: bool = False, trial_activity_reg: bool = False, name_prefix: str = "tcnn2d")  [source]
```
Creates a tunable 2D transposed convolution block.

### build_tcnn3d
```python
build_tcnn3d(trial: Any, kparams: KParams, x: keras.layers.Layer, filters_range: Union[int, Tuple[int, int]], filters_step: int, kernel_size_range: Union[Tuple[int, int, int], Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]], kernel_size_step: int, data_format: str = "channels_last", padding: str = "same", strides: Tuple[int, int, int] = (1,1,1), dilation_rate: Tuple[int, int, int] = (1,1,1), use_bias: bool = False, kernel_initializer: Initializer = GlorotUniform(), bias_initializer: Initializer = Zeros(), use_batch_norm: bool = True, trial_kernel_reg: bool = False, trial_bias_reg: bool = False, trial_activity_reg: bool = False, name_prefix: str = "tcnn3d")  [source]
```
Creates a tunable 3D transposed convolution block.


## ml.model.builders.dnn

### build_dnn
```python
build_dnn(trial: Any, kparams: KParams, x: keras.layers.Layer, units_range: Union[int, tuple[int, int]], dropout_rate_range: Union[float, tuple[float, float]], units_step: int = 10, dropout_rate_step: float = 0.1, kernel_initializer: Initializer = GlorotUniform(), bias_initializer: Initializer = Zeros(), use_bias: bool = True, use_batch_norm: bool = False, trial_kernel_reg: bool = False, trial_bias_reg: bool = False, trial_activity_reg: bool = False, name_prefix: str = "dnn")  [source]
```
Builds a dense network block with optional batch norm, dropout and regularization.


## ml.model.builders.gnn

### build_grid_adjacency
```python
build_grid_adjacency(rows: int, cols: int)  [source]
```
Build a grid adjacency matrix with GCN normalization.

### build_knn_adjacency
```python
build_knn_adjacency(rows: int, cols: int, k: int)  [source]
```
Construct a k‑nearest neighbour adjacency matrix on a 2‑D grid.

### build_gcn
```python
build_gcn(trial: Any, kparams: KParams, x: keras.layers.Layer, a_graph: tf.sparse.SparseTensor, units_range: Union[int, Tuple[int, int]], dropout_rate_range: Union[float, Tuple[float, float]], units_step: int = 10, dropout_rate_step: float = 0.1, kernel_initializer: Initializer = GlorotUniform(), bias_initializer: Initializer = Zeros(), use_bias: bool = True, use_batch_norm: bool = False, trial_kernel_reg: bool = False, trial_bias_reg: bool = False, trial_activity_reg: bool = False, name_prefix: str = "gcn")  [source]
```
Build a single GCN layer.

### build_gat
```python
build_gat(trial: Any, kparams: KParams, x: keras.layers.Layer, a_graph: tf.sparse.SparseTensor, units_range: Union[int, Tuple[int, int]], dropout_rate_range: Union[float, Tuple[float, float]], heads_range: Union[int, Tuple[int, int]], units_step: int = 10, dropout_rate_step: float = 0.1, heads_step: int = 1, concat_heads: bool = False, kernel_initializer: Initializer = GlorotUniform(), bias_initializer: Initializer = Zeros(), use_bias: bool = True, use_batch_norm: bool = False, trial_kernel_reg: bool = False, trial_bias_reg: bool = False, trial_activity_reg: bool = False, name_prefix: str = "gat")  [source]
```
Build a Graph Attention layer.

### build_cheb
```python
build_cheb(trial: Any, kparams: KParams, x: keras.layers.Layer, a_graph: tf.sparse.SparseTensor, units_range: Union[int, Tuple[int, int]], dropout_rate_range: Union[float, Tuple[float, float]], K_range: Union[int, Tuple[int, int]], units_step: int = 10, dropout_rate_step: float = 0.1, K_step: int = 1, kernel_initializer: Initializer = GlorotUniform(), bias_initializer: Initializer = Zeros(), use_bias: bool = True, use_batch_norm: bool = False, trial_kernel_reg: bool = False, trial_bias_reg: bool = False, trial_activity_reg: bool = False, name_prefix: str = "cheb")  [source]
```
Build a Chebyshev graph convolution layer.


## ml.model.builders.lstm

### build_lstm
```python
build_lstm(trial: Any, kparams: KParams, x: keras.layers.Layer, return_sequences: bool, units_range: Union[int, tuple[int, int]], units_step: int, dropout_rate_range: Union[float, tuple[float, float]], dropout_rate_step: float = 0.1, kernel_initializer: Initializer = GlorotUniform(), bias_initializer: Initializer = Zeros(), use_bias: bool = True, use_batch_norm: bool = False, trial_kernel_reg: bool = False, trial_bias_reg: bool = False, trial_activity_reg: bool = False, name_prefix: str = "lstm")  [source]
```
Builds a tunable LSTM block followed by optional batch norm and activation.


## ml.model.builders.se

### build_squeeze_excite_1d
```python
build_squeeze_excite_1d(x: keras.layers.Layer, trial: optuna.Trial, kparams: KParams, ratio_choices: List[int], name_prefix: str = "se_block")  [source]
```
Applies a squeeze‑and‑excitation block with Optuna tuned ratio and activations.


## ml.model.builders.skip

### trial_skip_connections
```python
trial_skip_connections(trial: optuna.trial.Trial, layers_list: Sequence[tf.Tensor], axis_to_concat: int = -1, print_combinations: bool = False, strategy: str = "final", merge_mode: str = "concat")  [source]
```
Adds optional skip connections between layers according to trial suggestions.

> [!CAUTION]
> All tensors to be merged must have compatible shapes; otherwise runtime errors may occur.


## ml.model.callbacks

### get_callbacks_model
```python
get_callbacks_model(backup_dir: str, tensorboard_logs: str)  [source]
```
Return common Keras callbacks for training.


## ml.model.hyperparams

### KParams
```python
@dataclass
class KParams:
    ...
```
Container for hyperparameter choices and samplers. Provides methods to set activation, regularizer and optimizer choices and to sample them within trials.


## ml.model.stats

### get_flops
```python
get_flops(model: keras.Model, batch_size: int = 1)  [source]
```
Return floating‑point operations for one forward pass.

### get_macs
```python
get_macs(model: keras.Model, batch_size: int = 1)  [source]
```
Estimate multiply‑accumulate operations.

### get_memory_and_time
```python
get_memory_and_time(model: keras.Model, batch_size: int = 1, device: int = 0, warmup_runs: int = 10, test_runs: int = 50, verbose: bool = True)  [source]
```
Measure peak memory usage and average inference time.

### get_model_usage_stats
```python
get_model_usage_stats(saved_model: str | keras.Model, n_trials: int = 10000, device: int = 0, rapl_path: str = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj", verbose: bool = True)  [source]
```
Estimate power draw and energy usage.

### write_model_stats_to_file
```python
write_model_stats_to_file(model: keras.Model, file_path: str, bits_per_param: int, batch_size: int, device: int = 0, n_trials: int = 1000, extra_attrs: Optional[List[str]] = None, verbose: bool = False)  [source]
```
Write statistics about a model to a text file.


## ml.model.tools

### convert_to_saved_model
```python
convert_to_saved_model(input_keras_path: str, output_zip_path: str)  [source]
```
Convert a `.keras` model to a zipped TensorFlow SavedModel.

### punish_model_flops
```python
punish_model_flops(target: Union[float, Sequence[float]], model: keras.Model, penalty_factor: float = 1e-10, direction: Literal["minimize", "maximize"] = "minimize")  [source]
```
Apply a penalty proportional to model FLOPs.

### punish_model_params
```python
punish_model_params(target: Union[float, Sequence[float]], model: keras.Model, penalty_factor: float = 1e-9, direction: Literal["minimize", "maximize"] = "minimize")  [source]
```
Apply a penalty proportional to parameter count.

### punish_model
```python
punish_model(target: Union[float, Sequence[float]], model: keras.Model, type: Literal["flops", "params", None] = None, flops_penalty_factor: float = 1e-10, params_penalty_factor: float = 1e-9, direction: Literal["minimize", "maximize"] = "minimize")  [source]
```
Convenience wrapper that applies FLOP or parameter penalties.


## ml.model.utils

### capture_model_summary
```python
capture_model_summary(model)  [source]
```
Capture a model summary as a string.


## ml.optuna.analyzer

### PlotConfig
```python
@dataclass
class PlotConfig:
    ...
```
Global configuration holder for plotting parameters used by the analyzer. Modify values via setters below.

### set_plot_config_param
```python
set_plot_config_param(param_name: str, value: Any)  [source]
```
Set a single plotting parameter in `PLOT_CFG`.

### set_plot_config_params
```python
set_plot_config_params(**kwargs: Any)  [source]
```
Set multiple plotting parameters at once.

### analyze_study
```python
analyze_study(study: optuna.Study, table_dir: str, top_frac: float = 0.2, param_name_mapping: Dict[str, str] = None, create_standalone: bool = False, save_data: bool = False, create_plotly: bool = False, plots: Optional[List[str]] = None)  [source]
```
Generate a comprehensive set of analysis plots from an Optuna study.


## ml.optuna.callbacks

### ImprovementStagnation
```python
class ImprovementStagnation:
    ...
```
Stop the study when recent improvement variance drops below a threshold.

### StopIfKeepBeingPruned
```python
class StopIfKeepBeingPruned:
    ...
```
Terminate optimization after a set number of consecutive pruned trials.

### NanLossPrunerOptuna
```python
class NanLossPrunerOptuna(callbacks.Callback):
    ...
```
Prunes a trial if NaN loss is encountered during training.

### get_callbacks_study
```python
get_callbacks_study(trial: optuna.Trial, tensorboard_logs: str = None, monitor: str = "val_loss")  [source]
```
Construct Keras callbacks customized for an Optuna trial.


## ml.optuna.model_tools

### estimate_training_memory
```python
estimate_training_memory(model: keras.Model, batch_size: int = 32)  [source]
```
Estimate total VRAM needed for model training.

### plot_model_param_distribution
```python
plot_model_param_distribution(build_model_fn: Callable[[optuna.Trial], keras.Model], bits_per_param: int, batch_size: int = 1, n_trials: int = 1000)  [source]
```
Sample random models and plot parameter histograms.

### set_user_attr_model_stats
```python
set_user_attr_model_stats(trial: optuna.Trial, model: keras.Model, bits_per_param: int, batch_size: int, n_trials: int = 10000, device: int = 0, verbose: bool = False)  [source]
```
Attach model statistics as user attributes on a trial and return them.


## ml.optuna.utils

### cleanup_non_top_trials
```python
cleanup_non_top_trials(all_trial_ids: Set[int], top_trial_ids: Set[int], cleanup_paths: List[Tuple[str, str]])  [source]
```
Remove files for trials that are not in the top‑K set.

### get_remaining_trials
```python
get_remaining_trials(study: optuna.Study, num_trials: int)  [source]
```
Return number of remaining trials.

### get_top_trials
```python
get_top_trials(study: optuna.Study, top_k: int, rank_key: str = "value", order: str = "descending")  [source]
```
Retrieve the top‑K trials sorted by a ranking key.

### rename_top_k_files
```python
rename_top_k_files(top_trials: List[optuna.Trial], file_configs: List[Tuple[str, str]])  [source]
```
Rename files of top‑K trials with ranking prefixes.

### save_trial_params_to_file
```python
save_trial_params_to_file(filepath: str, params: dict[str, float], **kwargs: str)  [source]
```
Save trial parameters and metadata to a text file.

### save_top_k_trials
```python
save_top_k_trials(top_trials: List[optuna.Trial], args_dir: str, study: optuna.Study, extra_attrs: Optional[List[str]] = None)  [source]
```
Save top‑K trial information to files.

### init_study_dirs
```python
init_study_dirs(run_dir, study_name: str = "optuna_study", subdirs = None)  [source]
```
Create a directory structure to store Optuna study results.


## notifications.email

### send_email
```python
send_email(subject: str, body: str, recipients_file: str, credentials_file: str, text_type: str = "plain", smtp_server: str = "smtp.gmail.com", smtp_port: int = 587)  [source]
```
Send an email notification to multiple recipients using stored credentials.


## runtime.monitoring

### run_auto_restart
```python
run_auto_restart(file_path: str, success_flag_file: str = "/tmp/success.flag", title: Optional[str] = None, max_restarts: int = 10, restart_delay: float = 3.0, recipients_file: Optional[str] = None, credentials_file: Optional[str] = None, restart_after_delay: Optional[float] = None, retry_attempts: int = None, supress_tf_warnings: bool = False, resource_usage_log_file: Optional[str] = None)  [source]
```
Monitor a script and automatically restart it on failure with optional email alerts.


## utils.io

### create_run_directory
```python
create_run_directory(prefix: str, base_dir: str = "runs")  [source]
```
Create a numbered run directory and return its path.

## utils.misc

### clear
```python
clear()  [source]
```
Clear terminal or notebook output.

### format_number
```python
format_number(number, precision=2)  [source]
```
Format numbers with scientific suffixes.

### format_bytes
```python
format_bytes(bytes_value, precision=2)  [source]
```
Format byte counts with binary units.

### format_scientific
```python
format_scientific(number, max_precision=2)  [source]
```
Format values using scientific notation.

### format_number_commas
```python
format_number_commas(number, precision=2)  [source]
```
Format numbers using commas as thousands separators.


## utils.system

### get_user_gpu_choice
```python
get_user_gpu_choice()  [source]
```
Interactively prompt for a GPU index.

### get_gpu_info
```python
get_gpu_info()  [source]
```
Print a detailed GPU configuration report.

### gpu_summary
```python
gpu_summary()  [source]
```
Display a concise GPU summary.

### log_resources
```python
log_resources(log_dir: str, interval: int = 5, **kwargs)  [source]
```
Log CPU, RAM and GPU usage at regular intervals.


## visualization.configs

### config_plt
```python
config_plt(style: str = "single-column")  [source]
```
Configure matplotlib for IEEE‑style figures.

