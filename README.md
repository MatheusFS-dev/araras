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
- [⚙️ Installation Instructions](#️-installation-instructions)
  - [📌 Prerequisites](#-prerequisites)
  - [🪜 Steps](#-steps)
- [📖 Usage](#-usage)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [🤝 Collaborators](#-collaborators)

## Features

- **Neural Architecture Search**: Leverage Optuna for efficient hyperparameter optimization.
- Custom Keras callbacks (e.g., NaN loss pruner for Optuna trials)
- **Keras/TensorFlow Utilities**: Simplify model building with custom builders and helpers.
- **Custom Callbacks**: Includes features like NaN loss pruning for Optuna trials.
- **Hyperparameter Management**: Fine-tune activations, regularizers, optimizers, and scalers.
- **Model Complexity Regularization**: Penalize loss based on FLOPs or parameter count.
- **System Resource Inspection**: Monitor GPU and system resources effortlessly.
- **Experiment Tracking**: Manage directories and files for seamless experiment organization.
- **Kernel Monitoring**: Receive email alerts for crashes or terminations.
- **Email Notifications**: Get notified for successes, warnings, or custom events.
- **Optuna Analysis Tools**: Visualize and analyze Optuna studies with ease.
- **Logging Utilities**: Track exceptions and resource usage effectively.
- **Modular Design**: Build extensible and customizable machine learning workflows.

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

## 📖 Usage

The package exposes many utilities grouped into modules. After installation you can
import individual functions directly:

```python
from araras.email.utils import send_email
```

Below is a summary of all modules and their functions.

### `araras.email.utils`

- **`get_credentials(file_path)`** – read sender email and password.
  - `file_path`: path to JSON with `"email"` and `"password"` keys.
  - **Returns**: `(email, password)` tuple.
- **`get_recipient_emails(file_path)`** – load recipient list.
  - `file_path`: JSON file containing an `"emails"` list.
- **`send_email(subject, body, recipients_file, credentials_file, text_type="plain")`** – send an email via SMTP.
  - `subject`: email subject line.
  - `body`: message body (plain or HTML).
  - `recipients_file`: path to recipients JSON.
  - `credentials_file`: path to credentials JSON.
  - `text_type`: MIME type, e.g. `"plain"` or `"html"`.
- **`run_with_notification(func, func_args, func_kwargs, recipients_file, credentials_file, subject_success="🎉 Task Completed Successfully", body_success="...", text_type="plain")`** – execute `func` and email success or error.
  - `func`: callable to run.
  - `func_args`: positional arguments for `func`.
  - `func_kwargs`: keyword arguments for `func`.
  - `recipients_file`: path to recipients JSON.
  - `credentials_file`: path to credentials JSON.
  - `subject_success`: subject line for success email.
  - `body_success`: body content for success email.
  - `text_type`: MIME type for both success and failure emails.
- **`notify_training_success(recipients_file, credentials_file, *, subject="🎉 Task Completed Successfully", body=None, text_type="html")`** – send a standard success message.
  - `recipients_file`: path to recipients JSON.
  - `credentials_file`: path to credentials JSON.
  - `subject`: subject line for the email.
  - `body`: optional body HTML/text; default template if `None`.
  - `text_type`: MIME type, e.g. `"html"` or `"plain"`.
- **`notify_warning(recipients_file, credentials_file, *, error=None, subject="❌ Task Failed with Error", text_type="html")`** – send an error notification.
  - `recipients_file`: path to recipients JSON.
  - `credentials_file`: path to credentials JSON.
  - `error`: optional `Exception` instance; full traceback used if `None`.
  - `subject`: subject line for the email.
  - `text_type`: MIME type for the message.

### `araras.keras.hparams`

- **`HParams`** – dataclass storing hyperparameter options.
  - `get_activation(trial, name)` – sample an activation function.
    - `trial`: Optuna trial object for sampling.
    - `name`: unique parameter name.
  - `get_regularizer(trial, name)` – sample a regularizer.
    - `trial`: Optuna trial object.
    - `name`: unique parameter name.
  - `get_optimizer(trial)` – sample and configure an optimizer.
    - `trial`: Optuna trial object.
  - `get_scaler(trial)` – select a scikit‑learn scaler.
    - `trial`: Optuna trial object.

### `araras.keras.utils.profiler`

- **`get_flops(model, batch_size=1)`** – total FLOPs for a forward pass.
  - `model`: Keras model to analyze.
  - `batch_size`: synthetic batch size for profiling.
- **`get_macs(model, batch_size=1)`** – MAC count assuming 2 FLOPs per MAC.
  - `model`: Keras model.
  - `batch_size`: synthetic batch size.
- **`get_memory_and_time(model, batch_size=1, device="GPU:0", warmup_runs=10, test_runs=50, verbose=False)`** – measure memory usage and inference time.
  - `model`: Keras model.
  - `batch_size`: batch size for profiling runs.
  - `device`: device string like `"GPU:0"` or `"CPU:0"`.
  - `warmup_runs`: number of warm-up iterations.
  - `test_runs`: number of timed iterations.
  - `verbose`: show progress bar during runs.

### `araras.keras.utils.punish`

- **`compute_flops_penalized_loss(loss, model, flops_penalty_factor=1e-10, operation="subtract")`** – adjust loss by FLOPs.
  - `loss`: original scalar loss value.
  - `model`: Keras model to profile.
  - `flops_penalty_factor`: scaling factor for FLOP penalty.
  - `operation`: `'add'` or `'subtract'` the penalty.
- **`compute_params_penalized_loss(loss, model, params_penalty_factor=1e-9, operation="subtract")`** – adjust loss by parameter count.
  - `loss`: original loss value.
  - `model`: Keras model.
  - `params_penalty_factor`: scaling factor for parameter penalty.
  - `operation`: `'add'` or `'subtract'` the penalty.

### `araras.keras.utils.summary`

- **`capture_model_summary(model)`** – return a string containing the Keras model summary.
  - `model`: Keras model.

### `araras.keras.skip_connections`

- **`trial_skip_connections(trial, layers_list, axis_to_concat=-1, print_combinations=False, strategy="final", merge_mode="concat")`** – build optional skip connections.
  - `trial`: Optuna trial object controlling skip choices.
  - `layers_list`: list of layer tensors.
  - `axis_to_concat`: axis used when concatenating outputs.
  - `print_combinations`: if `True`, print possible combinations.
  - `strategy`: `'final'` or `'any'` selection strategy.
  - `merge_mode`: `'concat'` or `'add'` merge mode.

### `araras.kernel.monitoring`

- **`start_monitor(pid, title, supress_tf_warnings=False)`** – launch a crash monitor.
-  - `pid`: process ID to monitor.
-  - `title`: label for the monitored process.
-  - `supress_tf_warnings`: suppress TensorFlow warnings.
- **`stop_monitor(monitor_info)`** – stop a monitor started with `start_monitor`.
-  - `monitor_info`: control info returned by `start_monitor`.
- **`check_crash_signal(monitor_info)`** – check if the monitored process crashed.
-  - `monitor_info`: dictionary from `start_monitor`.
- **`run_auto_restart(file_path, success_flag_file="/tmp/success.flag", title=None, max_restarts=10, restart_delay=3.0, recipients_file=None, credentials_file=None, restart_after_delay=None, retry_attempts=None, supress_tf_warnings=False)`** – run a script with auto‑restart and optional email alerts.
-  - `file_path`: path to script or notebook.
-  - `success_flag_file`: path for success flag file.
-  - `title`: optional human‑readable title.
-  - `max_restarts`: allowed restart attempts.
-  - `restart_delay`: seconds to wait between restarts.
-  - `recipients_file`: recipients JSON path.
-  - `credentials_file`: credentials JSON path.
-  - `restart_after_delay`: restart automatically after this delay.
-  - `retry_attempts`: retry count before failure email.
-  - `supress_tf_warnings`: suppress TensorFlow warnings.
- **`FlagBasedRestartManager`** – class implementing `run_file_with_restart` and restart logic.
-  - `max_restarts`, `restart_delay`, `recipients_file`, `credentials_file`, `retry_attempts` – constructor options.
-  - `run_file_with_restart(file_path, success_flag_file, title=None, restart_after_delay=None, supress_tf_warnings=False)` – run file with restart logic.

### `araras.plot.configs`

- **`config_plt(style="single-column")`** – configure matplotlib for IEEE‑style figures.
  - `style`: `'single-column'` or `'double-column'`.

### `araras.tensorflow.utils.model`

- **`get_model_usage_stats(saved_model, n_trials=10000, device="cpu", rapl_path="/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj", verbose=False)`** – measure power and energy usage of a model.
  - `saved_model`: path or Keras model instance.
  - `n_trials`: number of inference runs.
  - `device`: `'cpu'` or `'gpu'`.
  - `rapl_path`: path to RAPL energy counter for CPU runs.
  - `verbose`: show progress output.

### `araras.utils.cleanup`

- **`ChildProcessCleanup`** – helper class for terminating child processes.
  - `__init__(termination_timeout=2.0, kill_timeout=1.0)` – set timeouts.
  - `cleanup_children(exclude_pids=None)` – terminate or kill children.
    - `exclude_pids`: list of PIDs to skip.
  - `add_protected_pid(pid)` – exclude a PID from cleanup.
    - `pid`: process ID to protect.
  - `remove_protected_pid(pid)` – remove a protected PID.
    - `pid`: process ID to remove.
  - `get_child_count()` – number of child processes.

### `araras.utils.dir`

- **`create_run_directory(prefix, base_dir="runs")`** – create a new directory with an incrementing suffix.
  - `prefix`: directory name prefix.
  - `base_dir`: parent directory for all runs.

### `araras.utils.gpu`

- **`get_user_gpu_choice()`** – interactively select a GPU index.
  - *(no parameters)*
- **`get_gpu_info()`** – print detailed GPU and TensorFlow information.
  - *(no parameters)*
- **`gpu_summary()`** – display a short `nvidia-smi`‑like summary.
  - *(no parameters)*

### `araras.utils.logs`

- **`log_resources(log_dir, interval=5, **kwargs)`** – log system resources. Keyword flags include `cpu`, `ram`, `gpu`, `cuda`, and `tensorflow`.
  - `log_dir`: directory where CSV logs are saved.
  - `interval`: seconds between samples.
  - `**kwargs`: boolean flags selecting which resources to log.

### `araras.utils.misc`

- **`clear()`** – clear the terminal or notebook output.
  - *(no parameters)*
- **`format_number(number, precision=2)`** – format numbers with scientific suffixes.
  - `number`: numeric value.
  - `precision`: decimal places.
- **`format_bytes(bytes_value, precision=2)`** – format bytes in human‑readable form.
  - `bytes_value`: value in bytes.
  - `precision`: decimal places.
- **`format_scientific(number, max_precision=2)`** – format in scientific notation.
  - `number`: numeric value.
  - `max_precision`: maximum decimal places.
- **`format_number_commas(number, precision=2)`** – format numbers with commas.
  - `number`: numeric value.
  - `precision`: decimal places for floats.
- **`NotebookConverter.convert_notebook_to_python(notebook_path)`** – convert a notebook into a Python script.
  - `notebook_path`: path to `.ipynb` file.

### `araras.utils.terminal`

- **`SimpleTerminalLauncher`** – cross‑platform terminal launcher.
  - `__init__(supress_tf_warnings=False)` – set warning suppression.
  - `set_supress_tf_warnings(value)` – toggle TensorFlow warning suppression.
    - `value`: boolean flag.
  - `launch(command, working_dir)` – launch a command and return the process.
    - `command`: list of command strings.
    - `working_dir`: working directory for the new terminal.

### `araras.optuna.utils`

- **`get_remaining_trials(study, num_trials)`** – number of remaining trials.
  - `study`: Optuna study instance.
  - `num_trials`: total desired trials.
- **`cleanup_non_top_trials(all_trial_ids, top_trial_ids, cleanup_paths)`** – remove files from non‑top trials.
  - `all_trial_ids`: set of all trial IDs.
  - `top_trial_ids`: IDs of top trials to keep.
  - `cleanup_paths`: list of `(directory, template)` tuples.
- **`rename_top_k_files(top_trials, file_configs)`** – rename result files with ranking prefixes.
  - `top_trials`: list of ranked trials.
  - `file_configs`: list of `(directory, extension)` pairs.
- **`save_trial_params_to_file(filepath, params, **kwargs)`** – save trial parameters and metadata.
  - `filepath`: destination text file.
  - `params`: dictionary of hyperparameters.
  - `**kwargs`: additional metadata fields.
- **`get_top_trials(study, top_k, rank_key="value", rank_descending=True)`** – retrieve the best trials.
  - `study`: Optuna study.
  - `top_k`: number of top trials.
  - `rank_key`: attribute used for ranking.
  - `rank_descending`: sort order.
- **`save_top_k_trials(top_trials, args_dir, study, extra_attrs=None)`** – save top trial info to text files.
  - `top_trials`: list of trials.
  - `args_dir`: directory to save parameter files.
  - `study`: Optuna study.
  - `extra_attrs`: list of additional user attributes.
- **`init_study_dirs(run_dir, study_name="optuna_study", subdirs=None)`** – create Optuna study directory structure.
  - `run_dir`: base directory for experiment.
  - `study_name`: study directory name.
  - `subdirs`: list of subdirectories to create.

### `araras.optuna.analyze`

- **`set_plot_config_param(param_name, value)`** – modify a single plot configuration option.
  - `param_name`: configuration attribute to modify.
  - `value`: new value.
- **`set_plot_config_params(**kwargs)`** – modify multiple plot configuration options.
  - `**kwargs`: key/value pairs of parameters.
- **`format_title(template, display_name)`** – format plot titles.
  - `template`: title template string.
  - `display_name`: parameter display name.
- **`create_directories(table_dir, create_standalone=False, save_data=True)`** – prepare output folders.
  - `table_dir`: base directory for tables and figures.
  - `create_standalone`: also create standalone figure directories.
  - `save_data`: create CSV data directories.
- **`save_data_for_latex(data_dict, filename, data_dir)`** – save plot data to CSV for LaTeX.
  - `data_dict`: dictionary with x/y data.
  - `filename`: file name without extension.
  - `data_dir`: directory where CSV will be stored.
- **`get_param_display_name(param_name, param_name_mapping=None)`** – pretty‑print parameter names.
  - `param_name`: original parameter name.
  - `param_name_mapping`: optional mapping to display names.
- **`prepare_dataframe(study)`** – convert an Optuna study to a cleaned DataFrame.
  - `study`: Optuna study instance.
- **`classify_columns(df)`** – split DataFrame columns into numeric and categorical groups.
  - `df`: DataFrame with trial data.
- **`get_trial_subsets(df, top_frac)`** – separate best and worst trial subsets.
  - `df`: DataFrame with trial results.
  - `top_frac`: fraction of trials to treat as best/worst.
- **`format_numeric_value(x)`** – helper to format numbers for tables.
  - `x`: numeric value.
- **`save_summary_tables(df_best, df_worst, df_overall, dirs)`** – save CSV summary tables.
  - `df_best`: DataFrame of best trials.
  - `df_worst`: DataFrame of worst trials.
  - `df_overall`: DataFrame of all trials.
  - `dirs`: directory mapping for output files.
- **`describe_numeric(data, cols)`** – compute descriptive statistics for numeric columns.
  - `data`: DataFrame to analyze.
  - `cols`: list of numeric columns.
- **`create_frequency_table(data, cols)`** – frequency counts for categorical columns.
  - `data`: DataFrame to analyze.
  - `cols`: list of categorical columns.
- **`plot_hyperparameter_distributions(df, numeric_cols, cat_cols, dirs, plot_config=None, param_name_mapping=None)`** – visualize parameter distributions.
  - `df`: full DataFrame.
  - `numeric_cols`: numeric parameter columns.
  - `cat_cols`: categorical parameter columns.
  - `dirs`: output directories from `create_directories`.
  - `plot_config`: optional plot configuration object.
  - `param_name_mapping`: optional display name mapping.
- **`plot_param_importances(study, dirs)`** – bar chart of parameter importances.
  - `study`: Optuna study.
  - `dirs`: directories dict.
- **`plot_spearman_correlation(df, numeric_cols, dirs)`** – plot correlation heatmap.
  - `df`: DataFrame of trials.
  - `numeric_cols`: numeric columns to correlate.
  - `dirs`: directories dict.
- **`plot_parameter_boxplots(df, cat_cols, dirs, plot_config=None, param_name_mapping=None)`** – boxplots comparing best and worst trials.
  - `df`: trial DataFrame.
  - `cat_cols`: categorical parameter columns.
  - `dirs`: directories dict.
  - `plot_config`: optional config.
  - `param_name_mapping`: optional mapping.
- **`plot_trend_analysis(df, numeric_cols, dirs, plot_config=None, param_name_mapping=None)`** – line plots of parameter trends.
  - `df`: trial DataFrame.
  - `numeric_cols`: numeric columns to plot.
  - `dirs`: directories dict.
  - `plot_config`: optional config.
  - `param_name_mapping`: optional mapping.
- **`plot_optimal_ranges_analysis(study, df_top, dirs, plot_config=None, param_name_mapping=None)`** – visualize optimal parameter ranges.
  - `study`: Optuna study.
  - `df_top`: DataFrame of top trials.
  - `dirs`: directories dict.
  - `plot_config`: optional config.
  - `param_name_mapping`: optional mapping.
- **`print_study_columns(study, exclude=None)`** – list available study columns.
  - `study`: Optuna study.
  - `exclude`: columns to skip.
- **`analyze_study(study, fig_dir='figures', table_dir='tables', top_frac=0.2, param_name_mapping=None)`** – run the full analysis pipeline.
  - `study`: Optuna study to analyze.
  - `fig_dir`: directory for figures.
  - `table_dir`: directory for tables.
  - `top_frac`: fraction of best/worst trials.
  - `param_name_mapping`: optional mapping for display names.

### `araras.optuna.viz`

- **`report(study, metrics=None, summary_values=None, best_is_min=True, improvement_evaluator=None, error_evaluator=None, min_n_trials=DEFAULT_MIN_N_TRIALS)`** – update or create realtime plots for a study.
  - `study`: Optuna study instance.
  - `metrics`: mapping of metric names to callables returning values from a trial.
  - `summary_values`: mapping of labels to callables returning study-level values.
  - `best_is_min`: treat lower values as better when tracking best metrics.
  - `improvement_evaluator`: optional evaluator to compute improvement.
  - `error_evaluator`: optional evaluator to compute error bars.
  - `min_n_trials`: minimum number of trials before computing improvement.

**Example**

```python
import optuna
from araras.optuna import report

study = optuna.create_study(direction="minimize")

def objective(trial):
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    value = x**2 + y**2
    report(
        study,
        metrics={"loss": lambda t: t.value},
        summary_values={"Completed trials": lambda s: len(s.trials)},
    )
    return value

study.optimize(objective, n_trials=20)
```

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