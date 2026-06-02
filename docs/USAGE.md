# Publishing a New Release to PyPI

Follow these steps to release a new version of the package on PyPI:

1. Clean up previous build artifacts:
    ```bash
    rm -rf build/ dist/ *.egg-info
    ```

2. Build the package:
    ```bash
    python3 -m build
    ```

3. Verify the package integrity:
    ```bash
    python3 -m twine check dist/*
    ```

4. Upload the package to PyPI:
    ```bash
    python3 -m twine upload dist/*
    ```

# Installing the Package Locally or from GitHub

### Local Installation
To install the package locally without using PyPI, run:
```bash
pip install .
```

### Installation from GitHub
To install the package directly from the GitHub repository, use:
```bash
pip install git+https://github.com/MatheusFS-dev/araras.git
```

### Running the Monitoring CLI
After installing **araras** via `pip`, you can execute a script with
automatic restarts using the built-in command-line interface.

The current `monitor` command is prompt-driven and accepts only positional
target file paths:

```bash
monitor path/to/script.py another.ipynb
```

The launcher asks once per invocation for these shared settings:

- `Monitor title`: Blank input keeps each file stem as the display title.
- `Choose JSON folder`: Selects the email configuration directory.
- `Max restarts`: Blank input keeps the default `10`. Enter `0` to disable automatic restarts after the initial launch attempt.
- `Report logs`: Enabled by default. The monitor writes a report bundle under `runs/monitor_logs/<target-name>-<timestamp>/` unless you explicitly answer `n`.

When `Report logs` is enabled, each monitored file gets:

- A `<target-name>.log` event log.
- `samples.jsonl` with timestamped CPU, memory, and GPU samples over time.
- `restarts.json`, `summary.json`, and `summary.txt`.
- `cpu.png`, `memory.png`, `gpu_utilization.png`, `gpu_memory.png`, and `gpu_temperature.png`.

CPU and memory values are measured for the monitored process tree rather than
only the root PID. GPU values reflect host GPU telemetry from `nvidia-smi`
when it is available.
