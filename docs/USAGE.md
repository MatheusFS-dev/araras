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
- `Force periodic restart`: Enabled by default. Press Enter or answer `y`, then enter a positive number of minutes, including decimals such as `0.5`, to restart each running target after that fixed interval. Answer `n` to disable it.
- `Detect possible memory leak`: Enabled by default. Press Enter or answer `y`, then choose a positive warm-up in minutes; blank warm-up input keeps the five-minute default. Answer `n` to disable it.
- `Report logs`: Enabled by default. The monitor writes a report bundle under `runs/monitor_logs/<target-name>-<timestamp>/` unless you explicitly answer `n`.

Periodic restarts are intentional lifecycle events. They do not consume or
reset `Max restarts`, which remains reserved for genuine crashes and launch
failures. When email alerts are configured, the monitor sends a distinct
`Scheduled Restart Successful` confirmation after the replacement PID is
running and monitored. That confirmation includes an inline graph of the
previous PID's process-tree RAM usage from launch until the scheduled restart,
even when memory leak detection is disabled.

Possible memory leak detection samples RSS for the monitored process tree even
when `Report logs` is disabled. After the warm-up, it warns only when a rolling
10-minute window grows by at least 100 MiB with a fitted slope of at least
10 MiB/min. The window is divided into five two-minute median blocks, and at
least three of the four transitions must grow by 10 MiB or more. A final sample
is evaluated immediately before a scheduled restart so the restart boundary
cannot suppress an eligible warning. The warning is attempted once per target
across all restarts and includes an inline RAM graph. It does not stop or
restart the target, and the wording remains probabilistic because RSS growth
cannot prove a leak.

When `Report logs` is enabled, each monitored file gets:

- A `<target-name>.log` event log.
- `samples.jsonl` with timestamped CPU, memory, and GPU samples over time.
- `restarts.json`, `summary.json`, and `summary.txt`, with scheduled and crash/failure restart counts reported separately.
- `cpu.png`, `memory.png`, `gpu_utilization.png`, `gpu_memory.png`, and `gpu_temperature.png`.

CPU and memory values are measured for the monitored process tree rather than
only the root PID. GPU values reflect host GPU telemetry from `nvidia-smi`
when it is available.
