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

1. Ensure your target script writes a *success flag* file when it
   finishes successfully. The monitoring process watches for this file
   to stop restarting the script.
2. Invoke the CLI with the path to your script and the location of the
   flag file:

```bash
python -m araras.runtime.monitoring path/to/script.py -s /tmp/done.flag
```

Additional options allow controlling restart behavior, for example:

```bash
python -m araras.runtime.monitoring path/to/script.py -s /tmp/done.flag \
    -m 5 -d 3 -f 3600
```

To keep everything contained in your current tmux session use the
``--tmux-split`` option. This spawns both the target process and crash
monitor in new panes:

```bash
python -m araras.runtime.monitoring path/to/script.py -s /tmp/done.flag \
    --tmux-split --tmux-session mysession
```

Run with `--help` to see all available arguments.

