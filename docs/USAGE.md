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
After installing **araras** via `pip`, you can execute a script with automatic restarts using the built-in command-line interface:
```bash
python -m araras.runtime.monitoring path/to/script.py --success-flag-file /tmp/done.flag
```
Use `--help` to see all available options.
