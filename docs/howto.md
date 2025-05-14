# Description

This document is a concise compilation of practical guidance and commands for developers working with Git, Python virtual environments, and machine learning frameworks. It covers essential commands for Git operations, setting up Conda environments with CUDA support, creating and managing Python virtual environments, and using Docker for TensorFlow development. The document also includes instructions for running Jupyter notebooks in a WSL environment and provides a comprehensive command reference for Docker operations.


# Table of Contents

- [Description](#description)
- [Table of Contents](#table-of-contents)
  - [Useful Git and Python Virtual Environment Commands](#useful-git-and-python-virtual-environment-commands)
    - [Git Configuration](#git-configuration)
    - [Create and Switch Branch](#create-and-switch-branch)
    - [Add and Commit Changes](#add-and-commit-changes)
    - [Tagging a Commit](#tagging-a-commit)
    - [Pushing to Remote Repository](#pushing-to-remote-repository)
  - [Conda Virtual Environment with CUDA](#conda-virtual-environment-with-cuda)
    - [Create a Conda Environment](#create-a-conda-environment)
    - [Install ipykernel](#install-ipykernel)
    - [Add venv to kernels](#add-venv-to-kernels)
    - [Add Cuda to any conda enviroment](#add-cuda-to-any-conda-enviroment)
  - [Python Virtual Environment](#python-virtual-environment)
    - [Create Virtual Environment](#create-virtual-environment)
    - [Activate Virtual Environment](#activate-virtual-environment)
    - [Freeze Installed Packages](#freeze-installed-packages)
    - [Create a virtual environment with system site packages](#create-a-virtual-environment-with-system-site-packages)
  - [Using Jupyter and browser on WSL](#using-jupyter-and-browser-on-wsl)
  - [Docker](#docker)
    - [1. Verify Docker Installation](#1-verify-docker-installation)
    - [2. List Docker Images](#2-list-docker-images)
    - [3. List Docker Containers](#3-list-docker-containers)
    - [4. Run a New TensorFlow Docker Container](#4-run-a-new-tensorflow-docker-container)
      - [With GPU Support \& Port Forwarding](#with-gpu-support--port-forwarding)
      - [With Volume Mounting (For Persistent Data)](#with-volume-mounting-for-persistent-data)
    - [5. Start/Restart a Stopped Container](#5-startrestart-a-stopped-container)
    - [6. Install Dependencies Inside the Container](#6-install-dependencies-inside-the-container)
    - [7. Locate and Verify Notebooks](#7-locate-and-verify-notebooks)
    - [8. Start Jupyter Notebook](#8-start-jupyter-notebook)
    - [9. Access Jupyter Notebook](#9-access-jupyter-notebook)
    - [10. Train Your Model](#10-train-your-model)
    - [11. Exit the Container](#11-exit-the-container)
    - [12. Save Container State](#12-save-container-state)
    - [13. Remove Unused Containers \& Images](#13-remove-unused-containers--images)
    - [14. Full Command Reference](#14-full-command-reference)
---

## Useful Git and Python Virtual Environment Commands

### Git Configuration
Set up your Git user email and username.
```bash
git config user.email "your.email@example.com"
git config user.name "YourUsername"
```

### Create and Switch Branch
Create a new branch and switch to it.
```bash
git checkout -b branch
```

### Add and Commit Changes
Add all changes and commit with a message.
```bash
git add .
git commit -m "Released new patch"
```

### Tagging a Commit
Create a new tag for the current commit.
```bash
git tag v1.0
```

### Pushing to Remote Repository
Push the branch and tag to the remote repository.
```bash
git push https://your_token@github.com/YourUsername/YourRepo.git branch
git push origin v1.0
git push origin v1.0 --force
```

---

## Conda Virtual Environment with CUDA

### Create a Conda Environment

```bash
conda create --prefix ./env tensorflow-gpu
```

Activate the environment:

```bash
conda activate ./env
```

### Install ipykernel

```bash
pip install ipykernel notebook
```

### Add venv to kernels

```bash
python -m ipykernel install --user --name=tf-gpu --display-name "Python (tf-gpu)"
```

Verify by running:
```bash
jupyter kernelspec list
```

### Add Cuda to any conda enviroment
```bash
conda install -c conda-forge cudatoolkit cudnn
```

```bash
pip install tensorflow[and-cuda]
```

---

## Python Virtual Environment

### Create Virtual Environment
Create a new virtual environment.
```bash
python3 -m venv .venv
```

### Activate Virtual Environment
Activate the virtual environment.
```bash
source .venv/bin/activate
```

### Freeze Installed Packages
Save the current list of installed packages to a file.
```bash
pip freeze > requirements.txt
```

### Create a virtual environment with system site packages
```bash
virtualenv --system-site-packages .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --upgrade nvidia-pyindex
pip install -r ./requirements.txt
```

---

## Using Jupyter and browser on WSL


To install the notebook, run:
```bash
pip install notebook
```

Once the installation completes, launch it with:
```bash
jupyter notebook --no-browser --ip=127.0.0.1 --port=8888
```

This command starts the notebook server and automatically opens your default web browser pointed to the notebook home page. If it doesn't open automatically, open your browser and navigate to the URL provided in the terminal (typically http://localhost:8888/?token=YOUR_TOKEN).



## Docker

> [!TIP]
> Just use WSL if working on Windows.

### 1. Verify Docker Installation
Check if Docker is installed:
```sh
docker --version
```
Check if Docker is running:
```sh
docker ps -a
```

### 2. List Docker Images
To check available images:
```sh
docker images
```
**Output format:**
```
REPOSITORY          TAG       IMAGE ID        CREATED         SIZE
tensorflow/tensorflow  latest-gpu  abc123def456  3 days ago     5.6GB
```
- `{REPOSITORY}` – Image name (e.g., `tensorflow/tensorflow`)
- `{TAG}` – Version tag (e.g., `latest-gpu`)
- `{IMAGE ID}` – Unique ID of the image
- `{SIZE}` – Size of the image

**If TensorFlow is not available, pull it:**
```sh
docker pull tensorflow/tensorflow:latest-gpu
```

### 3. List Docker Containers
Check running containers:
```sh
docker ps
```
Check **all** containers, including stopped ones:
```sh
docker ps -a
```

**Output format:**
```
CONTAINER ID   IMAGE      COMMAND        STATUS         NAMES
123abc456def   tensorflow/tensorflow "bash"        Up 10 minutes   tensorflow_container
```
- `{CONTAINER ID}` – Unique ID for the container
- `{IMAGE}` – Docker image used
- `{COMMAND}` – Command running inside the container
- `{STATUS}` – Running (`Up`), stopped (`Exited`), etc.
- `{NAMES}` – Auto-assigned or user-defined name


### 4. Run a New TensorFlow Docker Container
#### With GPU Support & Port Forwarding
```sh
docker run -it --gpus all -p 8888:8888 --name {NAME} tensorflow/tensorflow:latest-gpu bash
```
- `-it` – Interactive mode (keeps the terminal open)
- `--gpus all` – Enables GPU usage inside the container
- `-p 8888:8888` – Maps port `8888` from the container to the host (needed for Jupyter)
- `--name {NAME}` – Assigns a custom name (e.g., `tensorflow_container`)
- `tensorflow/tensorflow:latest-gpu` – Specifies the image

#### With Volume Mounting (For Persistent Data)
To mount a local directory:
```sh
docker run -it --gpus all -p 8888:8888 -v /local/path:/workspace --name {NAME} tensorflow/tensorflow:latest-gpu bash
```
- `-v /local/path:/workspace` – Mounts the host directory `/local/path` into the container at `/workspace`


### 5. Start/Restart a Stopped Container
If a container exists but is stopped, restart it:
```sh
docker start -ai {NAME}
```
- `-a` – Attach to the container (shows output)
- `-i` – Interactive mode

### 6. Install Dependencies Inside the Container
Once inside the container, install required libraries:
```sh
pip install jupyterlab optuna autokeras
```

```sh
apt update && apt install -y git
```

Check if TensorFlow detects the GPU:
```sh
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### 7. Locate and Verify Notebooks
List files inside the container:
```sh
ls /workspace
```
Move to the correct directory:
```sh
cd /workspace
```

### 8. Start Jupyter Notebook
Run Jupyter with external access:
```sh
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
```
- `--ip=0.0.0.0` – Allows connections from any IP (required in Docker)
- `--port=8888` – Runs Jupyter on port `8888`
- `--allow-root` – Allows execution as `root` (required inside containers)

Copy the **access token** from the terminal output.

### 9. Access Jupyter Notebook
On your **host machine**, open a browser and enter:
```
http://localhost:8888/?token={YOUR_TOKEN}
```
Replace `{YOUR_TOKEN}` with the token shown in the terminal.

### 10. Train Your Model
Open `example.ipynb` and execute the cells.

### 11. Exit the Container
After training, **exit** without stopping the container:
```sh
exit
```
To **stop** the container:
```sh
docker stop {NAME}
```

### 12. Save Container State
If you want to keep modifications inside the container, **commit it as a new image**:
```sh
docker commit {CONTAINER ID} tensorflow_custom
```
Now, you can run it later with:
```sh
docker run -it --gpus all -p 8888:8888 --name {NEW_NAME} tensorflow_custom bash
```

### 13. Remove Unused Containers & Images
List all stopped containers:
```sh
docker ps -a -f "status=exited"
```
Remove a specific container:
```sh
docker rm {CONTAINER ID}
```
Remove all stopped containers:
```sh
docker container prune -f
```
Remove an image:
```sh
docker rmi {IMAGE ID}
```
Remove all unused images:
```sh
docker image prune -a -f
```

### 14. Full Command Reference
| Action                          | Command |
|---------------------------------|---------|
| Check Docker version | `docker --version` |
| List images | `docker images` |
| List all containers | `docker ps -a` |
| Run a new TensorFlow container | `docker run -it --gpus all -p 8888:8888 --name {NAME} tensorflow/tensorflow:latest-gpu bash` |
| Run container with volume | `docker run -it --gpus all -p 8888:8888 -v /local/path:/workspace --name {NAME} tensorflow/tensorflow:latest-gpu bash` |
| Start an existing container | `docker start -ai {NAME}` |
| Install Jupyter inside container | `pip install jupyterlab` |
| Start Jupyter Notebook | `jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root` |
| Stop a container | `docker stop {NAME}` |
| Exit a container | `exit` |
| Remove a container | `docker rm {CONTAINER ID}` |
| Remove all stopped containers | `docker container prune -f` |
| Remove an image | `docker rmi {IMAGE ID}` |
| Remove all unused images | `docker image prune -a -f` |
| Save container as image | `docker commit {CONTAINER ID} tensorflow_custom` |

If something breaks, use:
```sh
docker logs {CONTAINER ID}
```
To check what happened.