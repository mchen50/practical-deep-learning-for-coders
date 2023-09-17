# Setup GPU PyTorch

## GPU

Check GPU usage:

```shell
nvidia-smi
nvidia-smi -L
nvidia-smi --query-gpu=gpu_name --format=csv
```

## Python

Install python with [pyenv](https://github.com/pyenv/pyenv):

```shell
curl https://pyenv.run | bash
# Load pyenv automatically by appending
# the following to 
~/.bash_profile if it exists, otherwise ~/.profile (for login shells)
and ~/.bashrc (for interactive shells) :

export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

`exec "$SHELL"`

Install python:

```shell
pyenv versions
pyenv install -l
pyenv install 3.11.5
pyenv global 3.11.5
```

Error installing python:
[Required dependencies](https://github.com/pyenv/pyenv/wiki#suggested-build-environment)

```shell
sudo apt update; 
sudo apt install build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

## Poetry

[Install poetry](https://python-poetry.org/docs/):

```shell
curl -sSL https://install.python-poetry.org | python3 -
```

Add `export PATH="/home/ming-chen/.local/bin:$PATH"` to your shell configuration file.

`poetry --version`

- Error `Failed to create the collection: Prompt dismissed..`

```shell
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
```

Update Poetry:

```shell
poetry self update
```

Enable tab completion for Bash:

```shell
poetry completions bash >> ~/.bash_completion
```

```shell
# Create new shell or activate existing shell
poetry shell

# See all available virtual environments
poetry env list
poetry env list --full-path

# Remove
poetry env remove test-O3eWbxRl-py3.7
poetry env remove --all

# Run script
poetry run python ming_jupyternb/hello.py

# Run Test
poetry add pytest --group test
poetry run pytest
```

## Setup Jupyter Notebook

Install Jupyter Notebook:

```shell
poetry add numpy
poetry add jupyter --group dev
```

## Install CUDA

[Nice guide](https://medium.com/ibm-data-ai/straight-forward-way-to-update-cuda-cudnn-and-nvidia-driver-and-cudnn-80118add9e53)
[Another guide](https://askubuntu.com/questions/1077061/how-do-i-install-nvidia-and-cuda-drivers-into-ubuntu)

1. Find out which driver version you need:
    - <https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html>
    - <https://www.nvidia.com/download/find.aspx>
2. Search and download driver
    - `sudo apt search nvidia-driver`
    - `sudo apt install nvidia-driver-530`
    - `sudo reboot`
    - Confirm: `nvidia-smi`
3. Install CUDA toolkit
    - <https://developer.nvidia.com/cuda-11-7-0-download-archive>
    - Runfile (local)
    - Verify version: `nvcc -V`
4. Install cuDNN
    - <https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#download>

## Install PyTorch

```shell
# For Compute Platform CUDA 11.8 
poetry source add --priority=explicit pytorch-gpu-src https://download.pytorch.org/whl/cu118
poetry add --source pytorch-gpu-src torch torchvision torchaudio
```

[Kickstart PyTorch Training](https://cnvrg.io/pytorch-cuda/)
