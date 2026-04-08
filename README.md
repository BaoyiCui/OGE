# OGE (Orbital Game Environment)

OGE is a reinforcement learning environment for orbital game scenarios, focusing on spacecraft pursuit-evasion games. The environment provides a Gymnasium-compatible interface with a high-performance C++ backend and Python bindings for easy integration with mainstream reinforcement learning frameworks (such as SKRL).

## Features

- 🚀 High-performance orbital dynamics simulation (C++ backend)
- 🎮 Gymnasium-compatible reinforcement learning environment interface
- 🤖 Multi-agent pursuit-evasion game scenarios
- 📊 Integration with SKRL reinforcement learning library, supporting PPO and other algorithms
- 🔧 Flexible environment configuration system

## License

This project is licensed under the [GNU General Public License v2.0](LICENSE.md).

## 1. Development Environment Setup

### 1.1 Update GCC

Install GCC 13

```bash
sudo apt update
sudo apt install build-essential gcc-13 g++-13
```

Set as default version

```bash
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100
```

If you have multiple versions installed, you can switch between them:

```bash
sudo update-alternatives --config gcc
sudo update-alternatives --config g++
```

Verify the installation:

```bash
gcc --version
g++ --version
```

Expected output:

```text
(oge) ➜  OGE git:(main) ✗ gcc --version
gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
Copyright (C) 2023 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

(oge) ➜  OGE git:(main) ✗ g++ --version
g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
Copyright (C) 2023 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

### 1.2 Install vcpkg

Clone `vcpkg` and run the installation script

```bash
git clone https://www.github.com/microsoft/vcpkg
cd vcpkg
./bootstrap-vcpkg.sh
```

Add the following to your `~/.bashrc` or `~/.zshrc`

```bash
# >>> vcpkg
export VCPKG_ROOT=<path-to-vcpkg>
export PATH=$VCPKG_ROOT:$PATH
# <<< vcpkg
```

### 1.3 Python Environment Setup

Create a `conda` environment and install dependencies

```bash
conda create -n oge python=3.13
conda activate oge
pip install -r requirements.txt
```

Install the `skrl` library

```bash
conda activate oge
pip install "skrl[torch]==1.4.3"
```

## 2. Install OGE

```bash
conda activate oge
pip install .
```

## 3. Run SKRL Training Example

```bash
python scripts/train.py
```

## Citation

If you use OGE in your research, please cite this repository.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.