# OGE (Orbital Game Environment)

OGE 是一个用于强化学习研究的轨道博弈环境，专注于航天器轨道追逃博弈场景。该环境提供了 Gymnasium 兼容的接口，使用 C++ 后端实现高性能仿真，并通过 Python 接口方便地与主流强化学习框架（如 SKRL）集成。

## 特性

- 🚀 高性能的轨道动力学仿真（C++ 后端）
- 🎮 Gymnasium 兼容的强化学习环境接口
- 🤖 支持多智能体追逃博弈场景
- 📊 集成 SKRL 强化学习库，支持 PPO 等算法
- 🔧 灵活的环境配置系统

## License

本项目采用 [GNU General Public License v2.0](LICENSE.md) 开源协议。

## 1. 开发环境配置

### 1.1 更新 gcc

安装 GCC 13

```
sudo apt update
sudo apt install build-essential gcc-13 g++-13
```

设置为默认版本

```
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100
```

如果系统里有多个版本，可以用以下命令切换：

```
sudo update-alternatives --config gcc                                                                                                                                                                                
sudo update-alternatives --config g++
```

安装完后验证：

```
gcc --version                                                                                                                                                                                                        
g++ --version
```

输出类似如下内容：

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

### 1.2 安装 vcpkg

克隆 `vcpkg` 并且执行安装

```bash
git clone https://www.github.com/microsoft/vcpkg
cd vcpkg
./bootstrap-vcpkg.sh
```

将以下内容添加到 `~/.bashrc` 或者 `~/.zshrc`

```bash
# >>> vcpkg
export VCPKG_ROOT=<path-to-vcpkg>
export PATH=$VCPKG_ROOT:$PATH
# <<< vcpkg
```

### 1.3 Python 环境配置

创建 `conda` 环境并且安装基础依赖

```bash
conda create -n oge python=3.13
conda activate oge
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
```

安装 `skrl` 库

```bash
conda activate oge
cd third_party/skrl-1.4.3
pip install -e ".["torch"]" -i https://mirrors.ustc.edu.cn/pypi/simple
```

## 2. 安装 OGE

```bash
conda activate oge
pip install .
```

## 3. 运行 SKRL 训练示例

```bash
python scripts/train.py
```