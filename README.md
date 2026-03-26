# OGE

## 1. 开发环境配置

### 1.1 安装 vcpkg

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

### 1.2 Python 环境配置

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
