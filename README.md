# OGE
## 1. 开发环境配置
### 1.1 安装 vcpkg
```bash
git clone https://www.github.com/microsoft/vcpkg
cd vcpkg
./bootstrap-vcpkg.sh
```
将以下内容添加到`~/.bashrc`或者`~/.zshrc`
```bash
# >>> vcpkg
export VCPKG_ROOT=<path-to-vcpkg>
export PATH=$VCPKG_ROOT:$PATH
# <<< vcpkg
```

### 1.2 Python 环境配置
```bash
conda create -n oge python=3.13
conda activate marppo
```

## 2. 安装 OGE 

## 3. 运行 SKRL 训练示例
