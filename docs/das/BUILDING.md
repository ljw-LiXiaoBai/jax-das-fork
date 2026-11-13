<!-- DAS-INTRO:BEGIN -->

> [!IMPORTANT]
> **jax-das** 是面向 DAS 架构的 jax 下游适配发行版，
> 本项目基于 [ROCm/jax](https://github.com/ROCm/jax) 的 `rocm-jaxlib-v0.10.2`（`db744a7ed2a66ec1f2260c9546b2cb7d516507f4`）基线构建并集成 DAS 支持。
> 本项目不是 jax 官方发行版，而是基于[ROCm/jax](https://github.com/ROCm/jax/tree/rocm-jaxlib-v0.10.2) 二次开发；ROCm jax 源自[jax-ml/jax](https://github.com/jax-ml/jax/tree/jax-v0.10.2)二次开发。

* **目标架构：** jax
* **上游版本：** rocm-jaxlib-v0.10.2
* **上游基线：** `https://github.com/ROCm/jax/tree/db744a7ed2a66ec1f2260c9546b2cb7d516507f4`
* **Python 发布包名：** `jax-das`
* **Python 导入名：** `jax`
* **上游许可证：** Apache-2.0
* **第三方许可信息：** [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)

> `jax-das` 与上游 `jax` 会安装同名 `jax` 模块，
> 请勿在同一虚拟环境或容器中混装。
> 本项目由 Hygon Information Technology Co., Ltd. 修改并维护。

<!-- DAS-INTRO:END -->

---

# JAX 编译指南

本文档说明如何在 HCU 平台上编译 JAX。

## 默认编译配置

```text
ubuntu: 22.04          dtk:    26.04
python: 3.11           aillvm: 18
mpi:    5.0            bazel: 7.7.0

建议：CPU ≥ 32 核、内存 ≥ 64 GB、磁盘 ≥ 200 GB
```

## 前置准备

宿主机还需已安装 HCU 驱动，并确保 `/opt/hyhal` 目录存在：

```bash
ls /opt/hyhal   # 确认目录存在，编译与运行时都会用到
```

---

## 编译流程

本节说明 jax-das 的编译流程，支持以下环境组合：

| OS | DTK_VERSION | PYTHON_VERSION |
| --- | --- | --- |
| ubuntu | 26.04 | 3.11 / 3.12 |
| rockylinux | 26.04 | 3.11 / 3.12 |

### 1. 拉取基础镜像

推荐直接使用光合社区基础镜像。不同系统与 Python 版本对应关系如下：

| 环境 | 说明 |
| --- | --- |
| Ubuntu Python 3.11 | dev 镜像，不含 DTK |
| Ubuntu Python 3.12 | dtk 镜像，自带 DTK 26.04 |
| RockyLinux | 镜像 tag 自带 Python 版本，随 `PYTHON_VERSION` 变化；不含 DTK |

#### Ubuntu

Python 3.11：

```bash
docker pull harbor.sourcefind.cn:5443/dcu/admin/base/dev:ubuntu20.04-mpi4.0-gcc9.4-cmake3.19-py3.11-mkl2020.4.304
```

Python 3.12：

```bash
docker pull harbor.sourcefind.cn:5443/dcu/admin/base/dtk:26.04-ubuntu22.04-mpi5.0-gcc11.4-cmake3.29-py3.12
```

#### RockyLinux

根据实际 Docker 镜像 tag 选择对应版本：

```bash
docker pull harbor.sourcefind.cn:5443/dcu/admin/base/dev:rockylinux8.6-mpi5.0-gcc10.3-cmake3.29-py${PYTHON_VERSION}-mkl2020.4.304
```

### 2. 创建并启动容器

请根据实际情况替换以下参数：

| 参数 | 说明 |
| --- | --- |
| `{container-name}` | 容器名称 |
| `{image-name}` | 上一步拉取的镜像 ID 或镜像名称 |
| `{宿主机工作目录}:{挂载目录}` | 宿主机目录与容器内挂载目录 |

```text
docker run -it \
--name={container-name} \
--network=host \
--restart=always \
--privileged \
--ipc=host \
--shm-size=16G \
--group-add video \
--device=/dev/kfd \
--device=/dev/mkfd \
--device=/dev/dri \
--cap-add=SYS_PTRACE \
--security-opt seccomp=unconfined \
-v {宿主机工作目录}:{挂载目录} \
-v /opt/hyhal:/opt/hyhal:ro \
{image-name} /bin/bash
```

### 3. 安装 DTK（按需）

根据目标系统与 DTK 版本选择对应安装方式。

#### RockyLinux 8.6

DTK 26.04：

```text
wget https://download.sourcefind.cn:65024/file/1/DTK-26.04/Rocky8.6/DTK-26.04-Rocky8.6-x86_64.tar.gz
tar -zxvf DTK-26.04-Rocky8.6-x86_64.tar.gz -C /opt
cd /opt
rm -rf dtk
mv dtk-26.04 dtk
```

#### Ubuntu

DTK 26.04：

```text
wget https://download.sourcefind.cn:65024/file/1/DTK-26.04/Ubuntu22.04/DTK-26.04-Ubuntu22.04-x86_64.tar.gz
tar -zxvf DTK-26.04-Ubuntu22.04-x86_64.tar.gz -C /opt
cd /opt
rm -rf dtk
mv dtk-26.04 dtk
```

### 4. 安装 aillvm

安装过程约需 2 分钟。

```text
wget http://42.228.13.241:18000/ai_cc/Nightly/hcu_llvm_installer.sh 
chmod +x hcu_llvm_installer.sh
bash hcu_llvm_installer.sh --major 1.0.0 
rm -rf hcu_llvm_installer.sh
```

可选检查：

```text
ls /opt/dtk/aillvm/bin/
```

如能看到 `clang`、`clang++`、`llvm-ar`、`ld.lld` 等常用编译工具，则说明安装完成。

### 5. 设置 Python 环境变量

Python 3.11：

```bash
export PYTHON_BIN=python3.11
```

### 6. 编译 jax

可选：设置 PyPI 镜像源。

```bash
printf '[global]\nindex-url = https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple\ntrusted-host = mirrors.tuna.tsinghua.edu.cn\ntimeout = 120\n' > /etc/pip.conf
```

拉取 JAX 源码(本项目)：

```bash
git clone -b main https://github.com/HYGON-AI/jax-das.git
```

进入源码目录并安装编译依赖：

```bash
cd ./jax-das/scripts/

# 安装 Python 打包/修复工具
python3 -m pip install "wheel>=0.40" auditwheel patchelf

# 编译（耗时较长，建议在 tmux / screen 中执行）
bash build_jax_dtk.sh
```

编译产物位于 `./dist/`，包含以下 4 个 wheel 文件。

示例：

```text
jax-0.10.2+das.opt1.dtk{dtkversion}-py3-none-any.whl
jax_rocm6_pjrt-0.10.2+das.opt1.dtk{dtkversion}-py3-none-manylinux_{glibcversion}_x86_64.whl
jax_rocm6_plugin-0.10.2+das.opt1.dtk{dtkversion}-{cpversion}-{cpversion}-manylinux_{glibcversion}_x86_64.whl
jaxlib-0.10.2+das.opt1.dtk{dtkversion}-{cpversion}-{cpversion}-manylinux_{glibcversion}_x86_64.whl
```

其中，`{cpversion}` 对应 Python 版本，例如，Python 3.12 对应 `cp312`；dtk`{dtkversion}`对应 dtk 版本，如 dtk26.04 对应dtk2604；`{glibcversion}` 对应编译镜像的 glibc 版本，例如，glibc 2.28 对应 `2_28`。

### 7. 安装 jax 

安装 `./dist/` 目录下的 wheel 文件：

```bash
pip3 install absl-py hypothesis flatbuffers \
    ./dist/jax*.whl
```

### 8. 环境初始化与验证

```bash
source /opt/dtk/env.sh
python3 -c "import jax; print(jax.devices())"
```
确认能够列出 HCU 设备。

每次新开 shell 时执行以下命令完成环境初始化：

```bash
source /opt/dtk/env.sh
```

如果不想每次新开shell都要执行source，将 source 命令添加到 ~/.bashrc

```bash
cd ~
echo "source /opt/dtk/env.sh" >> ~/.bashrc
source ~/.bashrc
```

---

## 已知问题

### `jax.devices()` 报错或仅返回 CPU 设备

请依次检查以下项目：

1. 容器启动时是否透传了 `/dev/kfd`、`/dev/mkfd`、`/dev/dri`，并挂载了 `/opt/hyhal`；
2. 是否已执行 `source /opt/dtk/env.sh`。