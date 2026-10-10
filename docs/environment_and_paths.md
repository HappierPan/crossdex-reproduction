# 环境与路径

这些脚本是已完成实验的归档，未改写成跨机器的一键运行包。请先阅读脚本参数；部分入口默认会启动训练。

## 官方代码

原环境使用 `/home/siyu/CrossDex`。官方核心文件校验基于 commit：

```text
b94c567eb6a6d3a142279cce32475af3c986b8b3
```

获取官方代码后，以 `results/provenance/source_audit.json` 核对核心文件。官方许可证和安装说明请查看原仓库。本仓库没有复制官方项目或商业 SDK。

## 运行依赖

使用已有的 Isaac Gym Python 环境；不要用系统 Python 替代。需要 NVIDIA GPU 驱动、Isaac Gym Preview 4、匹配的 PyTorch，以及官方 CrossDex 对应版本的 IsaacGymEnvs、Hydra/OmegaConf 等依赖。映射数据准备另需 dex-retargeting、MANO 模型及 GRAB 数据。具体版本以原项目环境记录和官方 README 为准，这里没有生成未经验证的 pip 安装清单。

## 需要准备和调整的路径

- `/home/siyu/CrossDex`：官方仓库及资产、PCA 基、四种手的映射权重。
- `/root/crossdex_validation`：原实验脚本和输出路径；当前代码副本在本仓库同名目录。
- `/root/autodl-tmp/crossdex_experiments`：原 PPO 实验及基础配置位置。实际配置已备份在 `configs/experiments/`。
- `/root/crossdex_checkpoints`：DAgger 所需教师模型及配置。
- 本地 GRAB、MANO 和 dex-retargeting 数据、资产路径。

复制目录不会自动改变脚本中的路径。换机器使用前应检查 `REPO`、`BASE`、`ROOT`、`OUT`、`TEACHER` 等变量和子进程调用的绝对路径；不要直接运行历史队列或训练入口。

模型权重另行提供。原 Isaac Gym checkpoint 主要保存网络权重，不能据此声称完整恢复优化器和随机状态。仅有 YAML 配置也不能代替映射权重。

窗口回放还需要可用的本地图形会话。云端无窗口录像成功，不代表本地窗口配置已验证通过。
