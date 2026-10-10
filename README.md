# CrossDex reproduction

CrossDex 在 Isaac Gym 上的复现代码与实验记录。

[复现报告](docs/reproduction_report.md) · [代码与结果索引](docs/code_and_results.md) · [运行依赖说明](docs/environment_and_paths.md)

目前完成了四种手的动作映射、单魔方四手共享 PPO、单魔方 DAgger，以及糖盒和番茄汤罐的 PPO 实验。尚未完成论文的 45 物体联合蒸馏和未见手型评估。

仿真环境、奖励、动作控制及 PPO/DAgger 算法主体复用 [官方 CrossDex](https://github.com/PKU-RL/CrossDex)。本仓库收录新增的实验组织、评估、映射诊断及回放脚本。官方核心文件的校验记录见 `results/provenance/source_audit.json`。

| 目录 | 内容 |
| --- | --- |
| `crossdex_validation/` | 实际使用的训练、评估、映射和录像脚本 |
| `configs/` | 实际实验配置及四种手的映射网络配置 |
| `results/` | 成功率汇总、各评估种子的结果和来源记录 |
| `viewer/` | 本地四手 DAgger 窗口回放入口 |
| `docs/` | 报告、代码索引和环境说明 |

这是原云端实验的代码与结果归档。脚本保留原路径；在其他机器运行前，需要按环境说明调整路径并准备模型、数据和官方依赖。权重、数据集、录像、Isaac Gym SDK 和 Python 环境不包含在本仓库中。

仅整理本地文件，不会自动安装依赖或启动训练。
