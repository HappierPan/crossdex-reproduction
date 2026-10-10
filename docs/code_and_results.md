# 报告对应的代码与结果

脚本路径均相对仓库根目录。`crossdex_validation/` 下保留原文件组织结构。

| 报告内容 | 主要入口 | 结果目录 | 实际配置 |
| --- | --- | --- | --- |
| 两手 PPO 10000 次 | `crossdex_validation/ppo_10000/run_validation.py` | `results/two_hand_ppo_10000/` | `configs/experiments/two_hand_ppo_10000.json` |
| 四手 PPO 10000 次 | `crossdex_validation/four_hand_trial/run_10000_validation.py` | `results/four_hand_ppo_10000/` | `configs/experiments/four_hand_ppo_10000.json` |
| 单魔方 DAgger | `crossdex_validation/single_object_dagger/run_trial.py` | `results/single_object_dagger/` | `configs/experiments/single_object_dagger.json` |
| 糖盒、汤罐 PPO | `crossdex_validation/multi_object_teachers/train_teacher.py` | `results/sugar_box_ppo_10000/`、`results/tomato_soup_ppo_10000/` | `configs/experiments/` 中对应 JSON |
| 糖盒追加训练 | `crossdex_validation/sugar_box_followup/run_budgeted.py` | `results/sugar_box_followup/` | `configs/experiments/sugar_box_followup.json` |
| Ability、SVH 映射准备及训练 | `crossdex_validation/additional_hands/export_grab_pose45.py`、`train_full_mappings.py` | `results/mapping/` | `configs/mapping/` |
| 原映射质量及偏差诊断 | `crossdex_validation/check_mapping_quality.py`、`mapping_origin/` | `results/mapping/original/` | `configs/mapping/` |

PPO 评估入口为 `crossdex_validation/evaluate_existing_policy.py`；DAgger 评估入口为 `crossdex_validation/single_object_dagger/evaluate.py`。

正式四手评估使用 101、202、303、404、505 五个评估种子，每个种子 256 个环境，每个环境统计第一个完整回合，共 1280 回合。这里是评估种子，不是五次独立训练。成功判定调用官方环境。

四手 PPO 的录像脚本在 `crossdex_validation/four_hand_trial/replay_10000/`，DAgger 的录像脚本在 `crossdex_validation/single_object_dagger/`。`viewer/` 收录本地窗口入口，但没有包含原分发包内的 CrossDex、权重和资产，不能把此目录单独当成完整回放包。

Allegro、Shadow 的最初映射训练在本地完成；本仓库保存配置和已有诊断证据，尚未补齐那两次训练的原始日志及本地入口。Ability、SVH 的数据准备及训练脚本已收录。

`results/summary/success_rates.csv` 和 `experiment_results.json` 是结果汇总；具体成功回合可在各实验目录的逐种子 JSON 中核对。糖盒追加训练从已有权重初始化，优化器重建，不能当成完整恢复优化器状态的连续训练。

`results/checkpoint_index.json` 保存原实验中的模型引用和校验信息，不包含权重本身。`results/provenance/package_manifest.json` 记录本次复制的源文件和 SHA256；其他历史证据清单可能引用未随仓库上传的大文件。
