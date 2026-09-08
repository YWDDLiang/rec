# E005 两域基线阶段

单种子开发结果。每域768名用户、每人一个固定查询；最终test未读取。

| 配置 | Office R@10 | Office NDCG@10 | Industrial R@10 | Industrial NDCG@10 | 总分钟 |
|---|---:|---:|---:|---:|---:|
| single_office | 0.167969 | 0.137023 | 0.003906 | 0.001179 | 21.01 |
| single_industrial | 0.002604 | 0.001643 | 0.117188 | 0.086971 | 20.72 |
| joint_uniform | 0.169271 | 0.137818 | 0.114583 | 0.087648 | 20.76 |
| joint_empirical | 0.167969 | 0.139705 | 0.115885 | 0.089885 | 20.90 |

单域模型的未训练域列仅作诊断；单域与共享模型应按各自声明的使用方式比较。

四组更新数和样本使用量一致，实际tokens及每域训练暴露不同；下降不能单凭此表归因于梯度冲突。
来源为MiniOneRec公开数据，自有Qwen3-1.7B LoRA/合法SID SFT；不是其论文全参数SFT/RL复现。

阶段决策：Baseline training is learnable; review joint tradeoffs and start the matched method stage

配对差异、初始化回退与成本详情见summary.json。
