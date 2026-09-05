# 实际验证报告

日期：2026-09-05。以下数值来自本次本地执行的JSON/log，不是文献结果、预期结果或用户旧实验。运行环境：Python3.13.5、NumPy2.3.5、SciPy1.17.0、PyTorch2.10.0+cpu；CUDA不可用。HF/PEFT权重与真实数据未加载。

## 1. 测试与可复现范围

本轮最终单元测试 **110 passed**，完整输出见[pytest.txt](../results/pytest.txt)与[JUnit XML](../results/pytest.xml)。另运行10方向机制示例、精确二次互补干预、12次tiny SFT和6次完整组分类RL。Python语法、本地Markdown链接、来源卡路径由验证脚本检查；没有验证外部网络链接永远可用，也没有在GitHub Actions上运行本次CI。

`scripts/run_all.py`可重跑。非确定性CPU耗时不应逐小数比较；模型数据/种子和算法结果保留可重现设计。本次未进行GPU性能、内存峰值或第三方官方baseline复现。另已实际完成editable离线安装与import、HF脚本help和数据适配脚本help检查；help成功不等于HF模型训练已运行。另已实际完成editable离线安装与import、HF脚本help和数据适配脚本help检查；help成功不等于HF模型训练已运行。

## 2. 构造性机制证据

| 检验 | 实际结果 | 解释范围 |
|---|---|---|
| 参数空间可共同下降但当前池不足 | 参数问题正margin，旧数据margin−0.5；补入bridge变1.0 | 可达域区别存在；两种范数域不比较绝对差值 |
| 单例无益、配对有益 | 旧/单例margin0；配对margin0.5 | 非加性、非次模反例 |
| 二次实际一步更新 | 两个目标各下降0.04875，η=.1 | 给定损失下真实参数更新，不是benchmark |
| 残差不变性 | 最大误差约1.31e−14；正交误差约4.03e−14 | 指定线性空间的浮点精度检验 |
| 二次顺序恒等式 | 误差约5.28e−17，顺序效应范数约.02563 | 二次系统，不是LLM先修能力 |
| vector预算配置 | 最大二阶矩2.9→2.53569 | 固定合成分布下的配置目标 |
| 小参考集误差 | 经验响应正，但保守下界负 | 主动拒绝“已证明改善”的错误结论 |
| 单步探索 | 最大混合ε=.5，reward下界=.6 | 有效给定下界下单步保证 |
| SID与投机 | 解析概率误差在浮点精度内 | 单item有限目录，无速度主张 |

完整机制数据：[mechanisms.json](../results/mechanisms.json)、[complementarity.json](../results/complementarity.json)。smooth-step、完整dual-face、已知噪声风险、缺失/支持、科学bundle门禁等额外检验记录在单元测试中，不逐个伪造performance表。

## 3. Tiny SFT：负结果必须保留

随机初始化的小型多反馈神经推荐器，非预训练LLM；3个种子，4个方法，各25步，相同初始化、候选池随机种子和reference预算。数据为合成场景分布差异。每个目标为BCE，**越低越好**。

| 方法 | 目标1平均test BCE | 目标2平均test BCE |
|---|---:|---:|
| uniform | 0.708354 | 0.690102 |
| scalar | 0.725111 | 0.698126 |
| frontier | 0.719656 | 0.696674 |
| atoms_frontier | 0.719656 | 0.696674 |

结论：该简单局部配置没有稳定改善泛化；在这3种子均值上，uniform两个目标都更好。atoms提名后进行全池精确定价，收敛到相同解，同时增加计算。不能宣称“梯度原子提升推荐效果”“联合局部margin保证test效果”或“比全数据更好”。

这不否认用户原有SFT观察，因为数据/模型不同且没有拿到旧日志；它限制了**本次提出的简单原型**可以声称什么。没有在看完这些test后调参直到取胜；新增smoothness测试是独立构造反例，不是偷偷改这个结果。

原始逐种子和每步记录：[tiny_sft.json](../results/tiny_sft.json)。训练选择没有使用test标签；选择器重复使用selection，所以仍存在selection过拟合风险。

## 4. 分类RL：微小差异不是实证通过

4个prompt、3个动作、2个reward、组大小4，每次训练3840个rollout，3个种子。`base_prompt`按预先固定p=(.4,.3,.2,.1)抽样，**不是均匀分布**；早期代码名称uniform_prompt已修正为base_prompt。variance_prompt用minimax二阶矩proxy和p/q纠正。

| 方法 | 目标1平均期望reward | 目标2平均期望reward |
|---|---:|---:|
| base_prompt | 0.529289 | 0.546160 |
| variance_prompt | 0.529775 | 0.546242 |

差异很小，没有统计显著性结论；也没有预训练LLM、真实用户或线上留存。所用a_im是解析single-rollout score矩proxy，不是完整组矩oracle。完成的是完整组估计/预算接口验证。每步surrogate value可能接近0，但gradient_norm非零；这是中心化优势和ratio=1的代数结果，并不代表训练空转。

原始记录：[tiny_rl.json](../results/tiny_rl.json)，汇总：[summary.json](../results/summary.json)。

## 5. 正式通过/不通过

数学/接口门：所列假设下通过；代码运行门：CPU参考路径通过；新颖性门：01+06可继续研究，02/08不能独立通过；真实推荐效果门：**尚未通过，且本轮已有负证据**；科学执行/物理验证门：未执行。

下一层证据需要真实数据、小规模本地预训练模型、独立calibration/test和危险基线。不得把这些待办写成已完成，也不得把数学恒等式改写成用户体验改善。
