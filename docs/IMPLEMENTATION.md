# 实施规格、模块接口与没有完成的生产层

## 一、实际模块对应

| Idea | 可运行实现 | 端到端边界 |
|---|---|---|
| 01 | frontier.solve_frontier/parameter_feasibility/dual_face_gain/solve_smooth_step；uncertainty | full-pool LP；没有百万样本复杂度证明 |
| 02 | atoms.nuisance_residual/crossfit/OMP/learn_dictionary；column_generation | 原创OMP参考，不是EKFAC原论文复现 |
| 03 | trajectory.quadratic_rollout/adjoint/order | 精确二次；没有真实LLM HVP轨迹缓存 |
| 04 | observational MAR/DR/maturity；data CSV adapter | 假设/propensity由外部提供，未知时拒绝 |
| 05 | allocation、rl；run_tiny_rl | 分类策略完整组；没有HF-GRPO rollout集成 |
| 06 | acquisition.accept_pairs/rank_acquisition/best_acquisition_batch | 真实未标注响应预测和query服务未提供 |
| 07 | exploration.pagerank/absorbing_value/safe_mixture | 已知P/Q；没有从真实日志拟合退出过程 |
| 08 | decoding.CatalogTrie/exact_speculative_step | 单item/有限目录；没有beam/GPU加速 |
| 09 | science.ScientificCandidate/MethodDatasetBundle/greedy_coverage | 门禁不等于独立执行沙箱 |
| 10 | science.gaussian_information/posterior_update | 线性Gaussian，不是DFT/湿实验 |

函数的准确名称以源码为准；表中部分是模块内功能简称。不存在用空pass伪造核心功能的模块。

## 二、规范JSONL

每行Record需要`record_id,user_id,history,history_times,timestamp,item_id,scene,labels,split`。labels用JSON null表示未观测，而不是0；history_times必须严格早于该事件。split必须属于train/selection/test；跨分区record_id不能重复；全局时间顺序必须成立。合成示例见 `data/examples/records.jsonl`，不含真实用户。

原始prompt字段不直接送模型；训练器用canonical字段重建输入，避免把标签或任意控制语句混入。若需要物品描述和用户自然语言约束，必须扩展schema、时间可用性和模板测试；当前默认只是ID历史/候选/场景，不能据此充分证明语言模型独有价值。

## 三、本地真实日志适配

```bash
python scripts/prepare_kuairand.py --help
```

适配器从本地CSV读取官方逐事件字段，做时间拆分、历史构建、反馈mask和资源guard。不会自动下载大数据集、不会凭is_rand填propensity、不会读取整月统计。`--max-rows`是内存上限而不是悄悄只保留前几行；超过时拒绝并要求显式分片。需要的具体参数以help为准。

## 四、SFT共享训练循环

TinyFeedbackModel随机初始化，没有语言预训练。HFFeedbackRanker读取**本地**标准HF模型，`trust_remote_code=False`，LoRA dropout为0，固定反馈头。损失为每条记录已观测目标的平均BCE；参考目标则各自按可观测数量求均值。两者是不同但明确的estimand，响应矩阵使用实际训练loss梯度，不假装缺失标签不存在。

`gradient_snapshot`在eval模式计算逐例训练梯度和各目标参考梯度。梯度子空间明确，内存guard上限明确。选择后执行确定性全池加权SGD/AdamW：当前tiny代码不是只计算被选非零样本，因此没有节省训练token的主张。探索floor在求解前进入响应，不是结果出来后随意混入。

`frontier`的matrix certificate只针对当前参考梯度矩阵。只在全参数、plain SGD和非正局部margin时执行abstain；正margin不保证有限步改善。新增smooth-step内核未自动接入tiny训练，因为没有经过独立验证的population梯度误差与neighborhood smoothness。把经验Hessian norm当全程Lipschitz常数会制造假保证。

HF保存tokenizer、backbone/adapter、feedback_head和ranker_config。`load_saved`保留本地base路径，搬机器时需保持该路径或明确更新配置。此HF保存/加载入口仅做语法检查，未在本次依赖/权重条件下集成运行。

## 五、RL边界

`rl.py`要求完整组和显式objective weights，p/q作用在prompt级，不自归一化。合成分类RL使用LOO基线和已知完整reward，执行后policy reward可直接求期望。每步surrogate值接近0是ratio=1与组中心化造成，**梯度不为0**；结果记录gradient_norm验证训练并非空转。

真实候选只有被点击物品有观测时，不能凭空为所有候选构造完整reward。生产集成须选定可验证命中奖励/支持限制或真实反馈oracle，再实现policy generation、完整组收集、冻结旧logprob、KL和目标分量。本库没有掩盖这部分缺口。

## 六、分布式扩展应如何做，当前为什么拒绝torchrun

当前脚本检测WORLD_SIZE并拒绝>1，避免各rank独立选q却声称全局有效。扩展设计为：按record_id一致分片候选；每rank计算本地G/sketch；参考梯度按标签分子和计数all-reduce而非简单均分rank均值；集中或分布式求全局q；广播样本ID、版本和概率；每rank根据global normalization计算loss，避免DDP平均再除一次。最后用2进程CPU gloo先验证与单进程更新一致，再迁移GPU。

这只是明确的实施规格，**未实现DDP**。用户本轮没有要求特定GPU数，不能把旧项目的设备配置当作本任务资源事实。

## 七、完整复现与来源

`scripts/run_all.py`的命令均为真实入口；`configs/`可以由`scripts/run_config.py`读取；`scripts/validate_repo.py`验证本地Markdown链接、源码编译和来源卡一致性，不验证外部URL可访问性或论文事实。

第三方官方代码仅记录来源，没有clone副本或伪造commit。要补基线，应先记录license/commit/配置/数据，再运行官方结果和本问题等预算适配。现有脚本和CI不等于论文基线已经复现。
