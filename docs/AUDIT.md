# 严肃自审、debug记录与剩余风险

本文件记录此次实施过程中确实发现并修正的问题。它不是外部独立安全审计，也不保证无剩余bug。完整测试命令与环境见VALIDATION。

## 一、实际修正记录

| 发现 | 影响 | 修正与回归 |
|---|---|---|
| 最初92项测试中4项按错误接口断言 | 测试把tuple/dict/列表返回当成另一种契约 | 按实际声明的API修正测试；保留`results/pytest_initial.txt`，不将接口失配美化成方法失败 |
| 训练脚本误解dictionary/allocation返回对象 | 可能索引错误或训练入口失败 | 正确读取codes/q键，运行实际tiny SFT/RL |
| 单一dual witness可能误报新增数据充分性 | 非唯一最优对偶情形推论不成立 | 引入完整optimal-face判据，加入退化反例与有限差分测试 |
| acquisition优先级缺少旧margin项 | 不同成本下候选排序不符合定义 | 在除cost之前减old_margin，添加相应测试 |
| DR/importance的浮点动作ID被静默转整数 | 不合法数据会被当成另一个动作 | 明确要求有限整数，非法值抛错 |
| trie路径计算把非法q吞成概率0 | 坏分布看起来像不可达物品 | 在计算前验证q与item index；只对真实不可达前缀返回零 |
| Gaussian后验缺少完整PSD/有限观测检查 | 非协方差输入可能产生虚假信息 | 对称/PSD/finite guard与回归测试 |
| 直接调用split validator未逐记录检查history | 代码创建Record后可能绕过future-history检查 | validate_splits调用record.validate；加入未来历史回归 |
| 大量同时间戳事件逐条截断历史 | 后面的同刻事件丢失真正过去上下文 | 按时间戳块处理，块内历史一致，块后统一截断；12条tie回归 |
| raw正线性margin忽略有限步overshoot | 看似有利更新实际上增加损失 | 增加已知smoothness下界优化内核与大步反例；未知界不认证 |
| RL基线名uniform但实际按非均匀p | 误导实验解读 | 改名base_prompt，保存实际p和gradient_norm |
| HF保存缺少重构元信息 | 模型头与adapter难恢复 | 保存ranker_config并提供local-base加载入口；仍标记未运行HF集成 |
| 用atom提名后精确LP与无atom一样 | 无法支持atom提升效果，且额外成本 | 保留真实负结果，把atom从必选贡献降为可选 |

初始失配是测试/接口问题；时间戳、索引、判据、成本、证书则是实际实现/推论风险。两类分开，不声称“修好四个错误就证明全部理论”。

## 二、数学claim专项审计

LP证书只覆盖给定模型状态、给定响应矩阵和所用参数子空间；非正共同margin也可能是局部驻点，不意味着全局目标根本冲突。变量尺度、trust region和探索floor必须一致。

完整face的批次充分性要求新增列不改变旧可行域；新数据引入时重新平均base分布会破坏该条件。区间估计判据相对旧真实最优的比较需要旧上界，不是两个乐观点估计相减。

smooth-step只有外部可靠曲率和梯度误差时成立；SLSQP返回的是数值可行检查，不是独立最优dual-gap证明。AdamW只把rawgrad当proxy时，不提供真实更新证书。共享验证集的自适应复用也不满足简单Hoeffding独立条件。

原子残差不识别因果兴趣；cross-fit不保留in-sample exact orthogonality；dictionary编号没有跨时段固定语义；随机投影本身不自动提供原空间保证。

## 三、推荐科学专项审计

缺失反馈不填0；未曝光动作没有真实reward；同一click字段可能在UI间语义不同；评论不是曝光日志。已知随机标记不是完整propensity。标签噪声纠正的独立对称假设不能用于LLM同源自评。

候选目录与用户历史必须决策时可用；full-history/rolling-history/冷用户不同协议不能混合。普通BCE多头训练不自动意味着Top-K策略或长期用户价值提高。当前HFprompt主要是ID字段，不能据此突出语言理解独有贡献。

## 四、尚未实现/尚未验证

真实LLM权重的加载训练与保存回读；官方数据完整下载与训练；所有第三方方法的公平baseline；HVP长轨迹缓存；未标注响应预测和真实标注服务；HF-GRPO rollout及奖励服务；退出/删失估计；GPU树状投机；科学执行sandbox；DFT/湿实验；DDP。

没有在这些位置用`pass`冒充算法；相应原型接受显式给定的模型量或oracle矩阵，文档解释其获取成本。完整工业实施仓库仍需这些集成；当前交付为有测试的研究实现仓库。

## 五、代码/来源/隐私

没有GitHub写操作、没有外部收费调用、没有下载第三方权重或vendor其代码。原始PPT仅提取文字用于来源核对；公开上传前用户可按自身授权决定是否保留`ppt_text_extraction.json`，本库不假定PPT中引用图文都可再许可。

源代码MIT只覆盖原创内容；外部模型和数据遵循原许可。HF关闭trust_remote_code；日志示例为合成，不包含真实用户ID或敏感偏好。单进程guard避免未经测试的DDP给出错误全局权重。

## 六、最终意见

作为理论核验、领域调研与实现起点可交付。不能批准“已经有效的LLM4Rec方法”“十个idea全部通过”“WWW结果ready”等表述。最有价值的下一步是用真实多反馈数据检验互补机制与独立泛化；不是为当前负结果增加更多组件。
