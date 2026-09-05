# 下一位研究者/模型的交接

先读README、VALIDATION、AUDIT，**不要从“测试全绿”推断方法已经有效**。用户明确不希望将任务说成数据压缩，目标是数据选择带来更好的多目标LLM推荐；不要再把原平均梯度匹配作为主叙事。

主问题01+06：带符号目标响应、当前池的可达性、非加性互补批次、独立获取反馈。标准LP理论不是新颖性本身。02 atoms在现tiny实验中没有额外效果且更慢，除非新证据支持，不应强行升回主贡献。03时序只完成二次模型，真实HVP层待实现。

最关键未解决项：真实多反馈数据和语言任务；reference/calibration独立性；MoRec/DEALRec/GORACS/MiniRec/Filter-then-Weight/GREATS等危险基线；未标注获取前响应预测；真实LLM SFT/RL；分布式成本。不要用模型自评合成“真实用户偏好”来跳过数据缺口。

可立即执行 `python scripts/run_all.py`。本库tiny SFT负结果应完整保留。后续改法必须先在selection/calibration上定案，再评估新外层test；不能反复看现test直到调出胜利。把模型和数据扩大不是修复机制问题的替代方案。

Research集成应只新增llm4rec/，不覆盖crystal/和现有README。用户自己上传；本轮未做GitHub写操作。查看docs/provenance获取读取时commit和输入文件指纹。
