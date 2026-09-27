# 实验结果审阅与论文接入

审阅日期：2026-09-27。在线 Notion 页面仍标注 **As of 2026-09-22**。

当前最有把握的结果是 **E2 的自动评分提升与多样性下降并存、E3 已测原生配置下的执行效率、E5-T 的部署差异**。E4a 可以作为合成输入的语义检查。它们可以进入论文，但不能扩大成“等价算法下全面更快”“IR 开销已证明很低”或“异步更快达到相同质量”。

本文中的“可写入”指有可追溯报告、且结论限定在观测范围内；**不等于已独立复核原始训练日志、全部配置或所有验收门槛**。原始记录仓库当前不可访问，按作者指示跳过。本次不猜测缺失的运行数据，不画合成学习曲线，也不从摘要推造误差条。

## 来源与时序

| 来源 | 固定状态 | 用途 |
|---|---|---|
| [在线实验报告](https://app.notion.com/p/UniRL-Experiment-Report-3e324a4934c8816c841ecdbc58d616ff) | 2026-09-27 匿名浏览器读取；正文截至 2026-09-22 | 当前实验状态与主要结果 |
| [结果 PR #2](https://github.com/haonan3/UniRLReport/pull/2) | `ecce5097fa88a2ca336cf39c13e3cfd940e8c1ef` | E2 精确端点及种子变化、E3/E5-T 汇总数字；不能照搬所有解释 |
| [Issue #1](https://github.com/haonan3/UniRLReport/issues/1) | 最后更新 2026-09-11；读取时仍 OPEN、无评论 | 历史评测挂起的诊断，不能作为今天训练仍未启动的依据 |
| [UniRL #462](https://github.com/Tencent-Hunyuan/UniRL/pull/462) | 已于 2026-09-15 合并 | grader 子进程有界清理的上游修复状态 |

小型来源快照、数值输入和校验信息位于 [结果证据目录](artifacts/reported_results/2026-09-22/README.md)。没有将浏览器脚本、登录信息或大型日志放入论文仓库。

## 哪些结果可以写

| 实验 | 可支持的结论 | 数字与边界 | 本次放置 |
|---|---|---|---|
| E2 | 三个训练种子的外部自动评分提高，同时多样性降低 | HPSv3 Δ +8.6102；ImageReward Δ +1.3932；synthetic composition 相对 +53.5%；LPIPS 相对 −24.79% | RQ1 数字表 + 各种子端点图；质量与多样性相邻展示 |
| E3 | 当前 native recipes 下 UniRL 迭代更快 | 约 64.2 vs 111.3 s/iter，约 1.73×；三次过程重复，但估计器及实际 LoRA 参数数不同 | RQ2 表 + 迭代时间图；只展示来源明确的汇总 |
| E5-T | 在已测工作负载下，共享 8 GPU 优于 4 train + 4 rollout | 64.223 vs 146.244 s/iter，约 2.277×；是部署组合的差异 | 与 E3 分开的图面板；RQ3 解释 |
| E4a | 合成测试中两种分组与 ID join 一致 | 两种 scope 各 24/24，通过不等于真实 rollout 已验证 | RQ3 简短文字，明确 synthetic |

### E2：结果可用，但主结论必须包括多样性

- 三个种子方向一致，足以把“自动分数上升伴随多样性下降”作为当前主要观测。训练重复数是 **3**；每个 preference 评测行的 26,112 张图像不等于 26,112 个独立训练重复。
- composition 是本地 synthetic compositional set 的结果，不能改名为官方 GenEval2，也不能把其 +53.5% 相对变化写成 +53.5 个百分点。
- 图中使用种子端点差值。没有所有 checkpoint 的数值时，不从“先涨后平台”等文字重建曲线。
- 原稿 preference 评测约定每个 prompt 一张图；结果报告使用每个 prompt 16 张。需要归档这项协议变更、sampler 和聚合规则，不能把实际端点自动视为与原协议完全一致。
- rollout 50 已有 LPIPS 下降，只支持“截到 50 仍有下降”，不支持“任何 early stopping 都无效”。
- 外部自动评委上升并不能排除 reward exploitation，更不能代替人类偏好研究。原 runbook 的 diversity guard 已触发，因此不写“无 collapse”或“已排除 reward hacking”。
- E2-R 三种子已训练和导出，但还没有 held-out 质量比较。以后可做清楚标注的 native-recipe 质量对照；若要写严格 reproduction，仍需算法与有效工作量对齐。

### E3：速度有观测支撑，公平性结论需要收窄

- flow/CPS、选步、anchor/归一化以及 LoRA target 差异使其成为“配置与系统组合”的比较。不能把 1.73× 全部解释成框架开销差异。
- Notion 报告短程 reward 增量 VeRL 约 +0.10、UniRL 约 +0.075。30 步不足以建立 time-to-quality，不能用一次线性归一化得出的约 1.2× 当作正式结果。
- 报告的“参数匹配后 1.70×”没有可检查的独立匹配运行记录。本次不把它画作另一个 measured control，也不把它当作公平性问题已解决。
- VeRL generation 包含 scoring，UniRL 分开计时。没有统一阶段原始值，先不放 phase-stack 图，也不展示误导性的 generation 3.3×。
- SDPA/FA3 观测时间接近，可写 similar observed times；没有等价性检验，不写两者已证明等价。
- PR #2 正文的 UniRL 重复中位数 `63.62/64.31/64.29` 会得到 `64.29`；该提交表格报告 `64.22`，E5-T 段另列 `63.526/64.223/64.284`。它们不能混为同一批重复数据。图以在线报告精度呈现，数据保留明确来源的表格汇总；复现前不据此计算误差条或显著性。
- 尚未核对定义的 peak memory 数字不纳入本次图表，避免把角色、allocated/reserved 或统计窗口不同的数值放在同一列。

### E5-T：负向部署结果同样值得写

4+4 配置比共享 8 卡慢，且慢于 E3 的 colocated VeRL-Omni。它说明部署选择影响很大，而不是说明 UniRL 对任何布局都更快。训练并行度、驻留、权重发布和 reward 路径同时变化，不能把全部差距归因于通信。报告的 reward 主导诊断可以注明，但没有 phase 数据就不做定量归因。这也不能替代 E5-R。

## 还要推进什么

| 优先级 | 工作 | 为什么需要 | 完成标准 |
|---|---|---|---|
| P0 | E1 已有 checkpoint 的 MATH-500 外评 | 两个目标各一个种子、内部 AIME 不等于主要外部端点 | frozen base + 已保存 checkpoint 的统一评测；按原定义报告 avg@4、token budget、模板与 grader；记录截断和重复评测不确定性 |
| P0 | E4a 的真实 rollout 对照 + E5-R 最小实现 | 它们直接支撑 trajectory IR 的语义与成本，是论文核心缺口 | 同一真实 E4 payload、同一 device/transport/replay 输入，tree 与 flat-ID-join 语义一致；再测结构时间、metadata/driver memory、搬运量及相对模型时间 |
| P0 | E3 来源和计时口径统一 | 当前重复数据不一致，不能正式估计不确定性 | 固定 run IDs、warmup、计时边界、LoRA targets、总 GPU 数及聚合脚本；决定保留 native 比较还是另跑严格匹配实验 |
| P0/P1 | E4 rewrite 修复与短程健康检查 | chat markers 和 CLIP 截断可能影响质量及 grouping contrast | 新版本提取出正确改写、记录截断比例与 frozen-AR 身份；先短跑健康检查，再决定是否完整重跑 |
| P1 | E1 补训练种子；记录 DRPO 引入过程 | 单种子不能判断目标优劣或稳定性 | 主要端点可用后，固定配方与预算，补足复制；报告所有种子和失败 |
| P1 | E2-R 是否承担 reference 主张 | 训练与导出完成不代表质量已比较 | native 对照则明确标签；严格 reproduction 则先数学与工作量对齐，不用仅有外评分数替代 |
| P2 | E6 重启或保留探索性观察 | 正式 sweep 已取消，先导规则未通过 | 如重启，先解决学习预算、评测/RNG、lag 与 buffer 计数、完整时钟；再独立预注册质量目标 |

E4b 三个种子效果方向不同，只能说 **没有观察到一致优势**，不能说已经证明两种 grouping 等价。旧 PR 还披露大量 rewrite 残留 chat markers、超过 CLIP 窗口；不显著并不能证明这些问题无影响，也不能断言唯一限制是种子数。旧配置保留，不静默替换成修复后的结果。

E6 的 D0 pilot 从 0.798（rollout 150）降至 0.773（300），未通过“后者提高”的预注册条件。没有注册质量目标，也没有 time-to-quality 结果。它不是“异步一定无效”的证明。一次 checkpoint 重复评分所得差值区间也不是所有任务和配置通用的噪声下限。

**资源安排建议：**先做 E1 现有模型外评与 E4a/E5-R；同时完成 E3 的来源统一。E2 先补原始产物、可视化和多样性诊断，当前不急于扩模型或增加训练种子。不要优先扩大异步扫描来替代 IR 核心证据。

## 论文与 Overleaf 接入

- `main.tex` 引入生成的 `macros.tex`、`e2_table.tex`、`systems_table.tex`，数字由统一输入生成。
- E2 图展示三个种子的 HPSv3、composition 和 LPIPS 变化，分别保留原量纲；E3/E5-T 图把两个不同问题分成独立面板。
- GitHub 使用 PNG 预览；Overleaf 使用矢量 PDF，所有路径相对仓库根目录。`*.pdf` 的全局忽略只对结果图目录设置例外，主稿编译产物仍忽略。
- abstract、introduction、evaluation、limitations 和 conclusion 同时更新，移除“没有任何 GPU 结果”的旧说法，也不写完整 claim chain 已闭合。
- E6 改为简短的 exploratory 结果说明；原始预注册 grid 保留在 runbook，避免把计划表误读为已跑完的结果。
- 旧的 `main.pdf` / `build/main.pdf` 不能代表本次修改；以重新编译 `main.tex` 的结果为准。

## 本次未做

没有新跑 GPU 训练、没有补造缺失分数、没有改动原始预注册文档或上游训练代码、没有关闭历史 issue，也没有把原始日志未复核的报告提升为独立复现结果。
