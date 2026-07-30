# A 股大盘复盘数据与风险契约

本文档描述 A 股大盘复盘从数据采集、结构化证据、风险判定到报告和下游个股分析的统一契约。实现入口为 `src/market_analyzer.py`，证据构建位于 `src/services/a_share_review_evidence.py`，输出一致性校验位于 `src/services/market_review_consistency.py`。

## 权威口径与来源

同一指标只能有一个权威值。概览数据允许保留为兼容字段，但不得覆盖日期化事件池或在报告中与其并列展示。

| 指标 | 权威口径 | fallback 与质量要求 |
| --- | --- | --- |
| 上涨/下跌/平盘、成交额 | `get_market_stats(purpose="market_review")` 的首个有效 provider | 记录 `_provider`；失败时标记 `unknown`，不反推市场宽度 |
| 涨停、炸板、昨日涨停、跌停 | 实际交易日的四个 event pool | provider 按 manager 优先级有序 fallback；成功空集合是有效证据，失败链写入 `data_quality.sources/errors` |
| 指数现价 | 同交易日 runtime snapshot | 缺失或日期不匹配时使用不晚于交易日的历史收盘，并标记 `historical_close_fallback` |
| MA5/MA10/MA20 | 五个主要指数各自不少于 20 根、且不晚于交易日的日线 | 上证、深证、创业板、科创50、沪深300主动拉取；不足五个不得输出“主要指数平均涨跌” |
| 外围科技 | 纳指与半导体代理的带时间戳行情 | 两项和时间戳齐全才是 `ok`；否则最多黄色风险状态 |
| 主题与观察票 | 板块/概念排行、日期化涨停池和容量/换手证据交叉 | 只有排行不得确认主线；一字或严重缩量票只能是 `observation_only` |

每个证据源记录 `provider`、`dataset`、`source_role`、`as_of`、`trade_date`、`status` 和 `failure_reason`。展示层消费规范化结果，不自行选择第二套来源。

## 规范化快照

`market_review_payload.normalized_review_snapshot` 是 Prompt、Markdown、通知、历史记录、Market Light 和 `daily_market_context` 的共同输入，版本为 `a-share-market-review-snapshot-v1`。主要字段如下：

| 字段 | 语义 |
| --- | --- |
| `trade_date/generated_at/session_phase` | 实际数据交易日、生成时间和会话阶段 |
| `indices` | 五指数现价、涨跌、MA5/MA10/MA20、20 日区间、时间与来源 |
| `breadth/turnover/sentiment` | 市场宽度、成交额和日期化情绪结构的唯一权威值 |
| `external_context` | 纳指、半导体代理、时间戳和数据质量 |
| `theme_evidence` | 全部候选及最多三个 `actionable_directions` |
| `stock_candidates` | 角色、交易资格、触发、确认、失效、参考日期与单票仓位上限 |
| `risk_assessment` | 原始热度、风险状态、仓位模式、仓位上限和触发门槛 |
| `data_quality/field_sources` | 缺失、污染、失败原因和字段来源 |

旧字段 `market_light.score/status/label` 和 `a_share_evidence` 继续保留。新增字段均为 additive；旧历史缺少新字段时按原兼容路径读取，但不会获得 A 股确定性风险升级。

## 热度与风险算法

`market_light.score`、`market_heat_score` 和 `raw_heat_score` 仍是市场热度，使用市场宽度 45%、主要指数 35%、涨跌停结构 20% 加权。A 股指数维度只有五个主要指数全部覆盖时才为 available；非 A 股维持至少一个主要指数即可。

热度不直接决定 A 股 `risk_state`。风险状态按下列硬门槛计算：

| 状态 | 条件 | `position_mode` | 组合仓位上限 |
| --- | --- | --- | ---: |
| `green` | 数据质量完整；五指数均有 MA；指数、情绪、容量核心/可交易前排和外围科技全部通过 | `incremental_attack` | 60% |
| `red` | 外围科技显著负反馈且成长指数位于 MA5/MA10 下方；或热度低于 35；或成长指数弱且炸板率/昨日涨停溢价同时触发风险 | `defense_only` | 10% |
| `yellow` | 其他未全部通过或覆盖不完整的情况 | `confirmation_trial` | 30% |

结构化风险阈值包括：炸板率大于 30%、昨日涨停溢价中位数小于 0%、疑似一字/无换手占比不低于 25%、纳指不高于 -1.5% 或半导体代理不高于 -2.0%。外围科技负反馈只有与创业板/科创50短均线弱势共同出现时才形成 `external_tech_veto` 红灯否决。

## 主题与观察票约束

- `confirmed_mainline` 必须同时具备强度、涨停扩散、至少二板高度、50 亿元以上容量核心和可交易前排。
- `mainline_candidate` 也必须具备涨停扩散和可交易前排；两类可操作方向合计最多三个，超出部分降级。
- CPO、PCB、半导体分别判断，不把单一科技分支的修复外推到全部科技。
- 相近低容量主题可压缩为上位产业链，例如乳品/零售归入消费链、化学原料/制品归入化工链。
- 一字或严重缩量票的 `trade_eligibility=observation_only`、`position_cap_pct=0`、`trigger_type=无交易触发`。
- 条件交易票必须提供确认条件、失效参考和不高于 5% 的单票上限；报告必须提示 A 股新开仓 T+1 风险。

## 输出一致性与降级

报告固定采用九段式顺序：数据口径、热度与风险、指数/旧主线/外围、不能买什么、可操作方向、升级条件、观察票资格、仓位模式、风险与未验证项。

在持久化和通知前，`ensure_market_review_consistency()` 会校验情绪数字、缺失字段描述、热度/风险措辞、主题数量、观察票买入措辞、指数覆盖和日期。发现矛盾时不继续传播模型 Markdown，而是用同一规范化快照生成保守模板；错误写入 `data_quality.errors` 和 `consistency_fallback`。`daily_market_context` 优先读取结构化 `risk_state` 与 `position_cap_pct`，旧文本解析仅作兼容 fallback。

## 2026-07-29 固定回归场景

`tests/fixtures/a_share_review_20260729.json` 固定以下输入和预期，用于防止口径与风险回归：

| 验收项 | 旧行为 | 新契约 |
| --- | --- | --- |
| 涨跌停 | 概览近似值 83/13 与日期化值并存 | 只展示日期化 81 涨停、14 炸板、9 跌停 |
| 昨日涨停溢价 | 同时给出 `+0.42%` 和“未提供” | 样本 61，均值 `+0.65%`，中位数 `+0.42%`，不再称缺失 |
| 指数 | 三指数均值掩盖科创弱势 | 五指数均有 MA5/MA10/MA20；覆盖不足不计算平均值 |
| 热度与风险 | 高热度直接给出“强势，可进攻” | 热度仍可高，但纳指 `-2.10%`、半导体代理 `-4.50%` 且成长指数弱，判定 `red/defense_only/10%` |
| 科技 | 泛化成统一科技修复 | CPO、PCB、半导体分别归类和确认 |
| 主题 | 单日涨幅把消费直接确认为第一主线 | 可操作方向不超过三个；缺容量核心的消费方向降级 |
| 观察票 | 一字高标给出弱转强买点 | 爱丽家居为 `observation_only`，无交易触发，单票上限 0% |
| 下游 | Markdown、payload、通知和个股上下文可能不一致 | 四者消费同一快照；冲突报告触发保守模板 |

## 兼容、风险与回滚

本变更不新增配置、数据库表或平行复盘入口。主要风险是旧客户端将 `market_light.status` 当作热度颜色；A 股该字段现在代表确定性风险状态，热度应读取 `score/market_heat_score`。告警仍可读取旧快照，新快照会附加风险字段。

回滚时可撤销规范化快照、风险算法和一致性 guard，但应同步恢复 Prompt、报告模板、Market Light 文档及 `daily_market_context` 字段消费，避免只回滚其中一层造成契约分裂。历史 JSON 的新增字段为 additive，旧代码会忽略，无需数据迁移。
