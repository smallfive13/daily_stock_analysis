# -*- coding: utf-8 -*-
"""
===================================
大盘复盘分析模块
===================================

职责：
1. 获取大盘指数数据（上证、深证、创业板）
2. 搜索市场新闻形成复盘情报
3. 使用大模型生成每日大盘复盘报告
"""

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from inspect import getattr_static
from typing import Optional, Dict, Any, List

import pandas as pd

from src.config import get_config
from src.core.trading_calendar import get_effective_trading_date
from src.report_language import normalize_report_language
from src.search_service import SearchService
from src.core.market_profile import get_profile, MarketProfile
from src.core.market_strategy import get_market_strategy_blueprint
from src.llm.backend_registry import (
    resolve_generation_backend_id,
    resolve_generation_fallback_backend_id,
)
from src.llm.generation_backend import GenerationError
from src.schemas.market_light import MARKET_LIGHT_REGIONS, MarketLightSnapshot
from src.services.run_diagnostics import record_llm_run, record_llm_run_started
from src.services.intelligence_service import IntelligenceService
from src.services.a_share_review_evidence import (
    AShareReviewEvidenceService,
    aggregate_limit_up_pool as _aggregate_limit_up_pool,
    compute_index_key_levels as _compute_index_key_levels,
)
from data_provider.base import DataFetcherManager

logger = logging.getLogger(__name__)

CN_INDEX_KEY_LEVEL_TARGETS = [
    {"symbol": "000001", "name": "上证指数", "aliases": ("000001", "sh000001", "上证指数")},
    {"symbol": "399006", "name": "创业板指", "aliases": ("399006", "sz399006", "创业板指")},
    {"symbol": "000688", "name": "科创50", "aliases": ("000688", "sh000688", "科创50")},
]


_ENGLISH_SECTION_PATTERNS = {
    "market_summary": r"###\s*(?:1\.\s*)?Market Summary",
    "index_commentary": r"###\s*(?:2\.\s*)?(?:Index Commentary|Major Indices)",
    "sector_highlights": r"###\s*(?:4\.\s*)?(?:Sector Highlights|Sector/Theme Highlights)",
}

_CHINESE_SECTION_PATTERNS = {
    "market_summary": r"###\s*(?:[一二三四五六七八九十]+、)?(?:盘面总览|市场总结)",
    "index_commentary": r"###\s*(?:[一二三四五六七八九十]+、)?(?:指数结构|指数点评|主要指数|大盘风险门槛)",
    "sector_highlights": r"###\s*(?:[一二三四五六七八九十]+、)?(?:板块主线|热点解读|板块表现|热点排序)",
    "funds_sentiment": r"###\s*(?:[一二三四五六七八九十]+、)?(?:资金与情绪|资金动向|情绪温度)",
    "news_catalysts": r"###\s*(?:[一二三四五六七八九十]+、)?(?:消息催化|后市展望)",
}


@dataclass
class MarketIndex:
    """大盘指数数据"""
    code: str                    # 指数代码
    name: str                    # 指数名称
    current: float = 0.0         # 当前点位
    change: float = 0.0          # 涨跌点数
    change_pct: float = 0.0      # 涨跌幅(%)
    open: float = 0.0            # 开盘点位
    high: float = 0.0            # 最高点位
    low: float = 0.0             # 最低点位
    prev_close: float = 0.0      # 昨收点位
    volume: float = 0.0          # 成交量（手）
    amount: float = 0.0          # 成交额（元）
    amplitude: float = 0.0       # 振幅(%)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'name': self.name,
            'current': self.current,
            'change': self.change,
            'change_pct': self.change_pct,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'volume': self.volume,
            'amount': self.amount,
            'amplitude': self.amplitude,
        }


@dataclass
class MarketOverview:
    """市场概览数据"""
    date: str                           # 数据交易日/报告交易日
    generated_at: str = ""              # 报告生成时间
    run_date: str = ""                  # 运行自然日
    data_date: str = ""                 # 实际行情数据日期
    data_scope_note: str = ""           # 日期/数据口径说明
    is_non_trading_run: bool = False    # 是否为非交易日/盘外复盘
    indices: List[MarketIndex] = field(default_factory=list)  # 主要指数
    up_count: int = 0                   # 上涨家数
    down_count: int = 0                 # 下跌家数
    flat_count: int = 0                 # 平盘家数
    limit_up_count: int = 0             # 涨停家数
    limit_down_count: int = 0           # 跌停家数
    total_amount: float = 0.0           # 两市成交额（亿元）
    # north_flow: float = 0.0           # 北向资金净流入（亿元）- 已废弃，接口不可用
    
    # 板块涨幅榜
    top_sectors: List[Dict] = field(default_factory=list)     # 涨幅前5板块
    bottom_sectors: List[Dict] = field(default_factory=list)  # 跌幅前5板块
    top_concepts: List[Dict] = field(default_factory=list)    # 涨幅前5概念
    bottom_concepts: List[Dict] = field(default_factory=list) # 跌幅前5概念

    # 外围市场联动参考（仅 A 股复盘注入；美股为最近收盘快照）
    global_indices: List[MarketIndex] = field(default_factory=list)

    # A 股大盘复盘增强输入
    fund_inflow_sectors: List[Dict] = field(default_factory=list)
    fund_outflow_sectors: List[Dict] = field(default_factory=list)
    limit_up_structure: Dict = field(default_factory=dict)
    index_key_levels: List[Dict] = field(default_factory=list)
    a_share_evidence: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.data_date:
            self.data_date = self.date


@dataclass
class MarketLightReviewResult:
    """Internal market-review parts built from one overview fetch."""

    overview: MarketOverview
    report: str
    market_light_snapshot: Optional[Dict[str, Any]]
    structured_payload: Dict[str, Any] = field(default_factory=dict)


def aggregate_limit_up_pool(rows: List[Dict]) -> Dict[str, Any]:
    """Aggregate limit-up pool rows for market-review prompt input."""
    return _aggregate_limit_up_pool(rows)


def compute_index_key_levels(bars: List[Dict]) -> Dict[str, Any]:
    """Compatibility wrapper for the shared MA5/MA10/MA20 calculation."""
    return _compute_index_key_levels(bars)


class MarketAnalyzer:
    """
    大盘复盘分析器
    
    功能：
    1. 获取大盘指数实时行情
    2. 获取市场涨跌统计
    3. 获取板块涨跌榜
    4. 搜索市场新闻
    5. 生成大盘复盘报告
    """
    
    def __init__(
        self,
        search_service: Optional[SearchService] = None,
        analyzer=None,
        region: str = "cn",
        config: Optional[Any] = None,
    ):
        """
        初始化大盘分析器

        Args:
            search_service: 搜索服务实例
            analyzer: AI分析器实例（用于调用LLM）
            region: 市场区域 cn=A股 hk=港股 us=美股 jp=日本 kr=韩国
            config: 本次复盘使用的配置；未传时读取全局配置
        """
        self.config = config or get_config()
        self.search_service = search_service
        self.analyzer = analyzer
        self.data_manager = DataFetcherManager()
        self.region = region if region in ("cn", "us", "hk", "jp", "kr") else "cn"
        self.profile: MarketProfile = get_profile(self.region)
        self.strategy = get_market_strategy_blueprint(self.region)
        # 最近一次新闻检索状态：not_run / no_search_service / ok / no_results / error
        self._news_search_status: str = "not_run"

    def _log_context(self) -> str:
        return f"component=market_review region={self.region}"

    def _get_output_language(self) -> str:
        """Return the truthful report language (zh/en/ko) for payload and directives."""
        return normalize_report_language(
            getattr(getattr(self, "config", None), "report_language", "zh")
        )

    def _get_review_language(self) -> str:
        # Structural/template language. Korean reuses the English scaffolding;
        # the Korean output directive is applied in the prompt builder.
        language = self._get_output_language()
        return "en" if language == "ko" else language

    def _get_template_review_language(self) -> str:
        return self._get_review_language()

    def _get_market_scope_name(self, review_language: str | None = None) -> str:
        review_language = review_language or self._get_review_language()
        if self.region == "us":
            return "US market" if review_language == "en" else "美股市场"
        if self.region == "hk":
            return "Hong Kong market" if review_language == "en" else "港股市场"
        if self.region == "jp":
            return "Japan market" if review_language == "en" else "日本市场"
        if self.region == "kr":
            return "Korea market" if review_language == "en" else "韩国市场"
        if review_language == "en":
            return "A-share market"
        return "A股市场"

    def _get_turnover_unit_label(self) -> str:
        """Return the turnover unit label for the current market/language."""
        if self.region == "us":
            return "USD bn" if self._get_review_language() == "en" else "十亿美元"
        if self.region == "hk":
            return "HKD bn" if self._get_review_language() == "en" else "十亿港元"
        if self.region == "jp":
            return "JPY bn" if self._get_review_language() == "en" else "十亿日元"
        if self.region == "kr":
            return "KRW bn" if self._get_review_language() == "en" else "十亿韩元"
        return "CNY 100m" if self._get_review_language() == "en" else "亿"

    def _format_turnover_value(self, amount_raw: float) -> str:
        """Format raw turnover according to market-specific units."""
        if amount_raw == 0.0:
            return "N/A"
        if self.region in ("us", "hk", "jp", "kr"):
            return f"{amount_raw / 1e9:.2f}"
        if amount_raw > 1e6:
            return f"{amount_raw / 1e8:.0f}"
        return f"{amount_raw:.0f}"

    def _get_index_change_arrow(self, change_pct: float) -> str:
        if change_pct == 0:
            return "⚪"
        color_scheme = getattr(getattr(self, "config", None), "market_review_color_scheme", "green_up")
        if color_scheme == "red_up":
            return "🔴" if change_pct > 0 else "🟢"
        return "🟢" if change_pct > 0 else "🔴"

    def _get_review_title(self, date: str) -> str:
        if self._get_review_language() == "en":
            market_names = {
                "us": "US Market Recap",
                "hk": "HK Market Recap",
                "jp": "Japan Market Recap",
                "kr": "Korea Market Recap",
            }
            market_name = market_names.get(self.region, "A-share Market Recap")
            return f"## {date} {market_name}"
        return f"## {date} 大盘复盘"

    def _get_index_hint(self) -> str:
        if self._get_review_language() == "en":
            if self.region == "us":
                return "Analyze the key moves in the S&P 500, Nasdaq, Dow, and other major indices."
            if self.region == "hk":
                return "Analyze the key moves in the HSI, Hang Seng Tech, HSCEI, and other major indices."
            if self.region == "jp":
                return "Analyze the key moves in the Nikkei 225, TOPIX, and other major Japanese indices."
            if self.region == "kr":
                return "Analyze the key moves in the KOSPI, KOSDAQ, and other major Korean indices."
            return "Analyze the price action in the SSE, SZSE, ChiNext, and other major indices."
        return self.profile.prompt_index_hint

    def _get_strategy_prompt_block(self) -> str:
        if self.region == "hk" and self._get_review_language() == "en":
            return """## Strategy Blueprint: Hong Kong Market Regime Strategy
Focus on HSI trend, southbound flow dynamics, and sector rotation to define next-session risk posture.

### Strategy Principles
- Read market regime from HSI, HSTECH, and HSCEI alignment first.
- Track southbound capital flow as a key sentiment driver.
- Translate recap into actionable risk-on/risk-off stance with clear invalidation points.

### Analysis Dimensions
- Trend Regime: Classify the market as momentum, range, or risk-off.
  - Are HSI/HSTECH/HSCEI directionally aligned
  - Did volume confirm the move
  - Are key index levels reclaimed or lost
- Capital Flows: Map southbound flow and macro narrative into equity risk appetite.
  - Southbound net flow direction and magnitude
  - USD/HKD and China policy implications
  - Breadth and leadership concentration
- Sector Themes: Identify persistent leaders and vulnerable laggards.
  - Tech/internet platform trend persistence
  - Financials/property sensitivity to policy shifts
  - Defensive vs growth factor rotation

### Action Framework
- Risk-on: broad index breakout with expanding southbound participation.
- Neutral: mixed index signals; focus on selective relative strength.
- Risk-off: failed breakouts and rising volatility; prioritize capital preservation."""
        if self.region == "jp" and self._get_review_language() == "en":
            return """## Strategy Blueprint: Japan Market Regime Strategy
Focus on Nikkei 225, TOPIX, currency dynamics, and global risk appetite to define the next-session trading plan.

### Strategy Principles
- Read Nikkei 225 and TOPIX alignment first, then assess yen moves, semiconductor/export chains, and financials.
- Translate index conclusions into position sizing, trading pace, and risk-control actions.
- Base judgments only on available index data, news, and price action without inventing breadth or sector statistics.

### Analysis Dimensions
- Trend Regime: Classify Japan equities as advancing, range-bound, or defensive.
  - Are Nikkei 225 and TOPIX directionally aligned
  - Have key index ranges been reclaimed or lost
  - Are large-cap weights and growth chains moving together
- Macro & FX: Map yen, rates, and global risk appetite into equity impact.
  - Yen direction and implications for exporters
  - Bank of Japan and US Treasury yield narratives
  - Overseas technology and semiconductor read-through
- Theme Signals: Identify durable leadership and crowded areas to avoid.
  - Semiconductor, automation, and auto-chain persistence
  - Rotation between financials and domestic-demand stocks
  - Whether news catalysts confirm price action

### Action Framework
- Risk-on: major indices rise together with improving external risk appetite and stronger leadership.
- Neutral: index divergence or FX disruption; avoid chasing and wait for confirmation.
- Risk-off: major indices weaken or external risk rises; prioritize position control."""
        if self.region == "kr" and self._get_review_language() == "en":
            return """## Strategy Blueprint: Korea Market Regime Strategy
Focus on KOSPI, KOSDAQ, semiconductor heavyweights, and global technology risk appetite to define the next-session trading plan.

### Strategy Principles
- Read KOSPI and KOSDAQ alignment first, then assess heavyweight signals from Samsung Electronics, SK Hynix, and related technology leaders.
- Separate broad index beta, semiconductor cycle exposure, and growth-stock risk appetite.
- Base judgments only on available index data, news, and price action without inventing breadth or sector statistics.

### Analysis Dimensions
- Trend Regime: Classify Korea equities as advancing, range-bound, or defensive.
  - Are KOSPI and KOSDAQ directionally aligned
  - Are heavyweight technology names supporting the indices
  - Have key support or resistance levels been reclaimed or lost
- Technology Cycle: Map semiconductor, AI hardware, and global technology moves into Korea equity risk.
  - Memory and semiconductor-chain catalysts
  - US technology-market read-through
  - Foreign investor risk appetite signals
- Theme Signals: Identify durable leadership and crowded areas to avoid.
  - Rotation across batteries, autos, and internet platforms
  - KOSDAQ growth-stock risk appetite
  - Whether news catalysts confirm price action

### Action Framework
- Risk-on: KOSPI and KOSDAQ rise together with confirmed technology leadership and improving external risk appetite.
- Neutral: index or heavyweight divergence; keep sizing controlled and wait for confirmation.
- Risk-off: technology heavyweights weaken or external risk rises; prioritize drawdown control."""
        if self.region == "us" and self._get_review_language() == "zh":
            return """## 美股市场三段式复盘策略
聚焦指数趋势、宏观叙事与板块轮动，给出次日风控与仓位框架。

### 策略原则
- 先看标普500、纳斯达克、道琼斯是否同向，确认主线是否一致。
- 结合宏观与流动性指标，识别风险偏好是修复还是转弱。
- 将复盘输出映射为“进攻/均衡/防守”动作建议，并给出明确触发失效条件。

### 分析维度
- 趋势结构：明确市场处于上冲、震荡还是防守转向，判断是否存在关键支撑位背离。
- 资金与情绪：区分宏观政策、货币面与波动率对权益风险的影响。
- 主题线索：识别持续性最强的主题与板块轮动是否形成可交易主线。

### 行动框架
- 进攻：主板块联动上行且量能/风险位同步改善。
- 均衡：指数分化或量能未明显放大，仓位保守执行。
- 防守：突破失守且波动率抬升时，优先减码并保留反弹可交易性。"""
        if not (self.region == "cn" and self._get_review_language() == "en"):
            return self.strategy.to_prompt_block()
        return """## Strategy Blueprint: A-share Three-Phase Recap Strategy
Focus on index trend, liquidity, and sector rotation to shape the next-session trading plan.

### Strategy Principles
- Read index direction first, then confirm liquidity structure, and finally test sector persistence.
- Every conclusion must map to position sizing, trading pace, and risk-control actions.
- Base judgments on today's data and the latest 3-day news flow without inventing unverified information.

### Analysis Dimensions
- Trend Structure: Determine whether the market is in an uptrend, range, or defensive phase.
  - Are the SSE, SZSE, and ChiNext moving in the same direction
  - Is the market advancing on expanding volume or slipping on contracting volume
  - Have key support or resistance levels been reclaimed or broken
- Liquidity & Sentiment: Identify near-term risk appetite and market temperature.
  - Advance/decline breadth and limit-up/limit-down structure
  - Whether turnover is expanding or fading
  - Whether high-beta leaders are showing divergence
- Leading Themes: Distill tradable leadership and areas to avoid.
  - Whether leading sectors have clear event catalysts
  - Whether sector leaders are pulling the group higher
  - Whether weakness is broadening across lagging sectors

### Action Framework
- Offensive: indices rise in sync, turnover expands, and core themes strengthen.
- Balanced: index divergence or low-volume consolidation; keep sizing controlled and wait for confirmation.
- Defensive: indices weaken and laggards broaden; prioritize risk control and de-risking."""

    def _get_strategy_markdown_block(self, review_language: str | None = None) -> str:
        review_language = review_language or self._get_review_language()
        if self.region == "hk" and review_language == "en":
            return """### 6. Strategy Framework
- **Trend Regime**: Classify the market as momentum, range, or risk-off based on HSI/HSTECH/HSCEI alignment.
- **Capital Flows**: Track southbound flow direction and macro narrative for risk appetite signals.
- **Sector Themes**: Focus on tech/internet platform persistence and financials/property policy sensitivity.
"""
        if self.region == "jp" and review_language == "en":
            return """### 6. Strategy Framework
- **Trend Regime**: Classify Japan equities as advancing, range-bound, or defensive based on Nikkei 225/TOPIX alignment.
- **Macro & FX**: Track yen, rates, and global risk appetite for exporter and financial-sector implications.
- **Theme Signals**: Focus on semiconductor, automation, auto-chain, financial, and domestic-demand rotation.
"""
        if self.region == "kr" and review_language == "en":
            return """### 6. Strategy Framework
- **Trend Regime**: Classify Korea equities as advancing, range-bound, or defensive based on KOSPI/KOSDAQ alignment.
- **Technology Cycle**: Track semiconductor, AI hardware, and global technology read-through for market risk appetite.
- **Theme Signals**: Focus on battery, auto, internet-platform, and KOSDAQ growth-stock rotation.
"""
        if self.region == "us" and review_language == "zh":
            return """### 六、策略框架
- **趋势结构**：判断市场在进攻、震荡与防守中的状态是否一致。
- **资金与情绪**：结合波动率、宽度和主题轮动评估风险偏好。
- **主题主线**：识别可延续和可放大的行业主线与防守线索。
"""
        if not (self.region == "cn" and review_language == "en"):
            return self.strategy.to_markdown_block()
        return """### 6. Strategy Framework
- **Trend Structure**: Determine whether the market is in an uptrend, range, or defensive phase.
- **Liquidity & Sentiment**: Track breadth, turnover expansion, and whether leaders are diverging.
- **Leading Themes**: Focus on sectors with catalysts and sustained leadership while avoiding broadening weakness.
"""

    def _get_market_mood_text(self, mood_key: str, review_language: str | None = None) -> str:
        review_language = review_language or self._get_review_language()
        if review_language == "en":
            mapping = {
                "strong_up": "strong gains",
                "mild_up": "moderate gains",
                "mild_down": "mild losses",
                "strong_down": "clear weakness",
                "range": "range-bound trading",
            }
        else:
            mapping = {
                "strong_up": "强势上涨",
                "mild_up": "小幅上涨",
                "mild_down": "小幅下跌",
                "strong_down": "明显下跌",
                "range": "震荡整理",
            }
        return mapping[mood_key]

    def get_market_overview(self) -> MarketOverview:
        """
        获取市场概览数据
        
        Returns:
            MarketOverview: 市场概览数据对象
        """
        now = datetime.now()
        run_date = now.strftime('%Y-%m-%d')
        try:
            data_date = get_effective_trading_date(self.region, current_time=now).isoformat()
        except Exception as exc:
            logger.warning(
                "[大盘] %s action=resolve_data_date status=failed error=%s",
                self._log_context(),
                exc,
            )
            data_date = run_date

        is_non_trading_run = data_date != run_date
        if is_non_trading_run:
            data_scope_note = (
                f"本次复盘生成于 {run_date}，该自然日不是可复用的完整交易日；"
                f"报告中的“今日”指 {data_date} 的最新完整交易日数据，“明日/次日”指该交易日之后的下一交易日。"
            )
        else:
            data_scope_note = (
                f"本次复盘生成于 {run_date}，报告数据口径按 {data_date} 交易日处理。"
            )

        overview = MarketOverview(
            date=data_date,
            generated_at=now.isoformat(timespec="seconds"),
            run_date=run_date,
            data_date=data_date,
            data_scope_note=data_scope_note,
            is_non_trading_run=is_non_trading_run,
        )
        
        # 1. 获取主要指数行情（按 region 切换 A 股/美股）
        overview.indices = self._get_main_indices()

        # 2. 获取涨跌统计（A 股有，美股无等效数据）
        if self.profile.has_market_stats:
            self._get_market_statistics(overview)

        # 3. 获取板块涨跌榜（A 股有，美股暂无）
        if self.profile.has_sector_rankings:
            self._get_sector_rankings(overview)
            self._get_concept_rankings(overview)
            self._get_sector_fund_flow_rankings(overview)
        
        # 4. 获取外围市场联动数据（仅 A 股复盘，fail-open）
        if self.profile.has_global_context:
            self._get_global_indices(overview)

        # 5. 构建 A 股复盘证据（日期化情绪池、指数均线、题材与观察票，fail-open）
        if self.region == "cn":
            self._get_a_share_review_evidence(overview)

        # 6. 获取北向资金（可选）
        # self._get_north_flow(overview)

        return overview

    
    def _get_main_indices(self) -> List[MarketIndex]:
        """获取主要指数实时行情"""
        indices = []

        try:
            logger.info("[大盘] %s action=get_main_indices status=start", self._log_context())

            # 使用 DataFetcherManager 获取指数行情（按 region 切换）
            data_list = self.data_manager.get_main_indices(region=self.region)

            if data_list:
                for item in data_list:
                    index = MarketIndex(
                        code=item['code'],
                        name=item['name'],
                        current=item['current'],
                        change=item['change'],
                        change_pct=item['change_pct'],
                        open=item['open'],
                        high=item['high'],
                        low=item['low'],
                        prev_close=item['prev_close'],
                        volume=item['volume'],
                        amount=item['amount'],
                        amplitude=item['amplitude']
                    )
                    indices.append(index)

            if not indices:
                logger.warning("[大盘] %s action=get_main_indices status=empty", self._log_context())
            else:
                logger.info(
                    "[大盘] %s action=get_main_indices status=success count=%d",
                    self._log_context(),
                    len(indices),
                )

        except Exception as e:
            logger.error("[大盘] %s action=get_main_indices status=failed error=%s", self._log_context(), e)

        return indices

    def _get_market_statistics(self, overview: MarketOverview):
        """获取市场涨跌统计"""
        try:
            logger.info("[大盘] %s action=get_market_stats status=start", self._log_context())

            stats = self.data_manager.get_market_stats(purpose=f"market_review:{self.region}")

            if stats:
                overview.up_count = stats.get('up_count', 0)
                overview.down_count = stats.get('down_count', 0)
                overview.flat_count = stats.get('flat_count', 0)
                overview.limit_up_count = stats.get('limit_up_count', 0)
                overview.limit_down_count = stats.get('limit_down_count', 0)
                overview.total_amount = stats.get('total_amount', 0.0)

                logger.info(
                    "[大盘] %s action=get_market_stats status=success up=%s down=%s flat=%s "
                    "limit_up=%s limit_down=%s amount=%.0f亿",
                    self._log_context(),
                    overview.up_count,
                    overview.down_count,
                    overview.flat_count,
                    overview.limit_up_count,
                    overview.limit_down_count,
                    overview.total_amount,
                )
            else:
                logger.warning("[大盘] %s action=get_market_stats status=empty", self._log_context())

        except Exception as e:
            logger.error("[大盘] %s action=get_market_stats status=failed error=%s", self._log_context(), e)

    def _get_sector_rankings(self, overview: MarketOverview):
        """获取板块涨跌榜"""
        try:
            logger.info("[大盘] %s action=get_sector_rankings status=start", self._log_context())

            top_sectors, bottom_sectors = self.data_manager.get_sector_rankings(5)

            if top_sectors or bottom_sectors:
                overview.top_sectors = top_sectors
                overview.bottom_sectors = bottom_sectors

                logger.info(
                    "[大盘] %s action=get_sector_rankings status=success top=%s bottom=%s",
                    self._log_context(),
                    [s['name'] for s in overview.top_sectors],
                    [s['name'] for s in overview.bottom_sectors],
                )
            else:
                logger.warning("[大盘] %s action=get_sector_rankings status=empty", self._log_context())

        except Exception as e:
            logger.error("[大盘] %s action=get_sector_rankings status=failed error=%s", self._log_context(), e)

    def _get_concept_rankings(self, overview: MarketOverview):
        """获取概念/题材涨跌榜（fail-open）。"""
        try:
            logger.info("[大盘] %s action=get_concept_rankings status=start", self._log_context())

            top_concepts, bottom_concepts = self.data_manager.get_concept_rankings(5)

            if top_concepts or bottom_concepts:
                overview.top_concepts = top_concepts
                overview.bottom_concepts = bottom_concepts

                logger.info(
                    "[大盘] %s action=get_concept_rankings status=success top=%s bottom=%s",
                    self._log_context(),
                    [s.get('name') for s in overview.top_concepts],
                    [s.get('name') for s in overview.bottom_concepts],
                )
            else:
                logger.warning("[大盘] %s action=get_concept_rankings status=empty", self._log_context())

        except Exception as e:
            logger.warning("[大盘] %s action=get_concept_rankings status=failed error=%s", self._log_context(), e)

    def _get_sector_fund_flow_rankings(self, overview: MarketOverview):
        """获取板块资金流排行（fail-open）。"""
        try:
            logger.info("[大盘] %s action=get_sector_fund_flow_rankings status=start", self._log_context())

            top_sectors, bottom_sectors = self.data_manager.get_sector_fund_flow_rankings(5)
            if top_sectors or bottom_sectors:
                overview.fund_inflow_sectors = top_sectors
                overview.fund_outflow_sectors = bottom_sectors

                logger.info(
                    "[大盘] %s action=get_sector_fund_flow_rankings status=success top=%s bottom=%s",
                    self._log_context(),
                    [s.get('name') for s in overview.fund_inflow_sectors],
                    [s.get('name') for s in overview.fund_outflow_sectors],
                )
            else:
                logger.warning("[大盘] %s action=get_sector_fund_flow_rankings status=empty", self._log_context())

        except Exception as e:
            logger.warning("[大盘] %s action=get_sector_fund_flow_rankings status=failed error=%s", self._log_context(), e)

    def _get_a_share_review_evidence(self, overview: MarketOverview) -> None:
        """Build one structured A-share evidence snapshot and legacy compatibility fields."""
        try:
            logger.info("[大盘] %s action=get_a_share_review_evidence status=start", self._log_context())
            service = AShareReviewEvidenceService(self.data_manager)
            evidence = service.build(
                trade_date=(overview.data_date or overview.date).replace("-", ""),
                indices=overview.indices,
                sector_rankings={"top": overview.top_sectors, "bottom": overview.bottom_sectors},
                concept_rankings={"top": overview.top_concepts, "bottom": overview.bottom_concepts},
                market_snapshot={
                    "up_count": overview.up_count,
                    "down_count": overview.down_count,
                    "flat_count": overview.flat_count,
                    "total_amount": overview.total_amount,
                },
            )
            overview.a_share_evidence = evidence

            sentiment = evidence.get("sentiment_structure") if isinstance(evidence, dict) else None
            quality = evidence.get("data_quality") if isinstance(evidence, dict) else None
            missing_fields = quality.get("missing_fields") if isinstance(quality, dict) else []
            if isinstance(sentiment, dict) and "limit_up_pool" not in (missing_fields or []):
                overview.limit_up_structure = {
                    "total": int(sentiment.get("limit_up_count") or 0),
                    "industry_distribution": list(sentiment.get("industry_distribution") or []),
                    "max_consecutive_boards": int(sentiment.get("highest_consecutive_board") or 0),
                    "max_boards_stock": str(sentiment.get("highest_board_stock") or ""),
                    "total_break_count": int(sentiment.get("total_break_count") or 0),
                }
            else:
                overview.limit_up_structure = {}

            index_trend = evidence.get("index_trend") if isinstance(evidence, dict) else None
            if isinstance(index_trend, list):
                overview.index_key_levels = [dict(item) for item in index_trend if isinstance(item, dict)]

            logger.info(
                "[大盘] %s action=get_a_share_review_evidence status=success evidence_status=%s themes=%d stocks=%d",
                self._log_context(),
                evidence.get("status") if isinstance(evidence, dict) else "unknown",
                len(evidence.get("theme_candidates") or []) if isinstance(evidence, dict) else 0,
                len(evidence.get("stock_candidates") or []) if isinstance(evidence, dict) else 0,
            )
        except Exception as exc:
            logger.warning(
                "[大盘] %s action=get_a_share_review_evidence status=failed error=%s",
                self._log_context(),
                exc,
                exc_info=True,
            )
            overview.a_share_evidence = {}

    def _get_limit_up_structure(self, overview: MarketOverview):
        """获取涨停结构（fail-open）。"""
        try:
            logger.info("[大盘] %s action=get_limit_up_structure status=start", self._log_context())

            rows = self.data_manager.get_limit_up_pool(n=200)
            overview.limit_up_structure = aggregate_limit_up_pool(rows or [])

            if overview.limit_up_structure:
                logger.info(
                    "[大盘] %s action=get_limit_up_structure status=success total=%s",
                    self._log_context(),
                    overview.limit_up_structure.get("total"),
                )
            else:
                logger.warning("[大盘] %s action=get_limit_up_structure status=empty", self._log_context())

        except Exception as e:
            logger.warning("[大盘] %s action=get_limit_up_structure status=failed error=%s", self._log_context(), e)
    
    def _get_global_indices(self, overview: MarketOverview):
        """获取外围市场联动参考指数（fail-open，失败不影响复盘主流程）。"""
        try:
            logger.info("[大盘] %s action=get_global_indices status=start", self._log_context())

            data_list = self.data_manager.get_global_market_indices()
            for item in data_list or []:
                overview.global_indices.append(
                    MarketIndex(
                        code=str(item.get('code', '')),
                        name=str(item.get('name', '')),
                        current=float(item.get('current', 0) or 0),
                        change=float(item.get('change', 0) or 0),
                        change_pct=float(item.get('change_pct', 0) or 0),
                        open=float(item.get('open', 0) or 0),
                        high=float(item.get('high', 0) or 0),
                        low=float(item.get('low', 0) or 0),
                        prev_close=float(item.get('prev_close', 0) or 0),
                        volume=float(item.get('volume', 0) or 0),
                        amount=float(item.get('amount', 0) or 0),
                        amplitude=float(item.get('amplitude', 0) or 0),
                    )
                )

            if overview.global_indices:
                logger.info(
                    "[大盘] %s action=get_global_indices status=success count=%d",
                    self._log_context(),
                    len(overview.global_indices),
                )
            else:
                logger.warning("[大盘] %s action=get_global_indices status=empty", self._log_context())
        except Exception as e:
            logger.warning("[大盘] %s action=get_global_indices status=failed error=%s", self._log_context(), e)

    def _get_index_key_levels(self, overview: MarketOverview):
        """获取指数关键位参考（fail-open，失败不影响复盘主流程）。"""
        try:
            logger.info("[大盘] %s action=get_index_key_levels status=start", self._log_context())

            index_by_target = self._match_index_key_level_targets(overview.indices)
            for target in CN_INDEX_KEY_LEVEL_TARGETS:
                index = index_by_target.get(target["symbol"])
                if index is None:
                    continue
                try:
                    bars = self.data_manager.get_index_daily_history(target["symbol"], days=30)
                    levels = compute_index_key_levels(bars or [])
                except Exception as e:
                    logger.warning(
                        "[大盘] %s action=get_index_key_levels status=index_failed symbol=%s error=%s",
                        self._log_context(),
                        target["symbol"],
                        e,
                    )
                    continue
                if not levels:
                    continue
                overview.index_key_levels.append({
                    "name": target["name"],
                    "current": index.current,
                    "ma20": levels["ma20"],
                    "high_20d": levels["high_20d"],
                    "low_20d": levels["low_20d"],
                })

            if overview.index_key_levels:
                logger.info(
                    "[大盘] %s action=get_index_key_levels status=success count=%d",
                    self._log_context(),
                    len(overview.index_key_levels),
                )
            else:
                logger.warning("[大盘] %s action=get_index_key_levels status=empty", self._log_context())
        except Exception as e:
            logger.warning("[大盘] %s action=get_index_key_levels status=failed error=%s", self._log_context(), e)

    @staticmethod
    def _match_index_key_level_targets(indices: List[MarketIndex]) -> Dict[str, MarketIndex]:
        result: Dict[str, MarketIndex] = {}
        for target in CN_INDEX_KEY_LEVEL_TARGETS:
            aliases = tuple(str(alias).lower() for alias in target["aliases"])
            for idx in indices or []:
                code = str(getattr(idx, "code", "") or "").lower()
                name = str(getattr(idx, "name", "") or "").lower()
                if any(alias and (code.endswith(alias) or name == alias) for alias in aliases):
                    result[target["symbol"]] = idx
                    break
        return result

    # def _get_north_flow(self, overview: MarketOverview):
    #     """获取北向资金流入"""
    #     try:
    #         logger.info("[大盘] 获取北向资金...")
    #         
    #         # 获取北向资金数据
    #         df = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")
    #         
    #         if df is not None and not df.empty:
    #             # 取最新一条数据
    #             latest = df.iloc[-1]
    #             if '当日净流入' in df.columns:
    #                 overview.north_flow = float(latest['当日净流入']) / 1e8  # 转为亿元
    #             elif '净流入' in df.columns:
    #                 overview.north_flow = float(latest['净流入']) / 1e8
    #                 
    #             logger.info(f"[大盘] 北向资金净流入: {overview.north_flow:.2f}亿")
    #             
    #     except Exception as e:
    #         logger.warning(f"[大盘] 获取北向资金失败: {e}")
    
    def _build_dynamic_news_queries(self, overview: Optional[MarketOverview]) -> List[str]:
        """按当日实际领涨领跌板块生成事件导向检索词（仅 A 股，fail-open）。"""
        if overview is None or self.region != "cn":
            return []

        queries: List[str] = []
        seen_names = set()

        def _collect(rows: List[Dict], template: str, limit: int) -> None:
            count = 0
            for row in rows or []:
                if count >= limit:
                    break
                name = str((row or {}).get("name", "")).strip()
                if not name or name in seen_names:
                    continue
                seen_names.add(name)
                queries.append(template.format(name=name))
                count += 1

        _collect(overview.top_sectors, "{name} 板块 大涨 原因", 1)
        _collect(overview.bottom_sectors, "{name} 板块 大跌 原因", 1)
        _collect(overview.top_concepts, "{name} 概念 上涨 消息", 1)
        return queries[:3]

    def search_market_news(self, overview: Optional[MarketOverview] = None) -> List[Dict]:
        """
        搜索市场新闻

        Args:
            overview: 当日市场概览；提供时按领涨领跌板块追加事件导向检索词

        Returns:
            新闻列表（检索状态记录在 self._news_search_status，供 prompt 占位符区分
            “未配置搜索服务”与“已检索无结果”）
        """
        if not self.search_service:
            self._news_search_status = "no_search_service"
            logger.warning(
                "[大盘] %s action=search_market_news status=skipped reason=no_search_service",
                self._log_context(),
            )
            return []

        all_news = []
        seen_urls = set()

        # 按 region 使用不同的新闻搜索词；A 股追加当日板块事件检索词
        search_queries = list(self.profile.news_queries) + self._build_dynamic_news_queries(overview)
        review_language = self._get_review_language()
        market_names = {
            "cn": "大盘" if review_language == "zh" else "A-share market",
            "us": "美股市场" if review_language == "zh" else "US market",
            "hk": "港股市场" if review_language == "zh" else "HK market",
            "jp": "日本股市" if review_language == "zh" else "Japan stock market",
            "kr": "韩国股市" if review_language == "zh" else "Korea stock market",
        }
        
        try:
            logger.info("[大盘] %s action=search_market_news status=start", self._log_context())
            
            # 根据 region 设置搜索上下文名称，避免美股搜索被解读为 A 股语境
            market_name = market_names.get(self.region, "大盘")
            for query in search_queries:
                response = self.search_service.search_stock_news(
                    stock_code="market",
                    stock_name=market_name,
                    max_results=3,
                    focus_keywords=query.split()
                )
                if response and response.results:
                    added = 0
                    for item in response.results:
                        url = self._get_news_field(item, "url")
                        if url and url in seen_urls:
                            continue
                        if url:
                            seen_urls.add(url)
                        all_news.append(item)
                        added += 1
                    logger.info(
                        "[大盘] %s action=search_market_news status=query_success count=%d",
                        self._log_context(),
                        added,
                    )

            self._news_search_status = "ok" if all_news else "no_results"
            logger.info(
                "[大盘] %s action=search_market_news status=success count=%d",
                self._log_context(),
                len(all_news),
            )

        except Exception as e:
            self._news_search_status = "error" if not all_news else "ok"
            logger.error("[大盘] %s action=search_market_news status=failed error=%s", self._log_context(), e)

        return all_news
    
    def generate_market_review(self, overview: MarketOverview, news: List) -> str:
        """
        使用大模型生成大盘复盘报告
        
        Args:
            overview: 市场概览数据
            news: 市场新闻列表 (SearchResult 对象列表)
            
        Returns:
            大盘复盘报告文本
        """
        backend_error = self._get_analyzer_generation_backend_config_error()
        if backend_error is not None:
            logger.error(
                "[大盘] %s action=generate_review status=failed error_type=%s error=%s",
                self._log_context(),
                type(backend_error).__name__,
                backend_error,
            )
            record_llm_run(
                success=False,
                provider="litellm",
                model=getattr(self.config, "litellm_model", None),
                call_type="market_review",
                error_type=type(backend_error).__name__,
                error_message=backend_error,
            )
            raise backend_error

        if not self.analyzer or not self.analyzer.is_available():
            logger.warning(
                "[大盘] %s action=generate_review status=fallback_template reason=no_analyzer",
                self._log_context(),
            )
            return self._generate_template_review(overview, news)

        # 构建 Prompt
        prompt = self._build_review_prompt(overview, news)

        logger.info("[大盘] %s action=generate_review status=start", self._log_context())
        # Use the public generate_text() entry point - never access private analyzer attributes.
        llm_started_at = time.perf_counter()
        try:
            record_llm_run_started(
                provider="litellm",
                model=getattr(self.config, "litellm_model", None),
                call_type="market_review",
            )
            review = self.analyzer.generate_text(prompt, max_tokens=8192, temperature=0.7)
        except Exception as exc:
            record_llm_run(
                success=False,
                provider="litellm",
                model=getattr(self.config, "litellm_model", None),
                call_type="market_review",
                duration_ms=int((time.perf_counter() - llm_started_at) * 1000),
                error_type=type(exc).__name__,
                error_message=exc,
            )
            raise

        record_llm_run(
            success=bool(review),
            provider="litellm",
            model=getattr(self.config, "litellm_model", None),
            call_type="market_review",
            duration_ms=int((time.perf_counter() - llm_started_at) * 1000),
            error_type=None if review else "EmptyResponse",
            error_message=None if review else "empty market review response",
        )

        if review:
            logger.info(
                "[大盘] %s action=generate_review status=success length=%d",
                self._log_context(),
                len(review),
            )
            # Inject structured data tables into LLM prose sections
            return self._inject_data_into_review(review, overview, news)

        logger.warning(
            "[大盘] %s action=generate_review status=fallback_template reason=empty_llm_response",
            self._log_context(),
        )
        return self._generate_template_review(overview, news)

    def _get_analyzer_generation_backend_config_error(self) -> Optional[GenerationError]:
        """Return analyzer backend config errors without relying on dynamic mock attributes."""
        if self.analyzer is None:
            try:
                resolve_generation_backend_id(self.config)
                resolve_generation_fallback_backend_id(self.config)
            except GenerationError as exc:
                return exc
            return None
        missing = object()
        if getattr_static(self.analyzer, "get_generation_backend_config_error", missing) is missing:
            return None
        method = getattr(self.analyzer, "get_generation_backend_config_error", None)
        if not callable(method):
            return None
        error = method()
        return error if isinstance(error, GenerationError) else None

    def build_market_review_payload(
        self,
        overview: MarketOverview,
        news: List,
        report: str,
        market_light_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build the structured market-review contract consumed by API, Web, and notifications."""
        language = self._get_output_language()
        sections = self._split_report_sections(report)
        title = self._extract_report_title(report) or self._get_review_title(overview.date).lstrip("# ").strip()
        light = (
            market_light_snapshot or self.build_market_light_snapshot(overview)
            if self._supports_market_light()
            else None
        )
        breadth_dimensions = None
        if isinstance(light, dict):
            dimensions = light.get("dimensions")
            if isinstance(dimensions, dict):
                breadth_dimensions = dimensions.get("breadth")

        breadth_supported = bool(self.profile.has_market_stats)
        if breadth_supported and isinstance(breadth_dimensions, dict) and "available" in breadth_dimensions:
            breadth_supported = bool(breadth_dimensions.get("available"))

        has_breadth_data = False
        if breadth_supported:
            if isinstance(breadth_dimensions, dict) and "available" in breadth_dimensions:
                has_breadth_data = bool(breadth_dimensions.get("available"))
            else:
                breadth_available = overview.up_count + overview.down_count + overview.flat_count > 0
                limit_available = overview.limit_up_count + overview.limit_down_count > 0
                has_breadth_data = bool(breadth_available or limit_available)

        payload = {
            "version": 1,
            "kind": "market_review",
            "region": self.region,
            "language": language,
            "title": title,
            "generated_at": overview.generated_at or datetime.now().isoformat(),
            "date": overview.date,
            "run_date": overview.run_date or overview.date,
            "data_date": overview.data_date or overview.date,
            "is_non_trading_run": overview.is_non_trading_run,
            "data_scope_note": overview.data_scope_note,
            "market_scope": self._get_market_scope_name(language),
            "indices": [idx.to_dict() for idx in overview.indices],
            "sectors": {
                "top": list(overview.top_sectors or []),
                "bottom": list(overview.bottom_sectors or []),
            },
            "concepts": {
                "top": list(overview.top_concepts or []),
                "bottom": list(overview.bottom_concepts or []),
            },
            "news": [self._normalize_news_item(item) for item in (news or [])[:8]],
            "sections": sections,
            "markdown_report": report,
        }

        if light is not None:
            payload["market_light"] = light

        if has_breadth_data:
            payload["breadth"] = {
                "up_count": overview.up_count,
                "down_count": overview.down_count,
                "flat_count": overview.flat_count,
                "limit_up_count": overview.limit_up_count,
                "limit_down_count": overview.limit_down_count,
                "total_amount": overview.total_amount,
                "turnover_unit": self._get_turnover_unit_label(),
            }

        if self.region == "cn":
            payload["fund_flows"] = {
                "inflow": list(overview.fund_inflow_sectors or []),
                "outflow": list(overview.fund_outflow_sectors or []),
            }
            payload["limit_up_structure"] = dict(overview.limit_up_structure or {})
            payload["index_key_levels"] = list(overview.index_key_levels or [])
            payload["a_share_evidence"] = dict(overview.a_share_evidence or {})

        return payload

    def _supports_market_light(self) -> bool:
        return self.region in MARKET_LIGHT_REGIONS

    @staticmethod
    def _extract_report_title(report: str) -> str:
        for line in (report or "").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()
        return ""

    @classmethod
    def _split_report_sections(cls, report: str) -> List[Dict[str, str]]:
        text = (report or "").strip()
        if not text:
            return []
        matches = list(re.finditer(r"^(#{2,3})\s+(.+?)\s*$", text, flags=re.MULTILINE))
        if not matches:
            return [{"key": "full_review", "title": "Review", "markdown": text}]

        sections: List[Dict[str, str]] = []
        first_match = matches[0]
        starts_with_report_title = first_match.start() == 0 and first_match.group(1) == "##"
        content_start_index = 1 if starts_with_report_title else 0
        intro_start = first_match.end() if starts_with_report_title else 0
        intro_end = (
            matches[1].start()
            if starts_with_report_title and len(matches) > 1
            else (len(text) if starts_with_report_title else matches[0].start())
        )
        intro = text[intro_start:intro_end].strip()
        if intro:
            sections.append({"key": "overview", "title": "Overview", "markdown": intro})

        for index, match in enumerate(matches[content_start_index:], start=content_start_index):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            title = match.group(2).strip()
            markdown = text[start:end].strip()
            if not markdown:
                continue
            key = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "_", title).strip("_").lower()
            sections.append({
                "key": key or f"section_{index + 1}",
                "title": title,
                "markdown": markdown,
            })
        return sections

    @classmethod
    def _normalize_news_item(cls, item: Any) -> Dict[str, str]:
        return {
            "title": cls._compact_news_text(cls._get_news_field(item, "title"), limit=120),
            "snippet": cls._compact_news_text(cls._get_news_field(item, "snippet"), limit=260),
            "source": cls._compact_news_text(cls._get_news_field(item, "source"), limit=80),
            "published_date": cls._compact_news_text(cls._get_news_field(item, "published_date"), limit=40),
            "url": cls._compact_news_text(cls._get_news_field(item, "url"), limit=240),
        }
    
    def _inject_data_into_review(
        self,
        review: str,
        overview: MarketOverview,
        news: Optional[List] = None,
    ) -> str:
        """Inject structured data tables into the corresponding LLM prose sections."""
        # Build data blocks
        data_scope_block = self._build_data_scope_report_block(overview)
        stats_block = self._build_stats_block(overview)
        indices_block = self._build_indices_block(overview)
        sector_block = self._build_sector_block(overview)
        patterns = (
            _ENGLISH_SECTION_PATTERNS
            if self._get_review_language() == "en"
            else _CHINESE_SECTION_PATTERNS
        )

        data_scope_pattern = (
            r"###\s*(?:1\.\s*)?Data Scope"
            if self._get_review_language() == "en"
            else r"###\s*(?:[一二三四五六七八九十]+、)?数据口径"
        )
        if data_scope_block and not re.search(data_scope_pattern, review):
            review = self._insert_after_report_title(review, data_scope_block)

        if stats_block:
            review = self._insert_after_section(
                review,
                patterns["market_summary"],
                stats_block,
            )

        if indices_block:
            review = self._insert_after_section(
                review,
                patterns["index_commentary"],
                indices_block,
            )

        if sector_block:
            original_review = review
            review = self._insert_after_section(
                review,
                patterns["sector_highlights"],
                sector_block,
            )
            if review == original_review and sector_block not in review:
                fallback_heading = (
                    "### 4. Sector Highlights"
                    if self._get_review_language() == "en"
                    else "### 三、板块主线"
                )
                review = f"{review.rstrip()}\n\n{fallback_heading}\n{sector_block}\n"

        return review

    @staticmethod
    def _insert_after_report_title(text: str, block: str) -> str:
        """Insert a block immediately after the first markdown report title."""
        match = re.search(r"^##\s+.+?\s*$", text or "", flags=re.MULTILINE)
        if not match:
            return f"{block}\n\n{(text or '').lstrip()}".strip()
        insert_pos = match.end()
        return text[:insert_pos].rstrip() + "\n\n" + block + "\n\n" + text[insert_pos:].lstrip("\n")

    @staticmethod
    def _insert_after_section(text: str, heading_pattern: str, block: str) -> str:
        """Insert a data block at the end of a markdown section (before the next ### heading)."""
        import re
        # Find the heading
        match = re.search(heading_pattern, text)
        if not match:
            return text
        start = match.end()
        # Find the next ### heading after this one
        next_heading = re.search(r'\n###\s', text[start:])
        if next_heading:
            insert_pos = start + next_heading.start()
        else:
            # No next heading — append at end
            insert_pos = len(text)
        # Insert the block before the next heading, with spacing
        return text[:insert_pos].rstrip() + '\n\n' + block + '\n\n' + text[insert_pos:].lstrip('\n')

    def _build_stats_block(self, overview: MarketOverview) -> str:
        """Build market statistics block."""
        has_stats = overview.up_count or overview.down_count or overview.total_amount
        if not has_stats:
            return ""
        has_limit_structure = bool(getattr(overview, "limit_up_structure", None))
        if self._get_review_language() == "en":
            light = self.build_market_light_snapshot(overview)
            if has_limit_structure:
                continuation_note = (
                    "- **Sentiment data boundary**: limit-up pool, continuation height, break count, and industry distribution are provided separately when available; previous-limit-up premium, broken-board feedback, and one-tick board ratio remain unavailable."
                )
            else:
                continuation_note = (
                    "- **Data gap**: board-failure rate, continuation height, previous-limit-up premium, broken-board feedback, and one-tick board ratio are unavailable, so this measures heat rather than follow-through quality."
                )
            return "\n".join(
                [
                    f"- **Market Signal**: {light['score']}/100 "
                    f"({light['temperature_label']}, {light['label']})",
                    f"- **Drivers**: {'; '.join(light['reasons'])}",
                    f"- **Guidance**: {light['guidance']}",
                    "",
                    f"- **Breadth**: Advancers {overview.up_count} / Decliners {overview.down_count} / "
                    f"Flat {overview.flat_count}; "
                    f"Limit-up {overview.limit_up_count} / Limit-down {overview.limit_down_count}; "
                    f"Turnover {overview.total_amount:.0f} ({self._get_turnover_unit_label()})",
                    "",
                    continuation_note,
                ]
            )
        light = self.build_market_light_snapshot(overview)
        score, label = light["score"], light["temperature_label"]
        participation = overview.up_count + overview.down_count
        up_ratio = overview.up_count / participation if participation else 0.0
        limit_spread = overview.limit_up_count - overview.limit_down_count
        if has_limit_structure:
            continuation_note = (
                "- **情绪数据边界**：涨停池、最高连板、炸板次数和行业分布会在涨停结构中单独列示；当前仍未提供昨日涨停溢价、断板反馈和一字板比例。"
            )
        else:
            continuation_note = (
                "- **情绪数据边界**：当前未提供炸板率、连板高度、昨日涨停溢价、断板反馈和一字板比例；这里只能判断市场热度，不能判断接力质量。"
            )
        lines = [
            f"- **盘面信号**：{score}/100（{label}，{light['label']}）",
            f"- **信号依据**：{'；'.join(light['reasons'])}",
            f"- **操作建议**：{light['guidance']}",
            "",
            "| 指标 | 数值 | 观察 |",
            "|------|------|------|",
            f"| 上涨/下跌/平盘 | {overview.up_count} / {overview.down_count} / {overview.flat_count} | 上涨占比(不含平盘) {up_ratio:.1%} |",
            f"| 涨停/跌停 | {overview.limit_up_count} / {overview.limit_down_count} | 涨跌停差 {limit_spread:+d} |",
            f"| 两市成交额 | {overview.total_amount:.0f} 亿 | {self._describe_turnover(overview.total_amount)} |",
            "",
            continuation_note,
        ]
        return "\n".join(lines)

    def build_market_light_snapshot(self, overview: MarketOverview) -> Dict[str, Any]:
        """Build a deterministic market-light snapshot from structured breadth data."""
        scores = self._build_market_light_scores(overview)
        score = int(scores["score"])
        temperature_label = str(scores["temperature_label"])
        if score >= 60:
            status = "green"
        elif score >= 40:
            status = "yellow"
        else:
            status = "red"

        if self._get_review_language() == "en":
            label_map = {
                "green": "risk-on",
                "yellow": "balanced",
                "red": "risk-off",
            }
            guidance_map = {
                "green": "Risk appetite is acceptable, but keep exposure incremental when leadership evidence is incomplete.",
                "yellow": "Signals are mixed; keep position sizing moderate and wait for confirmation.",
                "red": "Risk is elevated; prioritize drawdown control and avoid chasing weak rebounds.",
            }
            reasons = self._build_market_light_reasons_en(overview, score)
        else:
            label_map = {
                "green": "可进攻",
                "yellow": "需观察",
                "red": "偏防守",
            }
            guidance_map = {
                "green": "风险偏好尚可，但主线证据不足时先试错，确认后再加仓。",
                "yellow": "信号分化，控制仓位并等待量价确认。",
                "red": "风险偏高，优先控制回撤，避免追高弱反弹。",
            }
            reasons = self._build_market_light_reasons_zh(overview, score)

        snapshot = MarketLightSnapshot(
            region=self.region,
            trade_date=overview.date,
            status=status,
            label=label_map[status],
            score=score,
            temperature_label=temperature_label,
            reasons=reasons,
            guidance=guidance_map[status],
            dimensions=scores["dimensions"],
            data_quality=str(scores["data_quality"]),
        )
        return snapshot.model_dump()

    def _build_market_light_reasons_zh(self, overview: MarketOverview, score: int) -> List[str]:
        participation = overview.up_count + overview.down_count
        up_ratio = overview.up_count / participation if participation else None
        reasons: List[str] = []
        if up_ratio is not None:
            if up_ratio >= 0.6:
                reasons.append(f"上涨家数占比 {up_ratio:.0%}，赚钱效应扩散")
            elif up_ratio <= 0.4:
                reasons.append(f"上涨家数占比 {up_ratio:.0%}，亏钱效应较强")
            else:
                reasons.append(f"上涨家数占比 {up_ratio:.0%}，市场分化")
        index_changes = [idx.change_pct for idx in overview.indices if idx.change_pct is not None]
        if index_changes:
            avg_change = sum(index_changes) / len(index_changes)
            reasons.append(f"主要指数平均涨跌幅 {avg_change:+.2f}%")
        if overview.limit_up_count or overview.limit_down_count:
            reasons.append(f"涨跌停差 {overview.limit_up_count - overview.limit_down_count:+d}")
        if not reasons and overview.total_amount:
            reasons.append(f"成交额 {overview.total_amount:.0f} 亿，{self._describe_turnover(overview.total_amount)}")
        if not reasons:
            reasons.append("结构化涨跌数据有限，按可用行情综合判断")
        return reasons[:4]

    def _build_market_light_reasons_en(self, overview: MarketOverview, score: int) -> List[str]:
        participation = overview.up_count + overview.down_count
        up_ratio = overview.up_count / participation if participation else None
        reasons: List[str] = []
        if up_ratio is not None:
            if up_ratio >= 0.6:
                reasons.append(f"advancers ratio {up_ratio:.0%}, breadth is expanding")
            elif up_ratio <= 0.4:
                reasons.append(f"advancers ratio {up_ratio:.0%}, downside pressure dominates")
            else:
                reasons.append(f"advancers ratio {up_ratio:.0%}, breadth is mixed")
        index_changes = [idx.change_pct for idx in overview.indices if idx.change_pct is not None]
        if index_changes:
            avg_change = sum(index_changes) / len(index_changes)
            reasons.append(f"average major-index change {avg_change:+.2f}%")
        if overview.limit_up_count or overview.limit_down_count:
            reasons.append(f"limit-up/down spread {overview.limit_up_count - overview.limit_down_count:+d}")
        if not reasons and overview.total_amount:
            reasons.append(f"turnover {overview.total_amount:.0f} ({self._get_turnover_unit_label()})")
        if not reasons:
            reasons.append("limited structured breadth data; using available market inputs")
        return reasons[:4]

    def _build_indices_block(self, overview: MarketOverview) -> str:
        """构建指数行情表格"""
        if not overview.indices:
            return ""
        if self._get_review_language() == "en":
            lines = [
                f"| Index | Last | Change % | Open | High | Low | Amplitude | Turnover ({self._get_turnover_unit_label()}) |",
                "|-------|------|----------|------|------|-----|-----------|-----------------|",
            ]
        else:
            lines = [
                "| 指数 | 最新 | 涨跌幅 | 开盘 | 最高 | 最低 | 振幅 | 成交额(亿) |",
                "|------|------|--------|------|------|------|------|-----------|",
            ]
        for idx in overview.indices:
            arrow = self._get_index_change_arrow(idx.change_pct)
            amount_raw = idx.amount or 0.0
            amount_str = self._format_turnover_value(amount_raw)
            lines.append(
                f"| {idx.name} | {idx.current:.2f} | {arrow} {idx.change_pct:+.2f}% | "
                f"{self._format_optional_number(idx.open)} | {self._format_optional_number(idx.high)} | "
                f"{self._format_optional_number(idx.low)} | {self._format_optional_pct(idx.amplitude)} | {amount_str} |"
            )
        return "\n".join(lines)

    def _build_sector_block(self, overview: MarketOverview) -> str:
        """Build industry and concept ranking blocks."""
        if (
            not overview.top_sectors
            and not overview.bottom_sectors
            and not overview.top_concepts
            and not overview.bottom_concepts
        ):
            return ""
        lines = []
        language = self._get_review_language()
        has_fund_flow = bool(
            getattr(overview, "fund_inflow_sectors", None)
            or getattr(overview, "fund_outflow_sectors", None)
        )
        if language == "en":
            if has_fund_flow:
                lines.append(
                    "- **Theme data boundary**: rankings can be cross-checked with sector fund-flow leaders when provided; capacity leaders, front-row stocks, and diffusion chains remain unavailable, so theme conclusions are candidates pending confirmation."
                )
            else:
                lines.append(
                    "- **Data gap**: rankings contain percentage moves only; limit-up counts, turnover share, capacity leaders, front-row stocks, and diffusion chains are unavailable, so theme conclusions are candidates pending confirmation."
                )
        else:
            if has_fund_flow:
                lines.append(
                    "- **板块数据边界**：当前榜单可结合资金净流入/流出板块交叉验证；仍未提供容量核心、前排个股和扩散链，热点只能按候选排序，不能直接确认主线。"
                )
            else:
                lines.append(
                    "- **板块数据边界**：当前榜单只含涨跌幅，未提供涨停数量、板块成交额、容量核心、前排个股和扩散链；热点只能按候选排序，不能直接确认主线。"
                )

        def append_ranking(title: str, name_label: str, rows: List[Dict]) -> None:
            if not rows:
                return
            if lines:
                lines.append("")
            lines.extend([
                title,
                f"| {'Rank' if language == 'en' else '排名'} | {name_label} | {'Change' if language == 'en' else '涨跌幅'} |",
                "|------|------|--------|",
            ])
            for rank, item in enumerate(rows[:5], 1):
                lines.append(
                    f"| {rank} | {item.get('name', '-')} | {self._format_signed_pct(item.get('change_pct'))} |"
                )

        if language == "en":
            append_ranking("#### Leading Industry Sectors", "Sector", overview.top_sectors)
            append_ranking("#### Lagging Industry Sectors", "Sector", overview.bottom_sectors)
            append_ranking("#### Leading Concept Themes", "Concept", overview.top_concepts)
            append_ranking("#### Lagging Concept Themes", "Concept", overview.bottom_concepts)
        else:
            append_ranking("#### 行业板块领涨 Top 5", "行业板块", overview.top_sectors)
            append_ranking("#### 行业板块领跌 Top 5", "行业板块", overview.bottom_sectors)
            append_ranking("#### 概念板块领涨 Top 5", "概念板块", overview.top_concepts)
            append_ranking("#### 概念板块领跌 Top 5", "概念板块", overview.bottom_concepts)
        return "\n".join(lines)

    def _build_news_block(self, news: List) -> str:
        """Build a compact source-aware news catalyst list for the rendered report."""
        if not news:
            return ""
        language = self._get_review_language()
        if language == "en":
            lines = [
                "#### News Catalysts",
            ]
        else:
            lines = [
                "#### 近三日市场线索",
            ]

        for idx, item in enumerate(news[:5], 1):
            lines.append(self._format_news_catalyst_line(idx, item, language=language))
        return "\n".join(lines)

    @staticmethod
    def _get_news_field(item: Any, field: str) -> str:
        if hasattr(item, field):
            value = getattr(item, field, "") or ""
        elif isinstance(item, dict):
            value = item.get(field, "") or ""
        else:
            value = ""
        return str(value).strip()

    @classmethod
    def _format_news_catalyst_line(cls, idx: int, item: Any, *, language: str = "zh") -> str:
        fallback_title = "Untitled catalyst" if language == "en" else "未命名线索"
        title = cls._compact_news_text(cls._get_news_field(item, "title"), limit=90) or fallback_title
        source = cls._compact_news_text(cls._get_news_field(item, "source"), limit=40)
        date_text = cls._compact_news_text(cls._get_news_field(item, "published_date"), limit=24)
        url = cls._compact_news_text(cls._get_news_field(item, "url"), limit=0)
        title_text = cls._escape_markdown_link_label(title)
        if url:
            title_text = f"[{title_text}]({url})"
        meta_parts = [part for part in (source, date_text) if part]
        if language == "en":
            meta = f" ({' / '.join(meta_parts)})" if meta_parts else ""
        else:
            meta = f"（{' / '.join(meta_parts)}）" if meta_parts else ""
        return f"- {idx}. {title_text}{meta}"

    @staticmethod
    def _compact_news_text(value: str, *, limit: int) -> str:
        text = " ".join(str(value or "").split())
        if limit <= 0 or len(text) <= limit:
            return text
        return text[: max(0, limit - 3)].rstrip() + "..."

    @staticmethod
    def _format_optional_number(value: float) -> str:
        return "N/A" if value in (None, 0, 0.0) else f"{value:.2f}"

    @staticmethod
    def _format_optional_pct(value: float) -> str:
        return "N/A" if value in (None, 0, 0.0) else f"{value:.2f}%"

    @staticmethod
    def _format_signed_pct(value: Any) -> str:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return "N/A"
        return f"{numeric_value:+.2f}%"

    @classmethod
    def _format_ranking_summary(cls, rows: List[Dict], limit: int = 3) -> str:
        parts = []
        for item in (rows or [])[:limit]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            parts.append(f"{name}({cls._format_signed_pct(item.get('change_pct'))})")
        return ", ".join(parts)

    @classmethod
    def _format_fund_flow_summary(cls, rows: List[Dict], limit: int = 3) -> str:
        parts = []
        for item in (rows or [])[:limit]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            try:
                net_inflow = float(item.get("net_inflow"))
            except (TypeError, ValueError):
                parts.append(name)
                continue
            parts.append(f"{name}({net_inflow:+.2f})")
        return ", ".join(parts)

    @staticmethod
    def _format_limit_up_industries(distribution: List[Dict]) -> str:
        parts = []
        for item in distribution or []:
            if not isinstance(item, dict):
                continue
            industry = str(item.get("industry") or "").strip()
            if not industry:
                continue
            try:
                count = int(item.get("count") or 0)
            except (TypeError, ValueError):
                count = 0
            if count > 0:
                parts.append(f"{industry}({count}家)")
        return "、".join(parts)

    @staticmethod
    def _format_limit_up_industries_en(distribution: List[Dict]) -> str:
        parts = []
        for item in distribution or []:
            if not isinstance(item, dict):
                continue
            industry = str(item.get("industry") or "").strip()
            if not industry:
                continue
            try:
                count = int(item.get("count") or 0)
            except (TypeError, ValueError):
                count = 0
            if count > 0:
                parts.append(f"{industry}({count})")
        return ", ".join(parts)

    @staticmethod
    def _format_optional_level(value: Any) -> str:
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return "N/A"

    @staticmethod
    def _escape_markdown_link_label(value: str) -> str:
        return value.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")

    @staticmethod
    def _describe_turnover(total_amount: float) -> str:
        if total_amount >= 15000:
            return "高活跃度"
        if total_amount >= 9000:
            return "中等活跃"
        if total_amount > 0:
            return "缩量观望"
        return "暂无数据"

    def _build_market_light_scores(self, overview: MarketOverview) -> Dict[str, Any]:
        """Build the canonical Market Light scores used by reports and alerts."""

        participants = overview.up_count + overview.down_count
        breadth_available = bool(self.profile.has_market_stats and participants > 0)
        breadth_score = 50
        if breadth_available:
            breadth_score = int(overview.up_count / participants * 100)

        index_changes = [idx.change_pct for idx in overview.indices if idx.change_pct is not None]
        index_available = bool(overview.indices and index_changes)
        index_score = 50
        if index_available:
            avg_change = sum(index_changes) / len(index_changes)
            index_score = int(max(0, min(100, 50 + avg_change * 12)))

        limit_total = overview.limit_up_count + overview.limit_down_count
        limit_available = bool(self.profile.has_market_stats and limit_total > 0)
        limit_score = 50
        if limit_available:
            limit_score = int(overview.limit_up_count / limit_total * 100)

        dimensions = {
            "breadth": {"score": breadth_score, "available": breadth_available},
            "index": {"score": index_score, "available": index_available},
            "limit": {"score": limit_score, "available": limit_available},
        }

        if not index_available:
            data_quality = "unavailable"
        elif all(dimension["available"] for dimension in dimensions.values()):
            data_quality = "ok"
        else:
            data_quality = "partial"

        score = int(round(breadth_score * 0.45 + index_score * 0.35 + limit_score * 0.20))
        if self._get_review_language() == "en":
            if score >= 70:
                label = "risk-on"
            elif score >= 55:
                label = "constructive"
            elif score >= 40:
                label = "mixed"
            else:
                label = "defensive"
        else:
            if score >= 70:
                label = "强势"
            elif score >= 55:
                label = "偏暖"
            elif score >= 40:
                label = "震荡"
            else:
                label = "偏弱"
        return {
            "score": score,
            "temperature_label": label,
            "dimensions": dimensions,
            "data_quality": data_quality,
        }

    def _build_market_temperature(self, overview: MarketOverview) -> tuple[int, str]:
        scores = self._build_market_light_scores(overview)
        score = int(scores["score"])
        label = str(scores["temperature_label"])
        return score, label

    def _build_data_scope_input_block(self, overview: MarketOverview) -> str:
        """Build the date/data-scope block passed to the LLM."""
        generated_at = overview.generated_at or "N/A"
        run_date = overview.run_date or overview.date
        data_date = overview.data_date or overview.date
        note = overview.data_scope_note or (
            f"报告数据口径按 {data_date} 交易日处理。"
            if self._get_review_language() != "en"
            else f"Report data is treated as the {data_date} trading session."
        )

        if self._get_review_language() == "en":
            non_trading = "yes" if overview.is_non_trading_run else "no"
            return f"""## Data Scope
- Generated at: {generated_at}
- Run date: {run_date}
- Effective trading/data date: {data_date}
- Non-trading or after-hours reuse: {non_trading}
- Scope note: {note}"""

        non_trading = "是" if overview.is_non_trading_run else "否"
        return f"""## 数据口径
- 生成时间: {generated_at}
- 运行自然日: {run_date}
- 实际数据交易日: {data_date}
- 是否非交易日/盘外复用: {non_trading}
- 口径说明: {note}"""

    def _build_data_gap_input_block(self, overview: MarketOverview) -> str:
        """State unsupported strategy inputs explicitly so the model does not infer them."""
        has_index_levels = bool(getattr(overview, "index_key_levels", None))
        has_limit_structure = bool(getattr(overview, "limit_up_structure", None))
        has_fund_flow = bool(
            getattr(overview, "fund_inflow_sectors", None)
            or getattr(overview, "fund_outflow_sectors", None)
        )
        evidence = getattr(overview, "a_share_evidence", {}) or {}
        sentiment = evidence.get("sentiment_structure") if isinstance(evidence, dict) else {}
        sentiment = sentiment if isinstance(sentiment, dict) else {}
        has_short_index_mas = any(
            isinstance(item, dict) and item.get("ma5") is not None and item.get("ma10") is not None
            for item in (evidence.get("index_trend") or [])
        ) if isinstance(evidence, dict) else False
        has_sentiment_quality = any(
            sentiment.get(key) is not None
            for key in ("broken_ratio", "previous_limit_premium_median_pct", "one_price_like_ratio")
        )
        has_theme_candidates = bool(evidence.get("theme_candidates")) if isinstance(evidence, dict) else False
        has_stock_candidates = bool(evidence.get("stock_candidates")) if isinstance(evidence, dict) else False
        if self._get_review_language() == "en":
            if has_index_levels:
                if has_short_index_mas:
                    lines = [
                        "- Index trend includes MA5/MA10/MA20 and 20-day high/low when the daily-history source succeeds; volume-at-price support/resistance remains unavailable.",
                    ]
                else:
                    lines = [
                        "- Index key levels include MA20 and 20-day high/low when provided; MA5/MA10 and volume-at-price support/resistance are still unavailable.",
                    ]
            else:
                lines = [
                    "- Index moving averages (MA5/MA10/MA20), prior highs/lows, and volume-at-price support/resistance are not available in this payload.",
                ]
            if self.profile.has_market_stats:
                if has_sentiment_quality:
                    lines.append(
                        "- Sentiment quality includes failed-board ratio, previous-limit premium, continuation height, and a one-price-board heuristic when their date-scoped pools are available."
                    )
                elif has_limit_structure:
                    lines.append(
                        "- Limit-up structure includes current limit-up pool size, maximum continuation height, break count, and industry distribution when provided; previous-limit-up premium, broken-board feedback, and one-tick board ratio are still not available."
                    )
                else:
                    lines.append(
                        "- Breadth contains advancers/decliners and limit-up/limit-down counts only; failed boards, limit-up continuation height, previous-limit-up premium, broken-board feedback, and one-tick board ratio are not available."
                    )
            else:
                lines.append("- Breadth and limit-up/limit-down structure are not available for this market.")
            if self.profile.has_sector_rankings:
                if has_theme_candidates and has_stock_candidates:
                    lines.append(
                        "- Theme and stock watchlists are evidence-ranked candidates from sector/concept rankings and date-scoped limit-up pools; trend-core coverage outside those pools remains unavailable."
                    )
                elif has_fund_flow:
                    lines.append(
                        "- Sector/theme rankings include percentage moves and fund-flow leaders when provided; capacity leaders, front-row stocks, and diffusion chains are still not available."
                    )
                else:
                    lines.append(
                        "- Sector/theme rankings contain percentage moves only; limit-up counts, turnover share, capacity leaders, front-row stocks, and diffusion chains are not available."
                    )
            else:
                lines.append("- Sector/theme rankings are not available for this market.")
            lines.append(
                "- Therefore, classify themes as candidates only; do not promote a one-day ranking into a confirmed main line without confirmation conditions."
            )
            return "## Data Gaps\n" + "\n".join(lines)

        if has_index_levels:
            if has_short_index_mas:
                lines = [
                    "- 当前指数趋势可提供 MA5/MA10/MA20 和 20 日高低点（若日线源成功）；仍未提供成交密集区，不能把关键位写成精确预测。",
                ]
            else:
                lines = [
                    "- 当前指数关键位可提供 MA20 和 20 日高低点（若数据源成功）；仍未提供 MA5/MA10 和成交密集区，不能把关键位写成精确预测。",
                ]
        else:
            lines = [
                "- 当前指数数据未提供 MA5/MA10/MA20、前高前低、成交密集区，不能精确给出技术支撑/压力。",
            ]
        if self.profile.has_market_stats:
            if has_sentiment_quality:
                lines.append(
                    "- 当前接力质量可使用炸板率、昨日涨停溢价、连板高度和疑似一字板比例；缺失的日期化事件池必须按数据质量标记降级。"
                )
            elif has_limit_structure:
                lines.append(
                    "- 当前涨停结构可提供涨停池、最高连板、炸板次数和行业分布（若数据源成功）；仍未提供昨日涨停溢价、断板反馈和一字板比例。"
                )
            else:
                lines.append(
                    "- 当前情绪数据只有涨跌家数、涨跌停家数和成交额；未提供炸板率、连板高度、昨日涨停溢价、断板反馈、一字板比例，不能评估接力质量。"
                )
        else:
            lines.append("- 当前市场不提供涨跌家数、涨跌停和成交额汇总，不能评估市场宽度与短线接力。")
        if self.profile.has_sector_rankings:
            if has_theme_candidates and has_stock_candidates:
                lines.append(
                    "- 当前已提供基于板块排行与日期化涨停池交叉评分的题材候选和观察票；涨停池之外的趋势核心仍需个股分析另行确认。"
                )
            elif has_fund_flow:
                lines.append(
                    "- 当前板块/题材榜可结合涨跌幅与资金净流入/流出榜（若数据源成功）；仍未提供容量核心、前排个股和扩散链，不能直接认定主线。"
                )
            else:
                lines.append(
                    "- 当前板块/题材榜只有涨跌幅；未提供涨停数量、板块成交额、容量核心、前排个股和扩散链，不能直接认定主线。"
                )
        else:
            lines.append("- 当前市场不提供行业/概念涨跌榜，不能做热点排序。")
        lines.append(
            "- 因此只能给“候选方向 + 次日确认条件”，不能把单日快照写成确定性交易主线。"
        )
        return "## 数据缺口\n" + "\n".join(lines)

    def _build_a_share_evidence_input_block(self, overview: MarketOverview) -> str:
        evidence = getattr(overview, "a_share_evidence", {}) or {}
        if self.region != "cn" or not isinstance(evidence, dict) or not evidence:
            return ""

        sentiment = evidence.get("sentiment_structure") or {}
        themes = [item for item in (evidence.get("theme_candidates") or []) if isinstance(item, dict)][:6]
        stocks = [item for item in (evidence.get("stock_candidates") or []) if isinstance(item, dict)][:10]
        risk_rules = [item for item in (evidence.get("risk_rules") or []) if isinstance(item, dict)]
        quality = evidence.get("data_quality") or {}

        if self._get_review_language() == "en":
            lines = [
                "## A-share Sentiment and Theme Evidence",
                f"- Evidence status: {self._prompt_cell(evidence.get('status') or 'unknown')}",
                (
                    "- Sentiment: limit-up {limit_up}; failed boards {broken}; failed-board ratio {ratio}; "
                    "limit-down {limit_down}; highest board {height}; previous-limit premium median {premium}; "
                    "one-price heuristic {one_price}."
                ).format(
                    limit_up=sentiment.get("limit_up_count", "N/A"),
                    broken=self._format_prompt_metric(sentiment.get("broken_board_count")),
                    ratio=self._format_prompt_ratio(sentiment.get("broken_ratio")),
                    limit_down=self._format_prompt_metric(sentiment.get("limit_down_count")),
                    height=self._format_prompt_metric(sentiment.get("highest_consecutive_board")),
                    premium=self._format_prompt_pct(sentiment.get("previous_limit_premium_median_pct")),
                    one_price=self._format_prompt_ratio(sentiment.get("one_price_like_ratio")),
                ),
            ]
            triggered = [item for item in risk_rules if item.get("triggered")]
            if triggered:
                lines.append("- Triggered risk gates:")
                lines.extend(
                    f"  - {self._prompt_cell(item.get('code'))}: {self._prompt_cell(item.get('evidence'))}; {self._prompt_cell(item.get('action'))}"
                    for item in triggered
                )
            if themes:
                lines.extend([
                    "### Ranked Theme Candidates",
                    "| Theme | Class | Score | Limit-up | Height | Confirmation | Invalidation |",
                    "| --- | --- | ---: | ---: | ---: | --- | --- |",
                ])
                lines.extend(
                    "| {theme} | {classification} | {score} | {limit_up} | {height} | {confirmation} | {invalidation} |".format(
                        theme=self._prompt_cell(item.get("theme")),
                        classification=self._prompt_cell(item.get("classification")),
                        score=self._format_prompt_metric(item.get("total_score")),
                        limit_up=self._format_prompt_metric(item.get("limit_up_count")),
                        height=self._format_prompt_metric(item.get("max_board_height")),
                        confirmation=self._prompt_cell(item.get("confirmation")),
                        invalidation=self._prompt_cell(item.get("invalidation")),
                    )
                    for item in themes
                )
            if stocks:
                lines.extend([
                    "### Core Watchlist Candidates",
                    "| Role | Code | Name | Theme | Entry type | Confirmation | Invalidation | Risk tags |",
                    "| --- | --- | --- | --- | --- | --- | --- | --- |",
                ])
                lines.extend(self._format_stock_candidate_prompt_row(item) for item in stocks)
        else:
            lines = [
                "## A股情绪与题材证据",
                f"- 证据状态: {self._prompt_cell(evidence.get('status') or 'unknown')}",
                (
                    "- 情绪结构: 涨停 {limit_up} 家；炸板 {broken} 家；炸板率 {ratio}；跌停 {limit_down} 家；"
                    "最高 {height} 板；昨日涨停溢价中位数 {premium}；疑似一字/无换手占比 {one_price}。"
                ).format(
                    limit_up=sentiment.get("limit_up_count", "N/A"),
                    broken=self._format_prompt_metric(sentiment.get("broken_board_count")),
                    ratio=self._format_prompt_ratio(sentiment.get("broken_ratio")),
                    limit_down=self._format_prompt_metric(sentiment.get("limit_down_count")),
                    height=self._format_prompt_metric(sentiment.get("highest_consecutive_board")),
                    premium=self._format_prompt_pct(sentiment.get("previous_limit_premium_median_pct")),
                    one_price=self._format_prompt_ratio(sentiment.get("one_price_like_ratio")),
                ),
            ]
            triggered = [item for item in risk_rules if item.get("triggered")]
            if triggered:
                lines.append("- 已触发风险门槛:")
                lines.extend(
                    f"  - {self._prompt_cell(item.get('code'))}: {self._prompt_cell(item.get('evidence'))}；{self._prompt_cell(item.get('action'))}"
                    for item in triggered
                )
            if themes:
                lines.extend([
                    "### 题材候选排序",
                    "| 题材 | 分类 | 分数 | 涨停数 | 高度 | 次日确认 | 失效条件 |",
                    "| --- | --- | ---: | ---: | ---: | --- | --- |",
                ])
                lines.extend(
                    "| {theme} | {classification} | {score} | {limit_up} | {height} | {confirmation} | {invalidation} |".format(
                        theme=self._prompt_cell(item.get("theme")),
                        classification=self._prompt_cell(item.get("classification")),
                        score=self._format_prompt_metric(item.get("total_score")),
                        limit_up=self._format_prompt_metric(item.get("limit_up_count")),
                        height=self._format_prompt_metric(item.get("max_board_height")),
                        confirmation=self._prompt_cell(item.get("confirmation")),
                        invalidation=self._prompt_cell(item.get("invalidation")),
                    )
                    for item in themes
                )
            if stocks:
                lines.extend([
                    "### 核心观察票候选",
                    "| 类别 | 代码 | 名称 | 题材 | 买点类型 | 验证条件 | 失效条件 | 风险标签 |",
                    "| --- | --- | --- | --- | --- | --- | --- | --- |",
                ])
                lines.extend(self._format_stock_candidate_prompt_row(item) for item in stocks)

        missing = quality.get("missing_fields") if isinstance(quality, dict) else None
        contaminated = quality.get("contaminated_fields") if isinstance(quality, dict) else None
        errors = quality.get("errors") if isinstance(quality, dict) else None
        if missing:
            lines.append(f"- Missing fields: {', '.join(self._prompt_cell(item) for item in missing)}")
        if contaminated:
            lines.append(f"- Unusable/contaminated fields: {', '.join(self._prompt_cell(item) for item in contaminated)}")
        if errors:
            lines.append(f"- Source failures: {'; '.join(self._prompt_cell(item) for item in errors[:6])}")
        return "\n".join(lines)

    @staticmethod
    def _prompt_cell(value: Any) -> str:
        return " ".join(str(value or "").replace("|", "/").split()) or "N/A"

    @staticmethod
    def _format_prompt_metric(value: Any) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.2f}".rstrip("0").rstrip(".")
        return str(value)

    @staticmethod
    def _format_prompt_ratio(value: Any) -> str:
        try:
            return f"{float(value):.1%}"
        except (TypeError, ValueError):
            return "N/A"

    @staticmethod
    def _format_prompt_pct(value: Any) -> str:
        try:
            return f"{float(value):+.2f}%"
        except (TypeError, ValueError):
            return "N/A"

    def _format_stock_candidate_prompt_row(self, item: Dict[str, Any]) -> str:
        themes = item.get("themes") or []
        risk_tags = item.get("risk_tags") or []
        return "| {category} | {code} | {name} | {themes} | {buy_point} | {validation} | {invalidation} | {risk_tags} |".format(
            category=self._prompt_cell(item.get("category")),
            code=self._prompt_cell(item.get("code")),
            name=self._prompt_cell(item.get("name")),
            themes=self._prompt_cell("、".join(str(value) for value in themes)),
            buy_point=self._prompt_cell(item.get("buy_point_type")),
            validation=self._prompt_cell(item.get("validation")),
            invalidation=self._prompt_cell(item.get("invalidation")),
            risk_tags=self._prompt_cell("、".join(str(value) for value in risk_tags)),
        )

    def _build_data_scope_report_block(self, overview: MarketOverview) -> str:
        """Build a deterministic report section for date/data scope."""
        generated_at = overview.generated_at or "N/A"
        run_date = overview.run_date or overview.date
        data_date = overview.data_date or overview.date
        note = overview.data_scope_note or f"报告数据口径按 {data_date} 交易日处理。"

        if self._get_review_language() == "en":
            return "\n".join(
                [
                    "### 1. Data Scope",
                    "| Item | Scope |",
                    "|------|-------|",
                    f"| Generated at | {generated_at} |",
                    f"| Run date | {run_date} |",
                    f"| Effective trading/data date | {data_date} |",
                    f"| Non-trading or after-hours reuse | {'yes' if overview.is_non_trading_run else 'no'} |",
                    f"| Scope note | {note} |",
                ]
            )

        return "\n".join(
            [
                "### 一、数据口径",
                "| 项目 | 口径 |",
                "|------|------|",
                f"| 生成时间 | {generated_at} |",
                f"| 运行自然日 | {run_date} |",
                f"| 实际数据交易日 | {data_date} |",
                f"| 是否非交易日/盘外复用 | {'是' if overview.is_non_trading_run else '否'} |",
                f"| 说明 | {note} |",
            ]
        )

    def _build_output_template_sections(self, review_language: str) -> str:
        """Build LLM output sections according to market data capabilities."""
        if review_language == "en":
            if self.profile.has_market_stats and self.profile.has_sector_rankings:
                return """### 1. Data Scope
(State generated time, effective trading/data date, and whether the report was generated on a non-trading or after-hours date.)

### 2. Market Summary
(Summarize tone, breadth, and whether the report is only a snapshot or can support a next-session plan.)

### 3. Index Risk Gates
(Discuss index strength/weakness using provided key levels when available; explicitly say if MA5/MA10/MA20 or support/resistance data is unavailable.)

### 4. Sentiment Temperature
(Use turnover, breadth, limit-up/down counts, and the limit-up structure when provided. If board-failure rate, continuation height, previous-limit-up premium, broken-board feedback, or one-tick board ratio is absent, explicitly state the limitation.)

### 5. Theme Ranking
(Classify sectors/themes as main-line candidates, diffusion, laggards, or unconfirmed one-day rotation. Cross-check gain/loss rankings with fund inflow/outflow leaders; if a sector rises while showing fund outflow, flag questionable persistence. When global context shows a large related move, explain the transmission chain.)

### 6. Core Watchlist
(If no stock-level leaders are provided, say no verifiable watchlist is available and give only sector-level observation conditions.)

### 7. Next-Session Strategy
(Start with what not to buy, then list confirmation conditions, entry trigger types, invalidation triggers, and position caps. Invalidation triggers must be anchored to provided observable data such as index key levels, global indices, limit-up structure, or sector fund-flow persistence. Do not use unverifiable wording like "if the market weakens". If main-line evidence is incomplete, use observation/trial positions before raising exposure.)

### 8. Risk Alerts
(List the main risks to monitor and end with "For reference only, not investment advice.")"""

            sections: List[str] = [
                """### 1. Data Scope
(State generated time, effective trading/data date, and whether the report was generated on a non-trading or after-hours date.)""",
                """### 2. Market Summary
(Summarize tone, available breadth/liquidity inputs, and whether the report is only a snapshot or can support a next-session plan.)""",
                """### 3. Index Risk Gates
(Discuss index strength/weakness using provided key levels when available; explicitly say if MA5/MA10/MA20 or support/resistance data is unavailable.)""",
            ]
            section_number = 4
            if self.profile.has_market_stats:
                sections.append(f"""### {section_number}. Sentiment Temperature
(Interpret only the provided turnover, participation, breadth, and flow signals. Use limit-up structure when provided and explicitly identify missing short-term continuation data.)""")
                section_number += 1
            if self.profile.has_sector_rankings:
                sections.append(f"""### {section_number}. Theme Ranking
(Analyze only the provided industry-sector and concept/theme rankings; classify them as candidates, not confirmed main lines, if leader/turnover/limit-up-chain evidence is unavailable.)""")
                section_number += 1
            sections.extend([
                f"""### {section_number}. News Catalysts
(Connect recent news to index price action and macro/external-market clues. Do not infer unsupported breadth, fund-flow, or sector-ranking data.)""",
                f"""### {section_number + 1}. Outlook
(Provide the near-term outlook based on index price action and the available news.)""",
                f"""### {section_number + 2}. Risk Alerts
(List the main risks to monitor.)""",
                f"""### {section_number + 3}. Strategy Plan
(Start with what not to buy, then list confirmation conditions, entry trigger types, invalidation triggers, and position caps. Anchor invalidation to observable data when available. End with "For reference only, not investment advice.")""",
            ])
            return "\n\n".join(sections)

        if self.profile.has_market_stats and self.profile.has_sector_rankings:
            return """### 一、数据口径
（写明生成时间、实际数据交易日、是否非交易日/盘外复用；明确“今日/明日”的交易日含义）

### 二、盘面总览
（概括指数、涨跌家数、成交额和情绪温度，明确这是大盘快照还是可执行策略）

### 三、大盘风险门槛
（说明上证、沪深300、创业板、科创50强弱；有指数关键位时引用关键位，没有 MA5/MA10/MA20、前高前低和支撑压力数据时必须明确写“暂无精确门槛”，不得编造）

### 四、情绪温度
（解读成交额、涨跌停、市场宽度；有涨停结构时结合连板高度、炸板情况判断情绪强度与亏钱效应；缺少昨日涨停溢价、断板反馈和一字板比例时必须说明，区分“热度”和“接力质量”）

### 五、热点排序
（把行业/概念分为：主线候选、扩散、补涨、伪相关/单日轮动；交叉验证资金净流入/流出榜与涨跌幅榜，涨幅高但资金流出的板块提示持续性存疑；若外围市场数据显示相关行业大幅波动，必须解释与 A 股板块的联动关系）

### 六、核心观察票
（若未提供个股龙头/容量票/趋势核心数据，必须写“暂无可验证观察票”，只能给板块级观察条件，不得编造股票）

### 七、明日交易计划
（先写不能买什么，再写确认条件，最后写买点类型、失效位和仓位上限；触发失效条件必须引用已提供的可观察锚点（指数关键位、外围指数、涨停结构或板块资金持续性），禁止使用“若市场走弱”这类不可验证表述；主线未确认时以观察/试错仓为主，只有指数与主线共振确认后再升仓）

### 八、风险提示
（列出需要关注的风险点；最后补充“建议仅供参考，不构成投资建议”。）"""

        numerals = ["一", "二", "三", "四", "五", "六", "七", "八"]
        section_number = 1
        sections: List[str] = []

        def add_section(title: str, hint: str) -> None:
            nonlocal section_number
            sections.append(f"### {numerals[section_number - 1]}、{title}\n{hint}")
            section_number += 1

        add_section("数据口径", "（写明生成时间、实际数据交易日、是否非交易日/盘外复用；明确“今日/明日”的交易日含义）")
        add_section("盘面总览", "（概括指数、可用市场宽度、成交活跃度和整体风险状态，明确这是大盘快照还是可执行策略）")
        add_section("大盘风险门槛", "（说明指数强弱；有指数关键位时引用关键位；没有 MA5/MA10/MA20、前高前低和支撑压力数据时必须明确写“暂无精确门槛”，不得编造）")
        if self.profile.has_sector_rankings:
            add_section("热点排序", "（仅分析已提供的行业板块与概念题材榜单；缺少龙头/成交额/涨停链时只给候选排序，不确认主线）")
        if self.profile.has_market_stats:
            add_section("情绪温度", "（仅解读已提供的成交额、涨跌停结构、市场宽度和风险偏好数据；有涨停结构时引用，缺失时明确缺失的接力质量指标）")
        add_section(
            "消息催化",
            "（结合近三日新闻和指数表现，提炼真正影响明日交易的催化或扰动；不要推断未提供的资金流、市场宽度或板块榜）",
        )
        add_section("明日交易计划", "（先写不能买什么，再写确认条件、买点类型、失效位和仓位上限；尽量锚定已提供的可观察数据）")
        add_section("风险提示", "（列出需要关注的风险点；最后补充“建议仅供参考，不构成投资建议”。）")
        return "\n\n".join(sections)

    def _build_review_prompt(self, overview: MarketOverview, news: List) -> str:
        """构建复盘报告 Prompt"""
        review_language = self._get_review_language()
        # Korean reuses the English structural template but the model is told to
        # write the entire shell, headings, guidance and conclusion in Korean.
        shell_language_label = "Korean (한국어)" if self._get_output_language() == "ko" else "English"

        # 指数行情信息（简洁格式，不用emoji）
        indices_text = ""
        for idx in overview.indices:
            direction = "↑" if idx.change_pct > 0 else "↓" if idx.change_pct < 0 else "-"
            indices_text += f"- {idx.name}: {idx.current:.2f} ({direction}{abs(idx.change_pct):.2f}%)\n"

        # 外围市场联动信息（仅 A 股复盘注入）
        global_text = ""
        for idx in overview.global_indices:
            direction = "↑" if idx.change_pct > 0 else "↓" if idx.change_pct < 0 else "-"
            global_text += f"- {idx.name}: {idx.current:.2f} ({direction}{abs(idx.change_pct):.2f}%)\n"

        # 板块信息
        top_sectors_text = self._format_ranking_summary(overview.top_sectors)
        bottom_sectors_text = self._format_ranking_summary(overview.bottom_sectors)
        top_concepts_text = self._format_ranking_summary(overview.top_concepts)
        bottom_concepts_text = self._format_ranking_summary(overview.bottom_concepts)
        fund_inflow_text = self._format_fund_flow_summary(getattr(overview, "fund_inflow_sectors", []))
        fund_outflow_text = self._format_fund_flow_summary(getattr(overview, "fund_outflow_sectors", []))

        limit_up_block = ""
        limit_up_structure = getattr(overview, "limit_up_structure", {}) or {}
        if self.profile.has_market_stats and limit_up_structure:
            max_stock = str(limit_up_structure.get("max_boards_stock") or "").strip()
            max_boards = int(limit_up_structure.get("max_consecutive_boards") or 0)
            total_break_count = int(limit_up_structure.get("total_break_count") or 0)
            industry_distribution = limit_up_structure.get("industry_distribution") or []
            if review_language == "en":
                max_stock_text = f" ({max_stock})" if max_stock else ""
                industry_text = self._format_limit_up_industries_en(industry_distribution)
                limit_up_block = (
                    "## Limit-up Structure\n"
                    f"- Limit-up pool: {int(limit_up_structure.get('total') or 0)} stocks | "
                    f"Highest consecutive boards: {max_boards}{max_stock_text} | "
                    f"Total break count: {total_break_count}\n"
                )
                if industry_text:
                    limit_up_block += f"- Limit-up industry distribution: {industry_text}"
            else:
                max_stock_text = f"（{max_stock}）" if max_stock else ""
                industry_text = self._format_limit_up_industries(industry_distribution)
                limit_up_block = (
                    "## 涨停结构\n"
                    f"- 涨停池数量: {int(limit_up_structure.get('total') or 0)} 家 | "
                    f"最高连板: {max_boards} 板{max_stock_text} | 炸板次数合计: {total_break_count}\n"
                )
                if industry_text:
                    limit_up_block += f"- 涨停行业分布: {industry_text}"

        index_key_levels_block = ""
        index_key_levels = getattr(overview, "index_key_levels", []) or []
        if index_key_levels:
            if review_language == "en":
                rows = [
                    "| Index | Current | MA5 | MA10 | MA20 | 20D High | 20D Low |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
                for item in index_key_levels:
                    rows.append(
                        "| {name} | {current} | {ma5} | {ma10} | {ma20} | {high_20d} | {low_20d} |".format(
                            name=item.get("name") or "",
                            current=self._format_optional_level(item.get("current")),
                            ma5=self._format_optional_level(item.get("ma5")),
                            ma10=self._format_optional_level(item.get("ma10")),
                            ma20=self._format_optional_level(item.get("ma20")),
                            high_20d=self._format_optional_level(item.get("high_20d")),
                            low_20d=self._format_optional_level(item.get("low_20d")),
                        )
                    )
                index_key_levels_block = (
                    "## Index Key Levels Reference (daily-bar local calculation, not a forecast)\n"
                    + "\n".join(rows)
                )
            else:
                rows = [
                    "| 指数 | 现价 | MA5 | MA10 | MA20 | 近20日高 | 近20日低 |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
                for item in index_key_levels:
                    rows.append(
                        "| {name} | {current} | {ma5} | {ma10} | {ma20} | {high_20d} | {low_20d} |".format(
                            name=item.get("name") or "",
                            current=self._format_optional_level(item.get("current")),
                            ma5=self._format_optional_level(item.get("ma5")),
                            ma10=self._format_optional_level(item.get("ma10")),
                            ma20=self._format_optional_level(item.get("ma20")),
                            high_20d=self._format_optional_level(item.get("high_20d")),
                            low_20d=self._format_optional_level(item.get("low_20d")),
                        )
                    )
                index_key_levels_block = "## 指数关键位参考（基于日线本地计算，非预测）\n" + "\n".join(rows)

        p2_context_blocks = [block for block in (limit_up_block, index_key_levels_block) if block]
        p2_context_block = ("\n\n" + "\n\n".join(p2_context_blocks)) if p2_context_blocks else ""
        
        # 新闻信息 - 支持 SearchResult 对象或字典
        news_text = ""
        for i, n in enumerate(news[:6], 1):
            # 兼容 SearchResult 对象和字典
            title = self._compact_news_text(self._get_news_field(n, "title"), limit=90)
            snippet = self._compact_news_text(self._get_news_field(n, "snippet"), limit=220)
            source = self._compact_news_text(self._get_news_field(n, "source"), limit=60)
            published_date = self._compact_news_text(self._get_news_field(n, "published_date"), limit=30)
            url = self._compact_news_text(self._get_news_field(n, "url"), limit=180)
            meta_parts = [part for part in (source, published_date) if part]
            meta = f" ({' / '.join(meta_parts)})" if meta_parts else ""
            url_line = f"\n   URL: {url}" if url else ""
            news_text += f"{i}. {title}{meta}\n   {snippet or '-'}{url_line}\n"

        data_scope_block = self._build_data_scope_input_block(overview)
        data_gap_block = self._build_data_gap_input_block(overview)
        a_share_evidence_block = self._build_a_share_evidence_input_block(overview)
        
        # 外围市场联动块（仅 has_global_context 市场注入；数据缺失时给显式反幻觉指令）
        global_block = ""
        global_requirement = ""
        if self.profile.has_global_context:
            if review_language == "en":
                if global_text:
                    global_block = (
                        "## Global Market Context (US indices are the latest close; mind the time difference)\n"
                        f"{global_text}"
                    )
                    global_requirement = (
                        "- Global context is provided: first judge whether today's A-share move was driven by "
                        "overseas markets, explain the transmission chain in Market Summary / catalysts, and only "
                        "cite the provided global indices.\n"
                    )
                else:
                    global_block = (
                        "## Global Market Context\n"
                        "- Global market data was not fetched this time. Mark any cross-market judgement as "
                        "\"data missing\"; do not invent overseas index moves."
                    )
            else:
                if global_text:
                    global_block = (
                        "## 外围市场联动（美股为最近收盘价，注意时差）\n"
                        f"{global_text}"
                    )
                    global_requirement = (
                        "- 已提供外围市场数据：必须先判断当日 A 股走势是否由外围事件驱动，在盘面总览与消息催化中说明传导链条"
                        "（例如外围半导体大跌 → A 股相关板块承压）；只可引用已提供的外围指数，禁止编造其他外围数据\n"
                    )
                else:
                    global_block = (
                        "## 外围市场联动\n"
                        "- 外围市场数据本次未获取。涉及外围驱动的判断必须注明“数据缺失”，禁止编造隔夜美股或亚太市场表现。"
                    )

        # 按 region 组装市场概况与板块区块（美股/港股/日韩无涨跌家数、板块数据）
        stats_block = ""
        sector_block = ""
        data_limits_block = ""
        if review_language == "en":
            if self.profile.has_market_stats:
                stats_block = f"""## Market Breadth
- Advancers: {overview.up_count} | Decliners: {overview.down_count} | Flat: {overview.flat_count}
- Limit-up: {overview.limit_up_count} | Limit-down: {overview.limit_down_count}
- Turnover: {overview.total_amount:.0f} ({self._get_turnover_unit_label()})"""

            if self.profile.has_sector_rankings:
                sector_block = f"""## Sector / Theme Performance
Industry leading: {top_sectors_text if top_sectors_text else "N/A"}
Industry lagging: {bottom_sectors_text if bottom_sectors_text else "N/A"}
Concept leading: {top_concepts_text if top_concepts_text else "N/A"}
Concept lagging: {bottom_concepts_text if bottom_concepts_text else "N/A"}"""
                if fund_inflow_text:
                    sector_block += f"\nFund inflow leaders: {fund_inflow_text}"
                if fund_outflow_text:
                    sector_block += f"\nFund outflow laggards: {fund_outflow_text}"

            data_limit_lines = []
            if not self.profile.has_market_stats:
                data_limit_lines.append(
                    "- Market breadth, aggregate turnover, participation, and fund-flow signals are not available for this market."
                )
            if not self.profile.has_sector_rankings:
                data_limit_lines.append("- Sector/theme ranking data is not available for this market.")
            if data_limit_lines:
                data_limits_block = "## Data Limits\n" + "\n".join(data_limit_lines)
        else:
            if self.profile.has_market_stats:
                stats_block = f"""## 市场概况
- 上涨: {overview.up_count} 家 | 下跌: {overview.down_count} 家 | 平盘: {overview.flat_count} 家
- 涨停: {overview.limit_up_count} 家 | 跌停: {overview.limit_down_count} 家
- 两市成交额: {overview.total_amount:.0f} 亿元"""

            if self.profile.has_sector_rankings:
                sector_block = f"""## 板块表现
行业领涨: {top_sectors_text if top_sectors_text else "暂无数据"}
行业领跌: {bottom_sectors_text if bottom_sectors_text else "暂无数据"}
概念领涨: {top_concepts_text if top_concepts_text else "暂无数据"}
概念领跌: {bottom_concepts_text if bottom_concepts_text else "暂无数据"}"""
                if fund_inflow_text:
                    sector_block += f"\n资金净流入板块: {fund_inflow_text}"
                if fund_outflow_text:
                    sector_block += f"\n资金净流出板块: {fund_outflow_text}"

            data_limit_lines = []
            if not self.profile.has_market_stats:
                data_limit_lines.append("- 该市场暂无涨跌家数、涨跌停、成交额汇总、参与度或资金流信号。")
            if not self.profile.has_sector_rankings:
                data_limit_lines.append("- 该市场暂无行业板块/概念题材涨跌榜。")
            if data_limit_lines:
                data_limits_block = "## 数据边界\n" + "\n".join(data_limit_lines)

        data_no_indices_hint = (
            "注意：由于行情数据获取失败，请主要根据【市场新闻】进行定性分析和总结，不要编造具体的指数点位。"
            if not indices_text
            else ""
        )
        if review_language == "en":
            data_no_indices_hint = (
                "Note: Market data fetch failed. Rely mainly on [Market News] for qualitative analysis. Do not invent index levels."
                if not indices_text
                else ""
            )
            news_search_status = getattr(self, "_news_search_status", "not_run")
            indices_placeholder = indices_text if indices_text else "No index data (API error)"
            if news_text:
                news_placeholder = news_text
            elif news_search_status == "no_search_service":
                news_placeholder = (
                    "News search is NOT configured (no search API key; see the search-engine section of "
                    ".env.example, e.g. BOCHA_API_KEYS / TAVILY_API_KEYS). This section is missing due to "
                    "configuration, NOT because there was no market-moving news. Mark news-side judgements "
                    "as \"data missing\"."
                )
            elif news_search_status in ("no_results", "error"):
                news_placeholder = (
                    "News search ran but returned no usable results. This does NOT mean there was no "
                    "market-moving news; write \"news search returned no results\" instead of \"no catalysts today\"."
                )
            else:
                news_placeholder = "No relevant news"
            data_boundary_requirement = (
                "- Respect Data Limits: do not invent or over-interpret unsupported breadth, fund-flow, turnover, participation, or sector-ranking data.\n"
                if data_limits_block
                else ""
            )
            execution_constraints = (
                "- If leader stocks, capacity leaders, and theme diffusion chains are not provided, do not invent a stock watchlist.\n"
                "- If main-line evidence is incomplete, frame exposure as observation/trial positions and only raise exposure after index and theme confirmation.\n"
                "- The strategy section must start with what not to buy, then confirmation conditions, entry trigger types, invalidation triggers, and position caps.\n"
            )
        else:
            news_search_status = getattr(self, "_news_search_status", "not_run")
            indices_placeholder = indices_text if indices_text else "暂无指数数据（接口异常）"
            if news_text:
                news_placeholder = news_text
            elif news_search_status == "no_search_service":
                news_placeholder = (
                    "新闻检索服务未配置（未设置搜索 API Key，可在 .env 配置 BOCHA_API_KEYS / TAVILY_API_KEYS 等）。"
                    "本节缺失是配置问题，不代表市场无重大消息；报告中涉及消息面的判断必须注明“消息面数据缺失”。"
                )
            elif news_search_status in ("no_results", "error"):
                news_placeholder = (
                    "已执行新闻检索但未获取到有效结果。这不代表市场无重大消息；"
                    "请勿输出“今日无消息/无催化”，应注明“消息面检索无结果”。"
                )
            else:
                news_placeholder = "暂无相关新闻"
            data_boundary_requirement = (
                "- 严格遵守数据边界：未提供涨跌家数、资金流、成交额汇总或板块榜时，不要编造或过度解读。\n"
                if data_limits_block
                else ""
            )
            execution_constraints = (
                "- 未提供个股龙头、容量核心、趋势核心、低位补涨清单时，不得编造具体观察票。\n"
                "- 主线证据不完整时，仓位表述必须以观察/试错仓为主，只能在指数与主线共振确认后提高仓位。\n"
                "- 明日交易计划必须按“先写不能买什么 -> 确认条件 -> 买点类型 -> 失效位 -> 仓位上限”的顺序输出。\n"
            )

        output_template_sections = self._build_output_template_sections(review_language)
        zh_market_scope_name = self._get_market_scope_name("zh")
        zh_report_title = f"{overview.date} 大盘复盘"
        if self.region in ("jp", "kr"):
            zh_report_title = f"{overview.date} {zh_market_scope_name}大盘复盘"
        workflow_hint = (
            "报告要像交易员盘后工作台：先给结论，再按数据表、主线、催化、计划展开"
            if self.profile.has_market_stats or self.profile.has_sector_rankings
            else "报告要像交易员盘后工作台：先给结论，再按指数、新闻催化和计划展开"
        )

        if review_language == "en":
            report_title = self._get_review_title(overview.date).removeprefix("## ").strip()
            return f"""You are a professional {self._get_market_scope_name('en')} analyst. Please produce a concise market recap report based on the data below.

[Requirements]
- Output pure Markdown only
- No JSON
- No code blocks
- Use emoji sparingly in headings (at most one per heading)
- The entire fixed shell, headings, guidance, and conclusion must be in {shell_language_label}
{data_boundary_requirement}{global_requirement}{execution_constraints}

---

# Today's Market Data

{data_scope_block}

{data_gap_block}

{a_share_evidence_block}

## Date
{overview.date}

## Major Indices
{indices_placeholder}

{global_block}

{stats_block}

{sector_block}{p2_context_block}

{data_limits_block}

## Market News
{news_placeholder}

{data_no_indices_hint}

{self._get_strategy_prompt_block()}

---

# Output Template (follow this structure)

## {report_title}

{output_template_sections}

---

Output the report content directly, no extra commentary.
"""

        # A 股场景使用中文提示语
        return f"""你是一位专业的{self._get_market_scope_name('zh')}分析师，请根据以下数据生成一份结构化的{self._get_market_scope_name('zh')}大盘复盘报告。

【重要】输出要求：
- 必须输出纯 Markdown 文本格式
- 禁止输出 JSON 格式
- 禁止输出代码块
- emoji 仅在标题处少量使用（每个标题最多1个）
- {workflow_hint}
- 不要重复列出已由系统注入的表格数据；正文负责解释表格背后的含义
{data_boundary_requirement}{global_requirement}{execution_constraints}

---

# 今日市场数据

{data_scope_block}

{data_gap_block}

{a_share_evidence_block}

## 日期
{overview.date}

## 主要指数
{indices_placeholder}

{global_block}

{stats_block}

{sector_block}{p2_context_block}

{data_limits_block}

## 市场新闻
{news_placeholder}

{data_no_indices_hint}

{self._get_strategy_prompt_block()}

---

# 输出格式模板（请严格按此格式输出）

## {zh_report_title}

> 一句话给出数据交易日市场状态、核心矛盾和下一交易日优先观察方向。

{output_template_sections}

---

请直接输出复盘报告内容，不要输出其他说明文字。
"""
    
    def _generate_template_review(self, overview: MarketOverview, news: List) -> str:
        """使用模板生成复盘报告（无大模型时的备选方案）"""
        template_language = self._get_template_review_language()
        mood_code = self.profile.mood_index_code
        # 根据 mood_index_code 查找对应指数
        # cn: mood_code="000001"，idx.code 可能为 "sh000001"（以 mood_code 结尾）
        # us: mood_code="SPX"，idx.code 直接为 "SPX"
        mood_index = next(
            (
                idx
                for idx in overview.indices
                if idx.code == mood_code or idx.code.endswith(mood_code)
            ),
            None,
        )
        if mood_index:
            if mood_index.change_pct > 1:
                market_mood = self._get_market_mood_text("strong_up", template_language)
            elif mood_index.change_pct > 0:
                market_mood = self._get_market_mood_text("mild_up", template_language)
            elif mood_index.change_pct > -1:
                market_mood = self._get_market_mood_text("mild_down", template_language)
            else:
                market_mood = self._get_market_mood_text("strong_down", template_language)
        else:
            market_mood = self._get_market_mood_text("range", template_language)
        
        # 指数行情（简洁格式）
        indices_text = ""
        for idx in overview.indices[:4]:
            direction = "↑" if idx.change_pct > 0 else "↓" if idx.change_pct < 0 else "-"
            indices_text += f"- **{idx.name}**: {idx.current:.2f} ({direction}{abs(idx.change_pct):.2f}%)\n"
        
        # 板块信息
        separator = ", " if template_language == "en" else "、"
        top_text = separator.join([s['name'] for s in overview.top_sectors[:3]])
        bottom_text = separator.join([s['name'] for s in overview.bottom_sectors[:3]])
        top_concept_text = separator.join([s['name'] for s in overview.top_concepts[:3]])
        bottom_concept_text = separator.join([s['name'] for s in overview.bottom_concepts[:3]])
        evidence = overview.a_share_evidence if isinstance(overview.a_share_evidence, dict) else {}
        sentiment = evidence.get("sentiment_structure") if isinstance(evidence.get("sentiment_structure"), dict) else {}
        theme_candidates = [
            item for item in (evidence.get("theme_candidates") or []) if isinstance(item, dict)
        ][:5]
        stock_candidates = [
            item for item in (evidence.get("stock_candidates") or []) if isinstance(item, dict)
        ][:8]

        if template_language == "en":
            data_scope_section = self._build_data_scope_report_block(overview)
            data_gap_section = self._build_data_gap_input_block(overview).replace("## Data Gaps", "### Data Gaps")
            stats_section = ""
            if self.profile.has_market_stats:
                sentiment_quality = ""
                if sentiment:
                    sentiment_quality = (
                        "\n- Follow-through quality: failed-board ratio {broken}; previous-limit premium median {premium}; "
                        "highest board {height}; one-price heuristic {one_price}."
                    ).format(
                        broken=self._format_prompt_ratio(sentiment.get("broken_ratio")),
                        premium=self._format_prompt_pct(sentiment.get("previous_limit_premium_median_pct")),
                        height=self._format_prompt_metric(sentiment.get("highest_consecutive_board")),
                        one_price=self._format_prompt_ratio(sentiment.get("one_price_like_ratio")),
                    )
                stats_section = f"""
### 4. Sentiment Temperature
| Metric | Value |
|--------|-------|
| Advancers | {overview.up_count} |
| Decliners | {overview.down_count} |
| Limit-up | {overview.limit_up_count} |
| Limit-down | {overview.limit_down_count} |
| Turnover ({self._get_turnover_unit_label()}) | {overview.total_amount:.0f} |
{sentiment_quality or "- Follow-through quality inputs are unavailable; treat the table as market heat only."}
"""
            sector_section = ""
            if self.profile.has_sector_rankings and (top_text or bottom_text or top_concept_text or bottom_concept_text):
                ranked_theme_lines = "\n".join(
                    f"- {self._prompt_cell(item.get('theme'))}: {self._prompt_cell(item.get('classification'))}, "
                    f"score {self._format_prompt_metric(item.get('total_score'))}; "
                    f"confirmation: {self._prompt_cell(item.get('confirmation'))}; "
                    f"invalidation: {self._prompt_cell(item.get('invalidation'))}."
                    for item in theme_candidates
                )
                sector_section = f"""
### 5. Theme Ranking
- **Industry Leaders**: {top_text or "N/A"}
- **Industry Laggards**: {bottom_text or "N/A"}
- **Concept Leaders**: {top_concept_text or "N/A"}
- **Concept Laggards**: {bottom_concept_text or "N/A"}
{ranked_theme_lines or "- Ranked limit-up diffusion evidence is unavailable; keep these as ranking-only candidates."}
"""
            watchlist_lines = "\n".join(
                f"- {self._prompt_cell(item.get('category'))} {self._prompt_cell(item.get('code'))} "
                f"{self._prompt_cell(item.get('name'))}: {self._prompt_cell(item.get('buy_point_type'))}; "
                f"confirmation: {self._prompt_cell(item.get('validation'))}; "
                f"invalidation: {self._prompt_cell(item.get('invalidation'))}."
                for item in stock_candidates
            )
            market_names = {
                "us": "US Market Recap",
                "hk": "HK Market Recap",
                "jp": "Japan Market Recap",
                "kr": "Korea Market Recap",
            }
            market_name = market_names.get(self.region, "A-share Market Recap")
            report = f"""## {overview.date} {market_name}

{data_scope_section}

{data_gap_section}

### 2. Market Summary
Today's {self._get_market_scope_name(template_language)} showed **{market_mood}**.

### 3. Index Risk Gates
{indices_text or "- No index data available"}
{stats_section}
{sector_section}

### 6. Core Watchlist
{watchlist_lines or "- No verifiable stock-level watchlist is available without leader/capacity/trend stock inputs."}

### 7. Next-Session Strategy
- Avoid chasing unconfirmed one-day movers.
- Raise exposure only after index confirmation and theme continuation are both visible.
- Use trial positions first when leadership evidence is incomplete.

### 8. Risk Alerts
Market conditions can change quickly. The data above is for reference only and does not constitute investment advice.

---
*Review Time: {datetime.now().strftime('%H:%M')}*
"""
            return report

        market_labels = {"cn": "A股", "us": "美股", "hk": "港股", "jp": "日股", "kr": "韩股"}
        market_label = market_labels.get(self.region, "A股")
        data_scope_section = self._build_data_scope_report_block(overview)
        data_gap_section = self._build_data_gap_input_block(overview).replace("## 数据缺口", "### 数据缺口")
        dashboard_block = self._build_stats_block(overview) if self.profile.has_market_stats else ""
        indices_block = self._build_indices_block(overview)
        sector_block = self._build_sector_block(overview) if self.profile.has_sector_rankings else ""
        summary_focus = (
            "指数承接、成交额变化和板块持续性"
            if self.profile.has_market_stats and self.profile.has_sector_rankings
            else "指数承接、消息催化和整体风险状态"
        )
        sector_section = (
            f"""
### 五、热点排序
{sector_block or "- 暂无板块涨跌榜数据。"}
"""
            if self.profile.has_sector_rankings
            else ""
        )
        funds_section = (
            f"""
### 四、情绪温度
{dashboard_block or "- 暂无市场宽度数据。"}

- 结合成交额和涨跌家数看，当前更适合等待确认，避免仅凭单一热点追高。
"""
            if self.profile.has_market_stats
            else ""
        )
        sentiment_quality = ""
        if sentiment:
            sentiment_quality = (
                "- 接力质量：炸板率 {broken}；昨日涨停溢价中位数 {premium}；最高 {height} 板；"
                "疑似一字/无换手占比 {one_price}。"
            ).format(
                broken=self._format_prompt_ratio(sentiment.get("broken_ratio")),
                premium=self._format_prompt_pct(sentiment.get("previous_limit_premium_median_pct")),
                height=self._format_prompt_metric(sentiment.get("highest_consecutive_board")),
                one_price=self._format_prompt_ratio(sentiment.get("one_price_like_ratio")),
            )
        ranked_theme_lines = "\n".join(
            f"- {self._prompt_cell(item.get('theme'))}：{self._prompt_cell(item.get('classification'))}，"
            f"评分 {self._format_prompt_metric(item.get('total_score'))}；"
            f"确认={self._prompt_cell(item.get('confirmation'))}；"
            f"失效={self._prompt_cell(item.get('invalidation'))}。"
            for item in theme_candidates
        )
        watchlist_lines = "\n".join(
            f"- {self._prompt_cell(item.get('category'))} {self._prompt_cell(item.get('code'))} "
            f"{self._prompt_cell(item.get('name'))}：{self._prompt_cell(item.get('buy_point_type'))}；"
            f"验证={self._prompt_cell(item.get('validation'))}；"
            f"失效={self._prompt_cell(item.get('invalidation'))}。"
            for item in stock_candidates
        )
        return f"""## {overview.date} 大盘复盘

> 今日{market_label}市场整体呈现**{market_mood}**态势，优先观察{summary_focus}。

{data_scope_section}

{data_gap_section}

### 二、盘面总览
- 当前复盘先按大盘快照处理，只有指数与热点持续性同时确认后，才升级为可执行进攻策略。

### 三、大盘风险门槛
{indices_block or indices_text or "暂无指数数据。"}

{funds_section}
{sentiment_quality}
{sector_section}
{ranked_theme_lines}

### 六、消息催化
- 暂无可用新闻时，应降低对题材持续性的确定性判断。

### 七、核心观察票
{watchlist_lines or "- 暂无可验证观察票：当前未提供情绪龙头、容量核心、前排和低位补涨清单。"}

### 八、明日交易计划
- 不能买：单日冲高、无板块支撑、缺少成交确认的后排跟风。
- 确认条件：指数不转弱，候选热点有前排继续封板或容量核心承接。
- 买点类型：确认后的回踩承接、放量突破或分歧转一致，不做无确认追高。
- 失效位：指数转弱、上涨家数收缩、跌停扩散或热点前排断板负反馈。
- 仓位上限：主线未确认前以观察/试错仓为主，确认后再逐步加仓。

### 九、风险提示
- 市场有风险，投资需谨慎。以上数据仅供参考，不构成投资建议。

---
*复盘时间: {datetime.now().strftime('%H:%M')}*
"""
    
    def _run_daily_review_parts(self) -> MarketLightReviewResult:
        """Run market review once and keep report/snapshot on the same overview."""
        logger.info("========== 开始大盘复盘分析 ==========")

        # 1. 获取市场概览
        overview = self.get_market_overview()

        # 2. 搜索市场新闻（传入概览以启用板块事件动态检索词）
        news = self.search_market_news(overview)
        news = self._merge_persisted_market_intelligence(news)

        # 3. 生成复盘报告
        report = self.generate_market_review(overview, news)
        snapshot = self.build_market_light_snapshot(overview) if self._supports_market_light() else None
        structured_payload = self.build_market_review_payload(
            overview,
            news,
            report,
            snapshot,
        )

        logger.info("========== 大盘复盘分析完成 ==========")

        return MarketLightReviewResult(
            overview=overview,
            report=report,
            market_light_snapshot=snapshot,
            structured_payload=structured_payload,
        )

    def _merge_persisted_market_intelligence(self, news: List) -> List:
        """Merge local persisted market intelligence and search news with bounded prompt/payload slot preservation."""
        search_news = list(news or [])
        merged_local = []
        seen_urls = {
            self._get_news_field(item, "url")
            for item in search_news
            if self._get_news_field(item, "url")
        }
        try:
            service = IntelligenceService(config=self.config)
            service.refresh_auto_sources()
            payload = service.list_items(
                scope_type="market",
                market=self.region,
                published_days=max(1, int(self.config.get_effective_news_window_days() or 1)),
                page=1,
                page_size=6,
            )
            for item in payload.get("items", []):
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url") or "")
                if url and url in seen_urls:
                    continue
                seen_urls.add(url)
                merged_local.append({
                    "title": item.get("title") or "未命名资讯",
                    "snippet": item.get("summary") or "",
                    "source": item.get("source") or item.get("source_name") or "local-intel",
                    "published_date": item.get("published_at") or "",
                    "url": "" if url.startswith("no-url:intel:") else url,
                })
        except Exception as exc:
            logger.debug("[大盘] %s action=load_local_intelligence status=failed error=%s", self._log_context(), exc)
        merged_news = []
        merged_local_index = 0
        merged_search_index = 0
        while merged_local_index < len(merged_local) or merged_search_index < len(search_news):
            if merged_local_index < len(merged_local):
                merged_news.append(merged_local[merged_local_index])
                merged_local_index += 1
            if merged_search_index < len(search_news):
                merged_news.append(search_news[merged_search_index])
                merged_search_index += 1
        return merged_news

    def run_daily_review(self) -> str:
        """
        执行每日大盘复盘流程

        Returns:
            复盘报告文本
        """
        return self.run_daily_review_with_snapshot().report

    def run_daily_review_with_snapshot(self) -> MarketLightReviewResult:
        """Run daily review and return the report plus its structured Market Light snapshot."""
        return self._run_daily_review_parts()


# 测试入口
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    )
    
    analyzer = MarketAnalyzer()
    
    # 测试获取市场概览
    overview = analyzer.get_market_overview()
    print(f"\n=== 市场概览 ===")
    print(f"日期: {overview.date}")
    print(f"指数数量: {len(overview.indices)}")
    for idx in overview.indices:
        print(f"  {idx.name}: {idx.current:.2f} ({idx.change_pct:+.2f}%)")
    print(f"上涨: {overview.up_count} | 下跌: {overview.down_count}")
    print(f"成交额: {overview.total_amount:.0f}亿")
    
    # 测试生成模板报告
    report = analyzer._generate_template_review(overview, [])
    print(f"\n=== 复盘报告 ===")
    print(report)
