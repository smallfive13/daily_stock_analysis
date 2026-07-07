# -*- coding: utf-8 -*-
"""Tests for A-share market review P2 context injection."""

import unittest
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from data_provider.akshare_fetcher import AkshareFetcher
from data_provider.base import DataFetcherManager
from data_provider.fundamental_adapter import AkshareFundamentalAdapter
from src.core.market_profile import US_PROFILE
from src.core.market_strategy import get_market_strategy_blueprint
from src.market_analyzer import (
    MarketAnalyzer,
    MarketIndex,
    MarketOverview,
    aggregate_limit_up_pool,
    compute_index_key_levels,
)


def _build_analyzer(region: str = "cn", language: str = "zh") -> MarketAnalyzer:
    with patch("src.market_analyzer.DataFetcherManager"):
        analyzer = MarketAnalyzer(region=region, config=SimpleNamespace(report_language=language))
    analyzer.data_manager = MagicMock()
    return analyzer


def _cn_overview() -> MarketOverview:
    return MarketOverview(
        date="2026-07-06",
        indices=[
            MarketIndex(code="sh000001", name="上证指数", current=3200.0, change_pct=0.6),
            MarketIndex(code="sz399006", name="创业板指", current=2100.0, change_pct=-0.2),
            MarketIndex(code="sh000688", name="科创50", current=980.0, change_pct=1.2),
        ],
        up_count=3200,
        down_count=1800,
        flat_count=100,
        limit_up_count=68,
        limit_down_count=8,
        total_amount=10500.0,
        top_sectors=[{"name": "半导体", "change_pct": 4.5}],
        bottom_sectors=[{"name": "煤炭", "change_pct": -2.1}],
        top_concepts=[{"name": "机器人", "change_pct": 3.2}],
        bottom_concepts=[{"name": "转基因", "change_pct": -1.8}],
    )


def _bars(count: int = 20) -> list[dict]:
    return [
        {
            "date": f"2026-06-{day:02d}",
            "open": 100.0 + day,
            "high": 101.0 + day,
            "low": 99.0 + day,
            "close": 100.0 + day,
        }
        for day in range(1, count + 1)
    ]


class LimitUpAggregationTestCase(unittest.TestCase):
    def test_aggregate_limit_up_pool_normal(self) -> None:
        result = aggregate_limit_up_pool([
            {"industry": "半导体", "consecutive_boards": 2, "break_count": 1, "name": "芯片A"},
            {"industry": "机器人", "consecutive_boards": 4, "break_count": 3, "name": "机器人B"},
            {"industry": "半导体", "consecutive_boards": 1, "break_count": 0, "name": "芯片C"},
        ])

        self.assertEqual(result["total"], 3)
        self.assertEqual(result["industry_distribution"][0], {"industry": "半导体", "count": 2})
        self.assertEqual(result["max_consecutive_boards"], 4)
        self.assertEqual(result["max_boards_stock"], "机器人B")
        self.assertEqual(result["total_break_count"], 4)

    def test_aggregate_limit_up_pool_skips_missing_industry_and_break_count(self) -> None:
        result = aggregate_limit_up_pool([
            {"industry": "", "consecutive_boards": 3, "name": "无行业A"},
            {"consecutive_boards": 1, "break_count": None, "name": "无行业B"},
        ])

        self.assertEqual(result["total"], 2)
        self.assertEqual(result["industry_distribution"], [])
        self.assertEqual(result["max_consecutive_boards"], 3)
        self.assertEqual(result["max_boards_stock"], "无行业A")
        self.assertEqual(result["total_break_count"], 0)

    def test_aggregate_limit_up_pool_empty(self) -> None:
        self.assertEqual(aggregate_limit_up_pool([]), {})


class IndexKeyLevelsTestCase(unittest.TestCase):
    def test_compute_index_key_levels_normal(self) -> None:
        result = compute_index_key_levels(_bars(20))

        self.assertEqual(result["ma20"], 110.5)
        self.assertEqual(result["high_20d"], 121.0)
        self.assertEqual(result["low_20d"], 100.0)

    def test_compute_index_key_levels_requires_20_bars(self) -> None:
        self.assertEqual(compute_index_key_levels(_bars(19)), {})


class P2PromptContextTestCase(unittest.TestCase):
    def test_prompt_renders_three_new_cn_blocks_with_data(self) -> None:
        analyzer = _build_analyzer("cn")
        overview = _cn_overview()
        overview.fund_inflow_sectors = [{"name": "半导体", "net_inflow": 12.34}]
        overview.fund_outflow_sectors = [{"name": "煤炭", "net_inflow": -8.9}]
        overview.limit_up_structure = {
            "total": 68,
            "industry_distribution": [{"industry": "机器人", "count": 5}, {"industry": "半导体", "count": 3}],
            "max_consecutive_boards": 4,
            "max_boards_stock": "机器人B",
            "total_break_count": 9,
        }
        overview.index_key_levels = [
            {"name": "上证指数", "current": 3200.0, "ma20": 3150.5, "high_20d": 3300.0, "low_20d": 3000.0},
        ]

        prompt = analyzer._build_review_prompt(overview, [])

        self.assertIn("资金净流入板块: 半导体(+12.34)", prompt)
        self.assertIn("资金净流出板块: 煤炭(-8.90)", prompt)
        self.assertIn("## 涨停结构", prompt)
        self.assertIn("涨停池数量: 68 家 | 最高连板: 4 板（机器人B） | 炸板次数合计: 9", prompt)
        self.assertIn("涨停行业分布: 机器人(5家)、半导体(3家)", prompt)
        self.assertIn("## 指数关键位参考（基于日线本地计算，非预测）", prompt)
        self.assertIn("| 上证指数 | 3200.00 | 3150.50 | 3300.00 | 3000.00 |", prompt)
        self.assertIn("触发失效条件必须引用已提供的可观察锚点", prompt)

    def test_prompt_omits_new_blocks_when_data_missing(self) -> None:
        analyzer = _build_analyzer("cn")
        prompt = analyzer._build_review_prompt(_cn_overview(), [])

        self.assertNotIn("资金净流入板块", prompt)
        self.assertNotIn("资金净流出板块", prompt)
        self.assertNotIn("## 涨停结构", prompt)
        self.assertNotIn("## 指数关键位参考", prompt)

    def test_us_prompt_contains_no_new_cn_blocks(self) -> None:
        analyzer = _build_analyzer("us", language="en")
        prompt = analyzer._build_review_prompt(MarketOverview(date="2026-07-06"), [])

        self.assertNotIn("资金净流入板块", prompt)
        self.assertNotIn("Fund inflow leaders", prompt)
        self.assertNotIn("## 涨停结构", prompt)
        self.assertNotIn("## Limit-up Structure", prompt)
        self.assertNotIn("## 指数关键位参考", prompt)
        self.assertNotIn("## Index Key Levels Reference", prompt)

    def test_english_cn_prompt_renders_new_blocks(self) -> None:
        analyzer = _build_analyzer("cn", language="en")
        overview = _cn_overview()
        overview.fund_inflow_sectors = [{"name": "Semis", "net_inflow": 3.21}]
        overview.limit_up_structure = {
            "total": 2,
            "industry_distribution": [{"industry": "Robotics", "count": 2}],
            "max_consecutive_boards": 3,
            "max_boards_stock": "RobotA",
            "total_break_count": 1,
        }
        overview.index_key_levels = [
            {"name": "上证指数", "current": 3200.0, "ma20": 3150.5, "high_20d": 3300.0, "low_20d": 3000.0},
        ]

        prompt = analyzer._build_review_prompt(overview, [])

        self.assertIn("Fund inflow leaders: Semis(+3.21)", prompt)
        self.assertIn("## Limit-up Structure", prompt)
        self.assertIn("## Index Key Levels Reference", prompt)
        self.assertIn("anchored to provided observable data", prompt)


class P2MarketOverviewFetchTestCase(unittest.TestCase):
    def test_get_market_overview_populates_p2_context_for_cn(self) -> None:
        analyzer = _build_analyzer("cn")
        analyzer.data_manager.get_main_indices.return_value = [
            {"code": "sh000001", "name": "上证指数", "current": 3200.0, "change": 1.0, "change_pct": 0.6,
             "open": 3180.0, "high": 3210.0, "low": 3170.0, "prev_close": 3199.0, "volume": 1.0, "amount": 2.0, "amplitude": 1.0},
        ]
        analyzer.data_manager.get_market_stats.return_value = {"limit_up_count": 2}
        analyzer.data_manager.get_sector_rankings.return_value = ([], [])
        analyzer.data_manager.get_concept_rankings.return_value = ([], [])
        analyzer.data_manager.get_sector_fund_flow_rankings.return_value = ([{"name": "半导体", "net_inflow": 10.0}], [])
        analyzer.data_manager.get_limit_up_pool.return_value = [
            {"industry": "半导体", "consecutive_boards": 2, "break_count": 1, "name": "芯片A"},
        ]
        analyzer.data_manager.get_global_market_indices.return_value = []
        analyzer.data_manager.get_index_daily_history.return_value = _bars(20)

        overview = analyzer.get_market_overview()

        self.assertEqual(overview.fund_inflow_sectors[0]["name"], "半导体")
        self.assertEqual(overview.limit_up_structure["total"], 1)
        self.assertEqual(overview.index_key_levels[0]["name"], "上证指数")

    def test_get_market_overview_survives_new_source_failures(self) -> None:
        analyzer = _build_analyzer("cn")
        analyzer.data_manager.get_main_indices.return_value = []
        analyzer.data_manager.get_market_stats.return_value = {}
        analyzer.data_manager.get_sector_rankings.return_value = ([], [])
        analyzer.data_manager.get_concept_rankings.return_value = ([], [])
        analyzer.data_manager.get_global_market_indices.return_value = []
        analyzer.data_manager.get_sector_fund_flow_rankings.side_effect = RuntimeError("flow boom")
        analyzer.data_manager.get_limit_up_pool.side_effect = RuntimeError("limit boom")
        analyzer.data_manager.get_index_daily_history.side_effect = RuntimeError("index boom")

        overview = analyzer.get_market_overview()

        self.assertEqual(overview.fund_inflow_sectors, [])
        self.assertEqual(overview.fund_outflow_sectors, [])
        self.assertEqual(overview.limit_up_structure, {})
        self.assertEqual(overview.index_key_levels, [])

    def test_non_cn_overview_does_not_call_new_cn_sources(self) -> None:
        analyzer = _build_analyzer("us", language="en")
        analyzer.profile = US_PROFILE
        analyzer.strategy = get_market_strategy_blueprint("us")
        analyzer.data_manager.get_main_indices.return_value = []

        overview = analyzer.get_market_overview()

        self.assertEqual(overview.fund_inflow_sectors, [])
        analyzer.data_manager.get_sector_fund_flow_rankings.assert_not_called()
        analyzer.data_manager.get_limit_up_pool.assert_not_called()
        analyzer.data_manager.get_index_daily_history.assert_not_called()


class SectorFundFlowAdapterTestCase(unittest.TestCase):
    def test_get_sector_fund_flow_rankings_extracts_top_and_bottom(self) -> None:
        adapter = AkshareFundamentalAdapter()
        df = pd.DataFrame(
            {
                "行业名称": ["半导体", "机器人", "煤炭"],
                "主力净流入": [12.5, 3.2, -8.1],
            }
        )

        with patch.object(adapter, "_call_df_candidates", return_value=(df, "stock_sector_fund_flow_rank", [])):
            result = adapter.get_sector_fund_flow_rankings(2)

        self.assertEqual(result["status"], "partial")
        self.assertEqual([item["name"] for item in result["top"]], ["半导体", "机器人"])
        self.assertEqual([item["name"] for item in result["bottom"]], ["煤炭", "机器人"])
        self.assertEqual(result["source_chain"], ["stock_sector_fund_flow_rank"])

    def test_get_capital_flow_keeps_sector_rankings_shape_after_refactor(self) -> None:
        adapter = AkshareFundamentalAdapter()
        stock_df = pd.DataFrame(
            {
                "股票代码": ["600519"],
                "主力净流入": [1.5],
                "5日": [2.5],
                "10日": [3.5],
            }
        )
        sector_df = pd.DataFrame(
            {
                "板块名称": ["半导体", "煤炭"],
                "净流入": [10.0, -5.0],
            }
        )

        with patch.object(
            adapter,
            "_call_df_candidates",
            side_effect=[
                (stock_df, "stock_individual_fund_flow", []),
                (sector_df, "stock_sector_fund_flow_rank", []),
            ],
        ):
            result = adapter.get_capital_flow("600519", top_n=1)

        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["stock_flow"]["main_net_inflow"], 1.5)
        self.assertEqual(result["sector_rankings"]["top"], [{"name": "半导体", "net_inflow": 10.0}])
        self.assertEqual(result["sector_rankings"]["bottom"], [{"name": "煤炭", "net_inflow": -5.0}])
        self.assertIn("capital_sector:stock_sector_fund_flow_rank", result["source_chain"])


class DataManagerP2ContextTestCase(unittest.TestCase):
    def test_get_sector_fund_flow_rankings_fail_open(self) -> None:
        manager = DataFetcherManager.__new__(DataFetcherManager)
        manager._fundamental_adapter = MagicMock()
        manager._fundamental_adapter.get_sector_fund_flow_rankings.side_effect = RuntimeError("boom")

        self.assertEqual(manager.get_sector_fund_flow_rankings(5), ([], []))

    def test_get_index_daily_history_iterates_fetchers_fail_open(self) -> None:
        class FailingFetcher:
            name = "FailingFetcher"
            priority = 1

            def get_index_daily_history(self, symbol: str, days: int = 30) -> list[dict]:
                raise RuntimeError("boom")

        class WorkingFetcher:
            name = "WorkingFetcher"
            priority = 2

            def get_index_daily_history(self, symbol: str, days: int = 30) -> list[dict]:
                return [{"date": "2026-07-01", "close": 1.0}]

        manager = DataFetcherManager(fetchers=[FailingFetcher(), WorkingFetcher()])

        self.assertEqual(manager.get_index_daily_history("000001"), [{"date": "2026-07-01", "close": 1.0}])


class AkshareIndexDailyHistoryTestCase(unittest.TestCase):
    def test_get_index_daily_history_normalizes_rows(self) -> None:
        df = pd.DataFrame(
            {
                "日期": ["2026-06-01", "2026-06-02"],
                "开盘": [3100, 3110],
                "最高": [3120, 3130],
                "最低": [3090, 3100],
                "收盘": [3115, 3125],
            }
        )
        fake_akshare = SimpleNamespace(index_zh_a_hist=lambda **kwargs: df)
        original_akshare = sys.modules.get("akshare")
        sys.modules["akshare"] = fake_akshare
        try:
            fetcher = AkshareFetcher(sleep_min=0, sleep_max=0)
            fetcher._set_random_user_agent = MagicMock()
            fetcher._enforce_rate_limit = MagicMock()

            result = fetcher.get_index_daily_history("000001", days=2)
        finally:
            if original_akshare is None:
                sys.modules.pop("akshare", None)
            else:
                sys.modules["akshare"] = original_akshare

        self.assertEqual(
            result,
            [
                {"date": "2026-06-01", "open": 3100.0, "high": 3120.0, "low": 3090.0, "close": 3115.0},
                {"date": "2026-06-02", "open": 3110.0, "high": 3130.0, "low": 3100.0, "close": 3125.0},
            ],
        )


if __name__ == "__main__":
    unittest.main()
