# -*- coding: utf-8 -*-
"""Tests for A-share market review global context, dynamic news queries, and search diagnostics."""

import unittest
from unittest.mock import MagicMock, patch

from src.core.market_profile import get_profile
from src.market_analyzer import MarketAnalyzer, MarketIndex, MarketOverview


def _build_analyzer(region: str = "cn") -> MarketAnalyzer:
    with patch("src.market_analyzer.DataFetcherManager"):
        return MarketAnalyzer(region=region)


def _overview_with_sectors() -> MarketOverview:
    return MarketOverview(
        date="2026-07-03",
        top_sectors=[{"name": "装卸搬运", "change_pct": 6.59}, {"name": "PCB", "change_pct": 4.2}],
        bottom_sectors=[{"name": "半导体", "change_pct": -5.8}],
        top_concepts=[{"name": "机器人", "change_pct": 5.1}],
    )


def _global_index(name: str, change_pct: float) -> MarketIndex:
    return MarketIndex(code="X", name=name, current=1000.0, change_pct=change_pct)


class MarketProfileGlobalContextTestCase(unittest.TestCase):
    def test_only_cn_profile_enables_global_context(self) -> None:
        self.assertTrue(get_profile("cn").has_global_context)
        for region in ("us", "hk", "jp", "kr"):
            self.assertFalse(get_profile(region).has_global_context, region)

    def test_cn_queries_include_overnight_us_market(self) -> None:
        self.assertTrue(
            any("隔夜" in q or "美股" in q for q in get_profile("cn").news_queries)
        )


class GlobalContextPromptTestCase(unittest.TestCase):
    def test_prompt_renders_global_block_with_data(self) -> None:
        analyzer = _build_analyzer("cn")
        overview = _overview_with_sectors()
        overview.global_indices = [
            _global_index("费城半导体指数", -6.27),
            _global_index("韩国KOSPI", -2.1),
        ]

        prompt = analyzer._build_review_prompt(overview, [])

        self.assertIn("外围市场联动", prompt)
        self.assertIn("费城半导体指数", prompt)
        self.assertIn("↓6.27%", prompt)
        self.assertIn("必须先判断当日 A 股走势是否由外围事件驱动", prompt)
        self.assertIn("禁止编造其他外围数据", prompt)

    def test_prompt_marks_global_data_missing_for_cn(self) -> None:
        analyzer = _build_analyzer("cn")
        prompt = analyzer._build_review_prompt(_overview_with_sectors(), [])

        self.assertIn("外围市场联动", prompt)
        self.assertIn("外围市场数据本次未获取", prompt)
        self.assertIn("禁止编造隔夜美股或亚太市场表现", prompt)

    def test_prompt_skips_global_block_for_us_region(self) -> None:
        analyzer = _build_analyzer("us")
        prompt = analyzer._build_review_prompt(MarketOverview(date="2026-07-03"), [])

        self.assertNotIn("外围市场联动", prompt)
        self.assertNotIn("Global Market Context", prompt)

    def test_get_market_overview_populates_global_indices_for_cn(self) -> None:
        analyzer = _build_analyzer("cn")
        analyzer.data_manager.get_main_indices.return_value = []
        analyzer.data_manager.get_market_stats.return_value = {}
        analyzer.data_manager.get_sector_rankings.return_value = ([], [])
        analyzer.data_manager.get_concept_rankings.return_value = ([], [])
        analyzer.data_manager.get_global_market_indices.return_value = [
            {"code": "SOX", "name": "费城半导体指数", "current": 4500.0, "change_pct": -6.27},
        ]

        overview = analyzer.get_market_overview()

        self.assertEqual(len(overview.global_indices), 1)
        self.assertEqual(overview.global_indices[0].name, "费城半导体指数")
        self.assertAlmostEqual(overview.global_indices[0].change_pct, -6.27, places=2)

    def test_get_market_overview_survives_global_fetch_failure(self) -> None:
        analyzer = _build_analyzer("cn")
        analyzer.data_manager.get_main_indices.return_value = []
        analyzer.data_manager.get_market_stats.return_value = {}
        analyzer.data_manager.get_sector_rankings.return_value = ([], [])
        analyzer.data_manager.get_concept_rankings.return_value = ([], [])
        analyzer.data_manager.get_global_market_indices.side_effect = RuntimeError("boom")

        overview = analyzer.get_market_overview()

        self.assertEqual(overview.global_indices, [])


class DynamicNewsQueryTestCase(unittest.TestCase):
    def test_dynamic_queries_built_from_sector_moves(self) -> None:
        analyzer = _build_analyzer("cn")
        queries = analyzer._build_dynamic_news_queries(_overview_with_sectors())

        self.assertEqual(
            queries,
            ["装卸搬运 板块 大涨 原因", "半导体 板块 大跌 原因", "机器人 概念 上涨 消息"],
        )

    def test_dynamic_queries_empty_without_overview_or_for_non_cn(self) -> None:
        analyzer = _build_analyzer("cn")
        self.assertEqual(analyzer._build_dynamic_news_queries(None), [])

        us_analyzer = _build_analyzer("us")
        self.assertEqual(us_analyzer._build_dynamic_news_queries(_overview_with_sectors()), [])

    def test_search_uses_dynamic_queries_and_dedupes_urls(self) -> None:
        analyzer = _build_analyzer("cn")
        result_a = MagicMock()
        result_a.url = "https://example.com/a"
        response = MagicMock()
        response.results = [result_a]
        analyzer.search_service = MagicMock()
        analyzer.search_service.search_stock_news.return_value = response

        news = analyzer.search_market_news(_overview_with_sectors())

        # 固定 3 条 + 动态 3 条 = 6 次检索；URL 相同的结果只保留一条
        self.assertEqual(analyzer.search_service.search_stock_news.call_count, 6)
        self.assertEqual(len(news), 1)
        self.assertEqual(analyzer._news_search_status, "ok")

        called_keywords = [
            " ".join(call.kwargs["focus_keywords"])
            for call in analyzer.search_service.search_stock_news.call_args_list
        ]
        self.assertIn("装卸搬运 板块 大涨 原因", called_keywords)
        self.assertIn("半导体 板块 大跌 原因", called_keywords)


class NewsSearchDiagnosticsTestCase(unittest.TestCase):
    def test_no_search_service_status_and_placeholder(self) -> None:
        analyzer = _build_analyzer("cn")
        analyzer.search_service = None

        news = analyzer.search_market_news(_overview_with_sectors())

        self.assertEqual(news, [])
        self.assertEqual(analyzer._news_search_status, "no_search_service")

        prompt = analyzer._build_review_prompt(_overview_with_sectors(), [])
        self.assertIn("新闻检索服务未配置", prompt)
        self.assertIn("不代表市场无重大消息", prompt)
        self.assertNotIn("暂无相关新闻", prompt)

    def test_no_results_status_and_placeholder(self) -> None:
        analyzer = _build_analyzer("cn")
        empty_response = MagicMock()
        empty_response.results = []
        analyzer.search_service = MagicMock()
        analyzer.search_service.search_stock_news.return_value = empty_response

        news = analyzer.search_market_news(_overview_with_sectors())

        self.assertEqual(news, [])
        self.assertEqual(analyzer._news_search_status, "no_results")

        prompt = analyzer._build_review_prompt(_overview_with_sectors(), [])
        self.assertIn("已执行新闻检索但未获取到有效结果", prompt)
        self.assertIn("消息面检索无结果", prompt)

    def test_unknown_status_keeps_legacy_placeholder(self) -> None:
        analyzer = _build_analyzer("cn")
        prompt = analyzer._build_review_prompt(_overview_with_sectors(), [])
        self.assertIn("暂无相关新闻", prompt)


class SectionHintTestCase(unittest.TestCase):
    def test_cn_template_requires_linkage_and_observable_triggers(self) -> None:
        analyzer = _build_analyzer("cn")
        sections = analyzer._build_output_template_sections("zh")

        self.assertIn("联动关系", sections)
        self.assertIn("谁在打谁、资金从哪来到哪去", sections)
        self.assertIn("先判断当日走势是否由外围事件驱动", sections)
        self.assertIn("资金净流入/流出榜与涨跌幅榜交叉验证", sections)
        self.assertIn("结合涨停结构（连板高度、炸板情况）", sections)
        self.assertIn("触发失效条件必须引用已提供的可观察锚点", sections)
        self.assertIn("禁止使用\"若市场走弱\"这类不可验证表述", sections)


if __name__ == "__main__":
    unittest.main()
