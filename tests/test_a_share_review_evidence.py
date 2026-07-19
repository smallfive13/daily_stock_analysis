# -*- coding: utf-8 -*-
"""Regression tests for the integrated A-share daily-review evidence contract."""

from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from data_provider.akshare_fetcher import AkshareFetcher
from data_provider.base import BaseFetcher, DataFetcherManager
from src.market_analyzer import MarketAnalyzer, MarketIndex, MarketOverview
from src.services.a_share_review_evidence import AShareReviewEvidenceService


def _bars(base: float = 100.0) -> list[dict]:
    return [
        {
            "date": f"2026-06-{day:02d}",
            "open": base + day,
            "high": base + day + 1,
            "low": base + day - 1,
            "close": base + day,
        }
        for day in range(1, 21)
    ]


class _EvidenceManager:
    def __init__(self) -> None:
        self.pools = {
            "limit_up": [
                {
                    "code": "600001",
                    "name": "芯片龙头",
                    "industry": "半导体",
                    "change_pct": 10.0,
                    "amount": 8_000_000_000,
                    "turnover_rate": 12.0,
                    "consecutive_boards": 3,
                    "first_limit_time": "092500",
                    "break_count": 0,
                    "limit_stat": "3/3",
                },
                {
                    "code": "600002",
                    "name": "芯片容量",
                    "industry": "半导体",
                    "change_pct": 9.8,
                    "amount": 12_000_000_000,
                    "turnover_rate": 8.0,
                    "consecutive_boards": 1,
                    "first_limit_time": "100100",
                    "break_count": 1,
                },
                {
                    "code": "300001",
                    "name": "机器人首板",
                    "industry": "机器人",
                    "change_pct": 20.0,
                    "amount": 2_000_000_000,
                    "turnover_rate": 18.0,
                    "consecutive_boards": 1,
                    "first_limit_time": "101500",
                    "break_count": 0,
                },
            ],
            "broken_board": [{"code": "600010"}, {"code": "600011"}],
            "previous_limit": [{"change_pct": -1.2}, {"change_pct": -0.4}, {"change_pct": 0.2}],
            "limit_down": [{"code": "600099"}],
        }

    def get_limit_event_pool_with_meta(self, pool_type: str, date: str, n: int):
        rows = self.pools[pool_type]
        return rows, [{"provider": "Fake", "status": "ok" if rows else "empty"}], ""

    def get_index_daily_history(self, symbol: str, days: int = 30):
        return _bars(100.0 if symbol == "000001" else 200.0)


class AShareReviewEvidenceServiceTestCase(unittest.TestCase):
    def test_builds_sentiment_themes_watchlist_and_risk_gates(self) -> None:
        manager = _EvidenceManager()
        service = AShareReviewEvidenceService(manager)

        evidence = service.build(
            trade_date="20260717",
            indices=[
                {"code": "sh000001", "name": "上证指数", "current": 121.0},
                {"code": "sz399006", "name": "创业板指", "current": 205.0},
                {"code": "sh000688", "name": "科创50", "current": 204.0},
            ],
            sector_rankings={
                "top": [{"name": "半导体", "change_pct": 4.2}],
                "bottom": [{"name": "煤炭", "change_pct": -1.1}],
            },
            concept_rankings={
                "top": [{"name": "机器人", "change_pct": 3.5}],
                "bottom": [{"name": "转基因", "change_pct": -0.8}],
            },
            market_snapshot={"up_count": 3000, "down_count": 1800, "flat_count": 100},
        )

        sentiment = evidence["sentiment_structure"]
        self.assertEqual(sentiment["limit_up_count"], 3)
        self.assertEqual(sentiment["broken_board_count"], 2)
        self.assertEqual(sentiment["broken_ratio"], 0.4)
        self.assertEqual(sentiment["previous_limit_premium_median_pct"], -0.4)
        self.assertEqual(sentiment["highest_consecutive_board"], 3)
        self.assertEqual(sentiment["one_price_like_count"], 1)
        self.assertEqual(evidence["index_trend"][0]["ma5"], 118.0)
        self.assertEqual(evidence["index_trend"][0]["ma10"], 115.5)
        self.assertTrue(any(item["theme"] == "半导体" for item in evidence["theme_candidates"]))
        self.assertTrue(any(item["category"] == "emotion_leader" for item in evidence["stock_candidates"]))
        self.assertTrue(any(item["category"] == "capacity_core" for item in evidence["stock_candidates"]))
        self.assertIn("high_broken_board_ratio", evidence["risk_tags"])
        self.assertIn("negative_previous_limit_premium", evidence["risk_tags"])
        self.assertIn("growth_index_below_short_ma", evidence["risk_tags"])
        self.assertEqual(evidence["data_quality"]["status"], "ok")

    def test_marks_missing_sources_without_aborting(self) -> None:
        manager = MagicMock()
        manager.get_limit_event_pool_with_meta.return_value = ([], [], "all sources failed")
        manager.get_index_daily_history.return_value = []
        service = AShareReviewEvidenceService(manager)

        evidence = service.build(
            trade_date="20260717",
            indices=[{"code": "sh000001", "name": "上证指数", "current": 3200.0}],
            sector_rankings={"top": [], "bottom": []},
            concept_rankings={"top": [], "bottom": []},
            market_snapshot={"up_count": 0, "down_count": 0, "flat_count": 0},
        )

        self.assertEqual(evidence["status"], "unknown")
        self.assertIn("limit_up_pool", evidence["data_quality"]["missing_fields"])
        self.assertIn("breadth", evidence["data_quality"]["contaminated_fields"])
        self.assertTrue(evidence["data_quality"]["errors"])


class LimitEventPoolManagerTestCase(unittest.TestCase):
    def test_manager_keeps_failure_chain_before_success(self) -> None:
        class FailingFetcher(BaseFetcher):
            name = "Failing"
            priority = 1

            def _fetch_raw_data(self, stock_code, start_date, end_date):
                return pd.DataFrame()

            def _normalize_data(self, df, stock_code):
                return df

            def get_limit_event_pool(self, pool_type, date=None, n=200):
                raise RuntimeError("boom")

        class WorkingFetcher(FailingFetcher):
            name = "Working"
            priority = 2

            def get_limit_event_pool(self, pool_type, date=None, n=200):
                return [{"code": "600001", "pool_type": pool_type}]

        manager = DataFetcherManager(fetchers=[FailingFetcher(), WorkingFetcher()])
        rows, chain, error = manager.get_limit_event_pool_with_meta("broken_board", "20260717", 20)

        self.assertEqual(rows[0]["code"], "600001")
        self.assertEqual([item["status"] for item in chain], ["failed", "ok"])
        self.assertEqual(error, "")


class AkshareLimitEventPoolTestCase(unittest.TestCase):
    def test_normalizes_previous_limit_pool(self) -> None:
        df = pd.DataFrame(
            {
                "代码": ["600001"],
                "名称": ["示例"],
                "涨跌幅": [-1.25],
                "最新价": [12.3],
                "成交额": [1_200_000_000],
                "换手率": [8.5],
                "所属行业": ["半导体"],
            }
        )
        fake_akshare = SimpleNamespace(stock_zt_pool_previous_em=lambda **kwargs: df)
        original_akshare = sys.modules.get("akshare")
        sys.modules["akshare"] = fake_akshare
        try:
            fetcher = AkshareFetcher(sleep_min=0, sleep_max=0)
            fetcher._set_random_user_agent = MagicMock()
            fetcher._enforce_rate_limit = MagicMock()
            rows = fetcher.get_limit_event_pool("previous_limit", date="20260717", n=20)
        finally:
            if original_akshare is None:
                sys.modules.pop("akshare", None)
            else:
                sys.modules["akshare"] = original_akshare

        self.assertEqual(rows[0]["pool_type"], "previous_limit")
        self.assertEqual(rows[0]["change_pct"], -1.25)
        self.assertEqual(rows[0]["industry"], "半导体")


class MarketReviewEvidenceContractTestCase(unittest.TestCase):
    def test_payload_and_prompt_expose_structured_evidence(self) -> None:
        with patch("src.market_analyzer.DataFetcherManager"):
            analyzer = MarketAnalyzer(region="cn", config=SimpleNamespace(report_language="zh"))
        overview = MarketOverview(
            date="2026-07-17",
            generated_at="2026-07-17T18:00:00",
            run_date="2026-07-17",
            data_date="2026-07-17",
            indices=[MarketIndex(code="sh000001", name="上证指数", current=3200.0)],
            a_share_evidence={
                "status": "ok",
                "sentiment_structure": {
                    "limit_up_count": 50,
                    "broken_board_count": 25,
                    "broken_ratio": 0.3333,
                    "limit_down_count": 5,
                    "previous_limit_premium_median_pct": -0.5,
                    "highest_consecutive_board": 4,
                    "one_price_like_ratio": 0.2,
                },
                "theme_candidates": [
                    {
                        "theme": "半导体",
                        "classification": "mainline_candidate",
                        "total_score": 72,
                        "limit_up_count": 6,
                        "max_board_height": 3,
                        "confirmation": "容量核心承接",
                        "invalidation": "前排断板",
                    }
                ],
                "stock_candidates": [
                    {
                        "category": "capacity_core",
                        "code": "600002",
                        "name": "芯片容量",
                        "themes": ["半导体"],
                        "buy_point_type": "回踩承接",
                        "validation": "板块共振",
                        "invalidation": "跌破承接位",
                        "risk_tags": [],
                    }
                ],
                "risk_rules": [
                    {
                        "code": "high_broken_board_ratio",
                        "triggered": True,
                        "evidence": "炸板率33.3%",
                        "action": "降低接力仓位",
                    }
                ],
                "risk_tags": ["high_broken_board_ratio"],
                "data_quality": {"missing_fields": [], "contaminated_fields": [], "errors": []},
            },
        )

        prompt = analyzer._build_review_prompt(overview, [])
        payload = analyzer.build_market_review_payload(overview, [], "## 复盘\n\n### 盘面总览\n内容")

        self.assertIn("A股情绪与题材证据", prompt)
        self.assertIn("炸板率 33.3%", prompt)
        self.assertIn("芯片容量", prompt)
        self.assertEqual(payload["a_share_evidence"]["risk_tags"], ["high_broken_board_ratio"])
        self.assertIn("fund_flows", payload)
        self.assertIn("index_key_levels", payload)


if __name__ == "__main__":
    unittest.main()
