# -*- coding: utf-8 -*-
"""Regression tests for the integrated A-share daily-review evidence contract."""

from __future__ import annotations

import sys
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from data_provider.akshare_fetcher import AkshareFetcher
from data_provider.base import BaseFetcher, DataFetcherManager
from src.market_analyzer import MarketAnalyzer, MarketIndex, MarketOverview
from src.services.a_share_review_evidence import AShareReviewEvidenceService, _bars_through_trade_date


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
    def test_index_history_drops_bars_after_target_trade_date(self) -> None:
        bars = [
            {"date": "2026-07-30", "close": 101},
            {"date": "2026-07-31", "close": 102},
        ]

        self.assertEqual(_bars_through_trade_date(bars, "20260729"), [])

    def test_builds_sentiment_themes_watchlist_and_risk_gates(self) -> None:
        manager = _EvidenceManager()
        service = AShareReviewEvidenceService(manager)

        evidence = service.build(
            trade_date="20260717",
            indices=[
                {"code": "sh000001", "name": "上证指数", "current": 121.0, "as_of": "2026-07-17"},
                {"code": "sz399006", "name": "创业板指", "current": 205.0, "as_of": "2026-07-17"},
                {"code": "sh000688", "name": "科创50", "current": 204.0, "as_of": "2026-07-17"},
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
            global_indices=[
                {"code": "IXIC", "name": "纳斯达克", "change_pct": 0.3, "as_of": "2026-07-17"},
                {"code": "SOX", "name": "费城半导体指数", "change_pct": 0.5, "as_of": "2026-07-17"},
            ],
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
        self.assertEqual(evidence["data_quality"]["status"], "partial")
        self.assertIn("index_trend_as_of:399001", evidence["data_quality"]["missing_fields"])

    def test_20260729_fixture_uses_canonical_sentiment_five_indices_and_theme_caps(self) -> None:
        fixture = json.loads(
            (Path(__file__).parent / "fixtures" / "a_share_review_20260729.json").read_text(encoding="utf-8")
        )

        class FixtureManager:
            def __init__(self) -> None:
                limit_up = [
                    {
                        "code": "603221",
                        "name": "爱丽家居",
                        "industry": "CPO",
                        "change_pct": 10,
                        "amount": 1_000_000_000,
                        "turnover_rate": 0.2,
                        "consecutive_boards": 7,
                        "first_limit_time": "092500",
                        "break_count": 0,
                    },
                    {
                        "code": "300001",
                        "name": "CPO容量",
                        "industry": "CPO",
                        "change_pct": 10,
                        "amount": 8_000_000_000,
                        "turnover_rate": 8,
                        "consecutive_boards": 2,
                        "first_limit_time": "100100",
                        "break_count": 1,
                    },
                    {
                        "code": "600101",
                        "name": "化工容量",
                        "industry": "化学原料",
                        "change_pct": 10,
                        "amount": 7_000_000_000,
                        "turnover_rate": 12,
                        "consecutive_boards": 3,
                        "first_limit_time": "101000",
                        "break_count": 0,
                    },
                ]
                filler_industries = ["低空经济", "机器人", "化学制品", "CPO"]
                for index in range(78):
                    limit_up.append(
                        {
                            "code": f"60{index + 1000:04d}",
                            "name": f"样本{index + 1}",
                            "industry": filler_industries[index % len(filler_industries)],
                            "change_pct": 10,
                            "amount": 1_500_000_000 + index * 10_000_000,
                            "turnover_rate": 6,
                            "consecutive_boards": 2 if index < 8 else 1,
                            "first_limit_time": "101500",
                            "break_count": 0,
                        }
                    )
                premiums = [0.42] * 31 + [0.8866667] * 30
                self.pools = {
                    "limit_up": limit_up,
                    "broken_board": [{"code": f"B{index}"} for index in range(14)],
                    "previous_limit": [{"change_pct": value} for value in premiums],
                    "limit_down": [{"code": f"D{index}"} for index in range(9)],
                }
                self.histories = {
                    item["symbol"]: [
                        {
                            "date": f"2026-07-{day + 9:02d}",
                            "open": close,
                            "high": close + 1,
                            "low": close - 1,
                            "close": close,
                            "provider": "fixture",
                        }
                        for day, close in enumerate(item["history_closes"])
                    ]
                    for item in fixture["indices"]
                }

            def get_limit_event_pool_with_meta(self, pool_type: str, date: str, n: int):
                rows = self.pools[pool_type]
                return rows, [{"provider": "fixture", "status": "ok"}], ""

            def get_index_daily_history(self, symbol: str, days: int = 30):
                return self.histories[symbol]

        manager = FixtureManager()
        evidence = AShareReviewEvidenceService(manager).build(
            trade_date="20260729",
            indices=[
                {
                    "code": item["code"],
                    "name": item["name"],
                    "current": item["current"],
                    "change_pct": item["change_pct"],
                    "as_of": fixture["trade_date"],
                }
                for item in fixture["indices"]
            ],
            sector_rankings=fixture["sector_rankings"],
            concept_rankings=fixture["concept_rankings"],
            market_snapshot=fixture["market_stats"],
            global_indices=fixture["external_indices"],
        )

        sentiment = evidence["sentiment_structure"]
        self.assertEqual(sentiment["limit_up_count"], 81)
        self.assertEqual(sentiment["broken_board_count"], 14)
        self.assertEqual(sentiment["limit_down_count"], 9)
        self.assertEqual(sentiment["broken_ratio"], 0.1474)
        self.assertEqual(sentiment["previous_limit_premium_mean_pct"], 0.65)
        self.assertEqual(sentiment["previous_limit_premium_median_pct"], 0.42)
        self.assertEqual(len(evidence["index_trend"]), 5)
        self.assertEqual(
            {item["name"] for item in evidence["index_trend"]},
            {"上证指数", "深证成指", "创业板指", "科创50", "沪深300"},
        )
        self.assertLessEqual(
            sum(item["classification"] in {"confirmed_mainline", "mainline_candidate"} for item in evidence["theme_candidates"]),
            3,
        )
        themes = {item["theme"]: item for item in evidence["theme_candidates"]}
        self.assertIn("CPO", themes)
        self.assertIn("PCB", themes)
        self.assertIn("半导体", themes)
        self.assertNotEqual(themes["消费链"]["classification"], "confirmed_mainline")
        leader = next(item for item in evidence["stock_candidates"] if item["name"] == "爱丽家居")
        self.assertEqual(leader["trade_eligibility"], "observation_only")
        self.assertEqual(leader["position_cap_pct"], 0)
        self.assertEqual(evidence["external_tech_context"]["status"], "ok")

    def test_external_tech_context_requires_timestamp_for_each_required_index(self) -> None:
        context = AShareReviewEvidenceService._build_external_tech_context(
            [
                {"code": "SPX", "change_pct": -0.5, "as_of": "2026-07-29"},
                {"code": "IXIC", "change_pct": -2.1, "as_of": "2026-07-29"},
                {"code": "SOX", "change_pct": -4.5},
            ]
        )

        self.assertEqual(context.status, "partial")
        self.assertEqual(context.as_of, "2026-07-29")

    def test_liquid_first_board_does_not_replace_tradeable_front_row(self) -> None:
        candidates = AShareReviewEvidenceService._build_theme_candidates(
            {
                "active_themes": [
                    {"name": "测试题材", "source": "sector", "rank": 1, "change_pct": 5.0, "strength_score": 90}
                ],
                "lagging_themes": [],
            },
            [
                {
                    "code": "600001",
                    "name": "一字高标",
                    "industry": "测试题材",
                    "amount": 1_000_000_000,
                    "turnover_rate": 0.2,
                    "consecutive_boards": 4,
                    "first_limit_time": "092500",
                    "break_count": 0,
                },
                {
                    "code": "600002",
                    "name": "容量首板",
                    "industry": "测试题材",
                    "amount": 8_000_000_000,
                    "turnover_rate": 8.0,
                    "consecutive_boards": 1,
                    "first_limit_time": "100000",
                    "break_count": 0,
                },
            ],
        )

        theme = candidates[0]
        self.assertTrue(theme.capacity_core_present)
        self.assertFalse(theme.tradeable_front_present)
        self.assertNotIn(theme.classification, {"confirmed_mainline", "mainline_candidate"})

    def test_index_history_is_requested_for_all_targets_without_runtime_snapshot(self) -> None:
        manager = MagicMock()
        manager.get_index_daily_history.side_effect = lambda symbol, days=30: _bars(100.0)
        rows, _, _ = AShareReviewEvidenceService(manager)._build_index_trend(
            [],
            trade_date="20260729",
        )

        self.assertEqual(manager.get_index_daily_history.call_count, 5)
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(item.current_source == "historical_close_fallback" for item in rows))

    def test_exact_historical_close_beats_runtime_snapshot_without_as_of(self) -> None:
        manager = MagicMock()
        manager.get_index_daily_history.side_effect = lambda symbol, days=30: _bars(100.0)
        rows, _, _ = AShareReviewEvidenceService(manager)._build_index_trend(
            [{"code": "sh000001", "name": "上证指数", "current": 9999.0, "change_pct": 9.9}],
            trade_date="20260620",
        )

        shanghai = next(item for item in rows if item.code == "000001")
        self.assertEqual(shanghai.current_source, "historical_close_fallback")
        self.assertEqual(shanghai.status, "ok")
        self.assertEqual(shanghai.current, 120.0)

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
