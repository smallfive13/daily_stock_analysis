# -*- coding: utf-8 -*-
"""Tests for localized market review wrappers."""

import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()

def _build_optional_module_stubs() -> dict[str, ModuleType]:
    stubs: dict[str, ModuleType] = {}
    google_module: ModuleType | None = None

    for module_name in ("google.generativeai", "google.genai", "anthropic"):
        try:
            importlib.import_module(module_name)
            continue
        except ImportError:
            stub = ModuleType(module_name)
            stubs[module_name] = stub
            if not module_name.startswith("google."):
                continue
            if google_module is None:
                try:
                    google_module = importlib.import_module("google")
                except ImportError:
                    google_module = ModuleType("google")
                    stubs["google"] = google_module
            setattr(google_module, module_name.split(".", 1)[1], stub)

    return stubs


sys.modules.update(_build_optional_module_stubs())
import src.core.market_review as market_review_module
from src.config import Config
from src.llm.generation_backend import GenerationError, GenerationErrorCode
from src.services.run_diagnostics import activate_run_diagnostic_context, reset_run_diagnostic_context
from src.storage import AnalysisHistory, DatabaseManager

run_market_review = market_review_module.run_market_review


class MarketReviewLocalizationTestCase(unittest.TestCase):
    def _make_notifier(self) -> MagicMock:
        notifier = MagicMock()
        notifier.save_report_to_file.return_value = "/tmp/market_review.md"
        notifier.is_available.return_value = True
        notifier.send.return_value = True
        return notifier

    def test_resolve_market_review_regions_returns_ordered_non_empty_list(self) -> None:
        cases = [
            (None, ["cn"]),
            ("", ["cn"]),
            ("both", ["cn", "hk", "us", "jp", "kr"]),
            (" CN,US,cn ", ["cn", "us"]),
            ("us,cn,us", ["cn", "us"]),
            ("jp", ["jp"]),
            ("KR", ["kr"]),
            ("kr,jp,us", ["us", "jp", "kr"]),
            ("eu,apac", ["cn"]),
            (",,", ["cn"]),
            ("HK", ["hk"]),
            ("invalid", ["cn"]),
        ]

        for raw_region, expected in cases:
            with self.subTest(raw_region=raw_region):
                self.assertEqual(
                    market_review_module._resolve_market_review_regions(raw_region),
                    expected,
                )

    def test_run_market_review_uses_english_notification_title(self) -> None:
        notifier = self._make_notifier()
        market_analyzer = MagicMock()
        market_analyzer.run_daily_review_with_snapshot.return_value = SimpleNamespace(
            report="## 2026-04-10 A-share Market Recap\n\nBody",
            market_light_snapshot={"region": "cn", "trade_date": "2026-04-10", "score": 60},
        )

        with patch.object(
            market_review_module,
            "get_config",
            return_value=SimpleNamespace(report_language="en", market_review_region="cn"),
        ), patch.object(
            market_review_module,
            "MarketAnalyzer",
            return_value=market_analyzer,
        ), patch.object(market_review_module, "_persist_market_review_history") as persist_history:
            result = run_market_review(notifier, send_notification=True)

        self.assertEqual(result, "## 2026-04-10 A-share Market Recap\n\nBody")
        saved_content = notifier.save_report_to_file.call_args.args[0]
        self.assertTrue(saved_content.startswith("[dsa-market-region]: # (cn)\n\n# 🎯 Market Review\n\n"))
        sent_content = notifier.send.call_args.args[0]
        self.assertTrue(sent_content.startswith("[dsa-market-region]: # (cn)\n\n🎯 Market Review\n\n"))
        self.assertTrue(notifier.send.call_args.kwargs["email_send_to_all"])
        self.assertEqual(notifier.send.call_args.kwargs["route_type"], "report")
        persist_history.assert_called_once()
        self.assertTrue(persist_history.call_args.kwargs["query_id"].startswith("market_review_"))

    def test_run_market_review_can_skip_report_file_for_context_generation(self) -> None:
        notifier = self._make_notifier()
        market_analyzer = MagicMock()
        market_analyzer.run_daily_review_with_snapshot.return_value = SimpleNamespace(
            report="CN body",
            market_light_snapshot={"region": "cn", "trade_date": "2026-03-06", "score": 60},
        )

        with patch.object(
            market_review_module,
            "get_config",
            return_value=SimpleNamespace(report_language="zh", market_review_region="cn"),
        ), patch.object(
            market_review_module,
            "MarketAnalyzer",
            return_value=market_analyzer,
        ), patch.object(market_review_module, "_persist_market_review_history") as persist_history:
            result = run_market_review(
                notifier,
                send_notification=False,
                return_structured=True,
                save_report_file=False,
            )

        self.assertIsInstance(result, market_review_module.MarketReviewRunResult)
        self.assertEqual(result.report, "CN body")
        notifier.save_report_to_file.assert_not_called()
        persist_history.assert_called_once()

    def test_run_market_review_passes_request_config_to_generation(self) -> None:
        notifier = self._make_notifier()
        request_config = SimpleNamespace(report_language="en", market_review_region="cn")
        global_config = SimpleNamespace(report_language="zh", market_review_region="cn")
        market_analyzer = MagicMock()
        market_analyzer.run_daily_review_with_snapshot.return_value = SimpleNamespace(
            report="English market review body",
            market_light_snapshot={"region": "cn", "trade_date": "2026-04-10", "score": 60},
        )

        with patch.object(
            market_review_module,
            "get_config",
            return_value=global_config,
        ), patch.object(
            market_review_module,
            "MarketAnalyzer",
            return_value=market_analyzer,
        ) as analyzer_cls, patch.object(market_review_module, "_persist_market_review_history") as persist_history:
            result = run_market_review(
                notifier,
                config=request_config,
                send_notification=False,
                return_structured=True,
            )

        self.assertEqual(analyzer_cls.call_args.kwargs["config"], request_config)
        self.assertEqual(persist_history.call_args.kwargs["config"], request_config)
        self.assertTrue(
            notifier.save_report_to_file.call_args.args[0].startswith(
                "[dsa-market-region]: # (cn)\n\n# 🎯 Market Review\n\n"
            )
        )
        self.assertEqual(result.market_review_payload["language"], "en")
        self.assertEqual(result.report, "English market review body")

    def test_run_market_review_returns_sector_fallback_for_merged_notification(self) -> None:
        notifier = self._make_notifier()
        market_analyzer = MagicMock()
        market_analyzer.run_daily_review_with_snapshot.return_value = SimpleNamespace(
            report="## 今日大盘\n\n盘面正文。",
            market_light_snapshot={"region": "cn", "trade_date": "2026-06-03", "score": 60},
            structured_payload={
                "kind": "market_review",
                "region": "cn",
                "language": "zh",
                "title": "今日大盘",
                "sections": [
                    {
                        "key": "overview",
                        "title": "概览",
                        "markdown": "盘面正文。",
                    }
                ],
                "sectors": {
                    "top": [{"name": "AI算力", "change_pct": 3.25}],
                    "bottom": [{"name": "煤炭", "change_pct": -1.12}],
                },
            },
        )

        with patch.object(
            market_review_module,
            "get_config",
            return_value=SimpleNamespace(report_language="zh", market_review_region="cn"),
        ), patch.object(
            market_review_module,
            "MarketAnalyzer",
            return_value=market_analyzer,
        ), patch.object(market_review_module, "_persist_market_review_history"):
            result = run_market_review(
                notifier,
                send_notification=True,
                merge_notification=True,
            )

        self.assertIn("## 今日大盘", result)
        self.assertIn("### 板块主线", result)
        self.assertIn("| 1 | AI算力 | +3.25% |", result)
        self.assertIn("| 1 | 煤炭 | -1.12% |", result)
        notifier.send.assert_not_called()

    def test_run_market_review_reraises_generation_backend_config_error(self) -> None:
        notifier = self._make_notifier()
        backend_error = GenerationError(
            error_code=GenerationErrorCode.BACKEND_NOT_CONFIGURED,
            stage="generation",
            retryable=False,
            fallbackable=False,
            backend="codex",
            details={
                "field": "GENERATION_BACKEND",
                "requested_backend": "codex",
            },
        )
        market_analyzer = MagicMock()
        market_analyzer.run_daily_review_with_snapshot.side_effect = backend_error

        with patch.object(
            market_review_module,
            "MarketAnalyzer",
            return_value=market_analyzer,
        ):
            with self.assertRaises(GenerationError) as exc_info:
                run_market_review(
                    notifier,
                    config=SimpleNamespace(report_language="zh", market_review_region="cn"),
                    send_notification=False,
                    save_report_file=False,
                    persist_history=False,
                )

        self.assertIs(exc_info.exception, backend_error)
        notifier.save_report_to_file.assert_not_called()
        notifier.send.assert_not_called()

    def test_run_market_review_merges_both_regions_with_english_wrappers(self) -> None:
        notifier = self._make_notifier()
        cn_analyzer = MagicMock()
        cn_analyzer.run_daily_review_with_snapshot.return_value = SimpleNamespace(
            report="CN body",
            market_light_snapshot={"region": "cn", "trade_date": "2026-03-06", "score": 60},
        )
        hk_analyzer = MagicMock()
        hk_analyzer.run_daily_review_with_snapshot.return_value = SimpleNamespace(
            report="HK body",
            market_light_snapshot={"region": "hk", "trade_date": "2026-03-06", "score": 58},
        )
        us_analyzer = MagicMock()
        us_analyzer.run_daily_review_with_snapshot.return_value = SimpleNamespace(
            report="US body",
            market_light_snapshot={"region": "us", "trade_date": "2026-03-06", "score": 55},
        )
        jp_analyzer = MagicMock()
        jp_analyzer.run_daily_review_with_snapshot.return_value = SimpleNamespace(
            report="JP body",
            market_light_snapshot={"region": "jp", "trade_date": "2026-03-06", "score": 54},
        )
        kr_analyzer = MagicMock()
        kr_analyzer.run_daily_review_with_snapshot.return_value = SimpleNamespace(
            report="KR body",
            market_light_snapshot={"region": "kr", "trade_date": "2026-03-06", "score": 53},
        )

        with patch.object(
            market_review_module,
            "get_config",
            return_value=SimpleNamespace(report_language="en", market_review_region="both"),
        ), patch.object(
            market_review_module,
            "MarketAnalyzer",
            side_effect=[cn_analyzer, hk_analyzer, us_analyzer, jp_analyzer, kr_analyzer],
        ), patch.object(market_review_module, "_persist_market_review_history") as persist_history:
            result = run_market_review(notifier, send_notification=True)

        self.assertIn("# A-share Market Recap\n\nCN body", result)
        self.assertIn("# HK Market Recap\n\nHK body", result)
        self.assertIn("> Next market recap follows", result)
        self.assertIn("# US Market Recap\n\nUS body", result)
        self.assertIn("# Japan Market Recap\n\nJP body", result)
        self.assertIn("# Korea Market Recap\n\nKR body", result)
        saved_content = notifier.save_report_to_file.call_args.args[0]
        self.assertTrue(saved_content.startswith("# 🎯 Market Review\n\n"))
        self.assertIn("# A-share Market Recap\n\nCN body", saved_content)
        self.assertIn("> Next market recap follows", saved_content)
        self.assertIn("# HK Market Recap\n\nHK body", saved_content)
        self.assertIn("# US Market Recap\n\nUS body", saved_content)
        self.assertIn("# Japan Market Recap\n\nJP body", saved_content)
        self.assertIn("# Korea Market Recap\n\nKR body", saved_content)
        self.assertIn(
            "# A-share Market Recap\n\nCN body",
            persist_history.call_args.kwargs["markdown_report"],
        )
        self.assertEqual(
            set(persist_history.call_args.kwargs["market_light_snapshots"]),
            {"cn", "hk", "us"},
        )
        sent_content = notifier.send.call_args.args[0]
        self.assertTrue(sent_content.startswith("🎯 Market Review\n\n"))
        self.assertIn("# US Market Recap\n\nUS body", sent_content)
        self.assertIn("# Japan Market Recap\n\nJP body", sent_content)
        self.assertIn("# Korea Market Recap\n\nKR body", sent_content)

    def test_run_market_review_comma_joined_subset_jp_kr(self) -> None:
        notifier = self._make_notifier()
        jp_analyzer = MagicMock()
        jp_analyzer.run_daily_review_with_snapshot.return_value = SimpleNamespace(
            report="JP body",
            market_light_snapshot={"region": "jp", "trade_date": "2026-03-06", "score": 54},
        )
        kr_analyzer = MagicMock()
        kr_analyzer.run_daily_review_with_snapshot.return_value = SimpleNamespace(
            report="KR body",
            market_light_snapshot={"region": "kr", "trade_date": "2026-03-06", "score": 53},
        )

        with patch.object(
            market_review_module,
            "get_config",
            return_value=SimpleNamespace(report_language="zh", market_review_region="cn"),
        ), patch.object(
            market_review_module,
            "MarketAnalyzer",
            side_effect=[jp_analyzer, kr_analyzer],
        ), patch.object(market_review_module, "_persist_market_review_history"):
            result = run_market_review(
                notifier, send_notification=False, override_region="jp,kr"
            )

        self.assertIn("# 日股大盘复盘\n\nJP body", result)
        self.assertIn("# 韩股大盘复盘\n\nKR body", result)
        self.assertNotIn("A股大盘复盘", result)
        self.assertNotIn("美股大盘复盘", result)

    def test_run_market_review_comma_joined_subset_cn_us(self) -> None:
        """Regression: compute_effective_region("both", {"cn","us"}) -> "cn,us"
        must produce A-share + US report without HK."""
        notifier = self._make_notifier()
        cn_analyzer = MagicMock()
        cn_analyzer.run_daily_review_with_snapshot.return_value = SimpleNamespace(
            report="CN body",
            market_light_snapshot={"region": "cn", "trade_date": "2026-03-06", "score": 60},
        )
        us_analyzer = MagicMock()
        us_analyzer.run_daily_review_with_snapshot.return_value = SimpleNamespace(
            report="US body",
            market_light_snapshot={"region": "us", "trade_date": "2026-03-06", "score": 55},
        )

        with patch.object(
            market_review_module,
            "get_config",
            return_value=SimpleNamespace(report_language="zh", market_review_region="cn"),
        ), patch.object(
            market_review_module,
            "MarketAnalyzer",
            side_effect=[cn_analyzer, us_analyzer],
        ), patch.object(market_review_module, "_persist_market_review_history"):
            result = run_market_review(
                notifier, send_notification=False, override_region="cn,us"
            )

        self.assertIn("# A股大盘复盘\n\nCN body", result)
        self.assertIn("# 美股大盘复盘\n\nUS body", result)
        self.assertNotIn("港股", result)
        self.assertNotIn("HK", result)

    def test_run_market_review_comma_joined_subset_cn_hk(self) -> None:
        """Regression: compute_effective_region("both", {"cn","hk"}) -> "cn,hk"
        must produce A-share + HK report without US."""
        notifier = self._make_notifier()
        cn_analyzer = MagicMock()
        cn_analyzer.run_daily_review_with_snapshot.return_value = SimpleNamespace(
            report="CN body",
            market_light_snapshot={"region": "cn", "trade_date": "2026-03-06", "score": 60},
        )
        hk_analyzer = MagicMock()
        hk_analyzer.run_daily_review_with_snapshot.return_value = SimpleNamespace(
            report="HK body",
            market_light_snapshot={"region": "hk", "trade_date": "2026-03-06", "score": 58},
        )

        with patch.object(
            market_review_module,
            "get_config",
            return_value=SimpleNamespace(report_language="zh", market_review_region="cn"),
        ), patch.object(
            market_review_module,
            "MarketAnalyzer",
            side_effect=[cn_analyzer, hk_analyzer],
        ), patch.object(market_review_module, "_persist_market_review_history"):
            result = run_market_review(
                notifier, send_notification=False, override_region="cn,hk"
            )

        self.assertIn("# A股大盘复盘\n\nCN body", result)
        self.assertIn("# 港股大盘复盘\n\nHK body", result)
        self.assertNotIn("美股", result)
        self.assertNotIn("US Market", result)

    def test_run_market_review_persists_only_current_run_market_light_snapshots(self) -> None:
        notifier = self._make_notifier()
        cn_analyzer = MagicMock()
        cn_analyzer.run_daily_review_with_snapshot.return_value = SimpleNamespace(
            report="CN body",
            market_light_snapshot={"region": "cn", "trade_date": "2026-03-06", "score": 60},
        )
        us_analyzer = MagicMock()
        us_analyzer.run_daily_review_with_snapshot.return_value = SimpleNamespace(
            report="US body",
            market_light_snapshot={"region": "us", "trade_date": "2026-03-06", "score": 55},
        )

        with patch.object(
            market_review_module,
            "get_config",
            return_value=SimpleNamespace(report_language="zh", market_review_region="cn"),
        ), patch.object(
            market_review_module,
            "MarketAnalyzer",
            side_effect=[cn_analyzer, us_analyzer],
        ), patch.object(market_review_module, "_persist_market_review_history") as persist_history:
            run_market_review(notifier, send_notification=False, override_region="cn,us")

        snapshots = persist_history.call_args.kwargs["market_light_snapshots"]
        self.assertEqual(set(snapshots), {"cn", "us"})
        self.assertEqual(snapshots["cn"]["score"], 60)
        self.assertEqual(snapshots["us"]["score"], 55)

    def test_run_market_review_jp_kr_skips_market_light_snapshot_schema(self) -> None:
        notifier = self._make_notifier()

        from src.market_analyzer import MarketOverview

        with patch.object(
            market_review_module.MarketAnalyzer,
            "get_market_overview",
            side_effect=[
                MarketOverview(date="2026-03-06"),
                MarketOverview(date="2026-03-06"),
            ],
        ), patch.object(
            market_review_module.MarketAnalyzer,
            "search_market_news",
            return_value=[],
        ), patch.object(
            market_review_module.MarketAnalyzer,
            "generate_market_review",
            side_effect=["JP body", "KR body"],
        ), patch.object(market_review_module, "_persist_market_review_history") as persist_history:
            result = run_market_review(
                notifier,
                config=SimpleNamespace(report_language="zh", market_review_region="jp,kr"),
                send_notification=False,
            )

        self.assertIn("# 日股大盘复盘\n\nJP body", result)
        self.assertIn("# 韩股大盘复盘\n\nKR body", result)
        self.assertEqual(persist_history.call_args.kwargs["market_light_snapshots"], {})
        payload = persist_history.call_args.kwargs["market_review_payload"]
        self.assertNotIn("market_light", payload["markets"]["jp"])
        self.assertNotIn("market_light", payload["markets"]["kr"])

    def test_run_market_review_normalizes_single_region_snapshot_key(self) -> None:
        notifier = self._make_notifier()
        market_analyzer = MagicMock()
        market_analyzer.run_daily_review_with_snapshot.return_value = SimpleNamespace(
            report="CN body",
            market_light_snapshot={
                "region": "cn",
                "trade_date": "2026-03-06",
                "score": 60,
            },
        )

        with patch.object(
            market_review_module,
            "get_config",
            return_value=SimpleNamespace(report_language="zh", market_review_region="cn"),
        ), patch.object(
            market_review_module,
            "MarketAnalyzer",
            return_value=market_analyzer,
        ) as analyzer_cls, patch.object(
            market_review_module, "_persist_market_review_history"
        ) as persist_history:
            run_market_review(notifier, send_notification=False, override_region="CN")

        self.assertEqual(analyzer_cls.call_args.kwargs["region"], "cn")
        persist_history.assert_called_once()
        self.assertEqual(persist_history.call_args.kwargs["region"], "cn")
        snapshots = persist_history.call_args.kwargs["market_light_snapshots"]
        self.assertEqual(set(snapshots), {"cn"})
        self.assertEqual(snapshots["cn"]["trade_date"], "2026-03-06")

    def test_run_market_review_invalid_comma_subset_falls_back_to_cn(self) -> None:
        notifier = self._make_notifier()
        market_analyzer = MagicMock()
        market_analyzer.run_daily_review_with_snapshot.return_value = SimpleNamespace(
            report="CN body",
            market_light_snapshot={
                "region": "cn",
                "trade_date": "2026-03-06",
                "score": 60,
            },
        )

        with patch.object(
            market_review_module,
            "get_config",
            return_value=SimpleNamespace(report_language="zh", market_review_region="cn"),
        ), patch.object(
            market_review_module,
            "MarketAnalyzer",
            return_value=market_analyzer,
        ) as analyzer_cls, patch.object(
            market_review_module, "_persist_market_review_history"
        ) as persist_history:
            result = run_market_review(
                notifier, send_notification=False, override_region="eu,apac"
            )

        self.assertEqual(result, "CN body")
        self.assertEqual(analyzer_cls.call_args.kwargs["region"], "cn")
        persist_history.assert_called_once()
        self.assertEqual(persist_history.call_args.kwargs["region"], "cn")
        snapshots = persist_history.call_args.kwargs["market_light_snapshots"]
        self.assertEqual(set(snapshots), {"cn"})

    def test_render_market_review_payload_markdown_does_not_repeat_title(self) -> None:
        markdown = market_review_module._render_market_review_payload_markdown(
            {
                "title": "2026-06-03 大盘复盘",
                "sections": [
                    {
                        "key": "daily_review",
                        "title": "2026-06-03 大盘复盘",
                        "markdown": "> 今日指数强弱分化。\n\n### 一、盘面总览\n正文",
                    }
                ],
            },
            wrapper_title="🎯 大盘复盘",
        )

        self.assertEqual(markdown.count("2026-06-03 大盘复盘"), 1)
        self.assertTrue(markdown.startswith("🎯 大盘复盘\n\n## 2026-06-03 大盘复盘"))

    def test_render_market_review_payload_markdown_prefixes_single_region_metadata(self) -> None:
        markdown = market_review_module._render_market_review_payload_markdown(
            {
                "region": "us",
                "title": "2026-06-03 大盘复盘",
                "sections": [
                    {
                        "key": "overview",
                        "title": "2026-06-03 大盘复盘",
                        "markdown": "> 今晚重点观察科技股承接。",
                    }
                ],
            },
            wrapper_title="🎯 大盘复盘",
        )

        self.assertTrue(markdown.startswith("[dsa-market-region]: # (us)\n\n🎯 大盘复盘"))
        self.assertEqual(markdown.count("[dsa-market-region]: # (us)"), 1)

    def test_render_market_review_payload_markdown_appends_structured_sector_fallback(self) -> None:
        markdown = market_review_module._render_market_review_payload_markdown(
            {
                "title": "2026-06-03 大盘复盘",
                "language": "zh",
                "sections": [
                    {
                        "key": "overview",
                        "title": "Overview",
                        "markdown": "> 今日指数强弱分化。",
                    }
                ],
                "sectors": {
                    "top": [{"name": "AI算力", "change_pct": 3.25}],
                    "bottom": [{"name": "煤炭", "change_pct": -1.12}],
                },
            },
            wrapper_title="🎯 大盘复盘",
        )

        self.assertIn("### 板块主线", markdown)
        self.assertIn("#### 领涨板块 Top 5", markdown)
        self.assertIn("| 1 | AI算力 | +3.25% |", markdown)
        self.assertIn("#### 领跌板块 Top 5", markdown)
        self.assertIn("| 1 | 煤炭 | -1.12% |", markdown)

    def test_render_market_review_payload_markdown_keeps_injected_chinese_sector_block_once(self) -> None:
        markdown = market_review_module._render_market_review_payload_markdown(
            {
                "title": "2026-06-03 大盘复盘",
                "language": "zh",
                "markdown_report": (
                    "## 2026-06-03 大盘复盘\n\n"
                    "### 板块表现\n\n"
                    "#### 行业板块领涨 Top 5\n"
                    "| 排名 | 行业板块 | 涨跌幅 |\n"
                    "|------|------|--------|\n"
                    "| 1 | AI算力 | +3.25% |"
                ),
                "sectors": {
                    "top": [{"name": "AI算力", "change_pct": 3.25}],
                    "bottom": [{"name": "煤炭", "change_pct": -1.12}],
                },
            }
        )

        self.assertEqual(markdown.count("#### 行业板块领涨 Top 5"), 1)
        self.assertNotIn("### 板块主线", markdown)
        self.assertNotIn("#### 领涨板块 Top 5", markdown)
        self.assertNotIn("#### 领跌板块 Top 5", markdown)

    def test_render_market_review_payload_markdown_appends_each_market_sector_fallback(self) -> None:
        markdown = market_review_module._render_market_review_payload_markdown(
            {
                "language": "zh",
                "markdown_report": (
                    "## A 股大盘\n\n今日震荡。\n\n"
                    "---\n\n"
                    "## 港股大盘\n\n今日反弹。\n\n"
                    "---\n\n"
                    "## 美股大盘\n\n科技走强。"
                ),
                "markets": {
                    "cn": {
                        "title": "A 股大盘",
                        "language": "zh",
                        "sectors": {"top": [{"name": "AI算力", "change_pct": 3.25}]},
                    },
                    "hk": {
                        "title": "港股大盘",
                        "language": "zh",
                        "sectors": {"top": [{"name": "科技", "change_pct": 2.18}]},
                    },
                    "us": {
                        "title": "美股大盘",
                        "language": "zh",
                        "sectors": {"top": [{"name": "半导体", "change_pct": 1.86}]},
                    },
                },
            }
        )

        self.assertIn("### A 股大盘 / 板块主线", markdown)
        self.assertIn("| 1 | AI算力 | +3.25% |", markdown)
        self.assertIn("### 港股大盘 / 板块主线", markdown)
        self.assertIn("| 1 | 科技 | +2.18% |", markdown)
        self.assertIn("### 美股大盘 / 板块主线", markdown)
        self.assertIn("| 1 | 半导体 | +1.86% |", markdown)
        self.assertLess(markdown.index("### A 股大盘 / 板块主线"), markdown.index("## 港股大盘"))
        self.assertLess(markdown.index("### 港股大盘 / 板块主线"), markdown.index("## 美股大盘"))

    def test_render_market_review_payload_markdown_checks_duplicate_titles_by_market_wrapper(self) -> None:
        duplicate_title = "2026-06-03 大盘复盘"
        markdown = market_review_module._render_market_review_payload_markdown(
            {
                "language": "zh",
                "markdown_report": (
                    "# A股大盘复盘\n\n"
                    f"## {duplicate_title}\n\n"
                    "### 板块表现\n\n"
                    "#### 行业板块领涨 Top 5\n"
                    "| 排名 | 行业板块 | 涨跌幅 |\n"
                    "|------|------|--------|\n"
                    "| 1 | AI算力 | +3.25% |\n\n"
                    "---\n\n"
                    "> 以下为下一市场大盘复盘\n\n"
                    "# 港股大盘复盘\n\n"
                    f"## {duplicate_title}\n\n"
                    "港股正文。\n\n"
                    "---\n\n"
                    "> 以下为下一市场大盘复盘\n\n"
                    "# 美股大盘复盘\n\n"
                    f"## {duplicate_title}\n\n"
                    "美股正文。"
                ),
                "markets": {
                    "cn": {
                        "title": duplicate_title,
                        "language": "zh",
                        "sectors": {"top": [{"name": "AI算力", "change_pct": 3.25}]},
                    },
                    "hk": {
                        "title": duplicate_title,
                        "language": "zh",
                        "sectors": {"top": [{"name": "科技", "change_pct": 2.18}]},
                    },
                    "us": {
                        "title": duplicate_title,
                        "language": "zh",
                        "sectors": {"top": [{"name": "半导体", "change_pct": 1.86}]},
                    },
                },
            }
        )

        self.assertEqual(markdown.count("#### 行业板块领涨 Top 5"), 1)
        self.assertEqual(markdown.count(f"### {duplicate_title} / 板块主线"), 2)
        self.assertIn("| 1 | 科技 | +2.18% |", markdown)
        self.assertIn("| 1 | 半导体 | +1.86% |", markdown)

    def test_render_market_review_payload_markdown_preserves_segment_boundaries_after_fallback(self) -> None:
        markdown = market_review_module._render_market_review_payload_markdown(
            {
                "language": "en",
                "markdown_report": (
                    "## CN Market\n\n"
                    "CN overview.\n\n"
                    "## HK Market\n\n"
                    "HK overview.\n\n"
                    "---\n\n"
                    "## US Market\n\n"
                    "US overview."
                ),
                "markets": {
                    "cn": {
                        "title": "CN Market",
                        "language": "en",
                        "sectors": {"top": [{"name": "AI", "change_pct": 3.25}]},
                    },
                    "hk": {
                        "title": "HK Market",
                        "language": "en",
                        "sectors": {"top": [{"name": "Tech", "change_pct": 2.18}]},
                    },
                    "us": {
                        "title": "US Market",
                        "language": "en",
                        "sectors": {},
                    },
                },
            }
        )

        self.assertIn("| 1 | AI | +3.25% |\n\n## HK Market", markdown)
        self.assertIn("| 1 | Tech | +2.18% |\n\n---\n\n## US Market", markdown)
        self.assertNotIn("+3.25% |## HK Market", markdown)
        self.assertNotIn("+2.18% |---", markdown)

    def test_persist_market_review_history_saves_markdown_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            old_db_path = os.environ.get("DATABASE_PATH")
            os.environ["DATABASE_PATH"] = os.path.join(temp_dir, "market_review_history.db")
            Config._instance = None
            DatabaseManager.reset_instance()
            try:
                saved = market_review_module._persist_market_review_history(
                    review_report="## 今日大盘\n\n复盘正文",
                    markdown_report="# 🎯 大盘复盘\n\n## 今日大盘\n\n复盘正文",
                    region="cn",
                    config=SimpleNamespace(report_language="zh"),
                    query_id="market-task-001",
                    market_light_snapshots={
                        "cn": {
                            "region": "cn",
                            "trade_date": "2026-03-06",
                            "status": "red",
                            "score": 30,
                            "label": "偏防守",
                            "temperature_label": "偏弱",
                            "reasons": ["test"],
                            "guidance": "test",
                            "dimensions": {
                                "breadth": {"score": 20, "available": True},
                                "index": {"score": 30, "available": True},
                                "limit": {"score": 10, "available": True},
                            },
                            "data_quality": "ok",
                        }
                    },
                    market_review_payload={
                        "version": 1,
                        "kind": "market_review",
                        "region": "cn",
                        "sections": [{"title": "今日大盘", "markdown": "复盘正文"}],
                    },
                )

                self.assertGreater(saved, 0)
                db = DatabaseManager.get_instance()
                with db.get_session() as session:
                    row = session.query(AnalysisHistory).filter(
                        AnalysisHistory.query_id == "market-task-001"
                    ).first()
                    self.assertIsNotNone(row)
                    self.assertEqual(row.id, saved)
                    self.assertEqual(row.code, market_review_module.MARKET_REVIEW_HISTORY_CODE)
                    self.assertEqual(row.name, "大盘复盘")
                    self.assertEqual(row.report_type, market_review_module.MARKET_REVIEW_REPORT_TYPE)
                    self.assertEqual(row.news_content, "## 今日大盘\n\n复盘正文")
                    self.assertIn("# 🎯 大盘复盘", row.raw_result)
                    self.assertIn('"market_light_snapshots"', row.context_snapshot)
                    self.assertIn('"market_review_payload"', row.context_snapshot)
                    self.assertIn('"trade_date": "2026-03-06"', row.context_snapshot)
                    snapshot = json.loads(row.context_snapshot or "{}")
                    self.assertEqual(snapshot["market_review_region"], "cn")
                    self.assertEqual(snapshot["market_review_payload"]["region"], "cn")
                    self.assertIn("analysis_context_pack_overview", snapshot)
            finally:
                DatabaseManager.reset_instance()
                Config._instance = None
                if old_db_path is None:
                    os.environ.pop("DATABASE_PATH", None)
                else:
                    os.environ["DATABASE_PATH"] = old_db_path

    def test_run_market_review_persists_notification_diagnostics_after_history_save(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            old_db_path = os.environ.get("DATABASE_PATH")
            os.environ["DATABASE_PATH"] = os.path.join(temp_dir, "market_review_notification.db")
            Config._instance = None
            DatabaseManager.reset_instance()
            query_id = "market-task-notification"
            notifier = self._make_notifier()
            market_analyzer = MagicMock()
            market_analyzer.run_daily_review_with_snapshot.return_value = SimpleNamespace(
                report="## 今日大盘\n\n复盘正文",
                market_light_snapshot={"region": "cn", "trade_date": "2026-03-06", "score": 60},
            )
            token = activate_run_diagnostic_context(
                trace_id="trace-market-notification",
                task_id=query_id,
                query_id=query_id,
                stock_code=market_review_module.MARKET_REVIEW_HISTORY_CODE,
                trigger_source="api",
            )
            try:
                with patch.object(market_review_module, "MarketAnalyzer", return_value=market_analyzer):
                    result = run_market_review(
                        notifier,
                        config=SimpleNamespace(report_language="zh", market_review_region="cn"),
                        send_notification=True,
                        query_id=query_id,
                        trigger_source="api",
                    )

                self.assertEqual(result, "## 今日大盘\n\n复盘正文")
                db = DatabaseManager.get_instance()
                with db.get_session() as session:
                    row = session.query(AnalysisHistory).filter(
                        AnalysisHistory.query_id == query_id
                    ).first()
                    self.assertIsNotNone(row)
                    context_snapshot = json.loads(row.context_snapshot)
                    notification_runs = context_snapshot["diagnostics"]["notification_runs"]
                    self.assertEqual(notification_runs[-1]["status"], "success")
                    self.assertTrue(notification_runs[-1]["success"])
            finally:
                reset_run_diagnostic_context(token)
                DatabaseManager.reset_instance()
                Config._instance = None
                if old_db_path is None:
                    os.environ.pop("DATABASE_PATH", None)
                else:
                    os.environ["DATABASE_PATH"] = old_db_path

    def test_run_market_review_reuses_generated_query_id_for_notification_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            old_db_path = os.environ.get("DATABASE_PATH")
            os.environ["DATABASE_PATH"] = os.path.join(temp_dir, "market_review_generated_query.db")
            Config._instance = None
            DatabaseManager.reset_instance()
            notifier = self._make_notifier()
            market_analyzer = MagicMock()
            market_analyzer.run_daily_review_with_snapshot.return_value = SimpleNamespace(
                report="## 今日大盘\n\n复盘正文",
                market_light_snapshot={"region": "cn", "trade_date": "2026-03-06", "score": 60},
            )
            token = activate_run_diagnostic_context(
                trace_id="trace-market-generated",
                task_id="task-market-generated",
                stock_code=market_review_module.MARKET_REVIEW_HISTORY_CODE,
                trigger_source="cli",
            )
            try:
                with patch.object(market_review_module, "MarketAnalyzer", return_value=market_analyzer):
                    result = run_market_review(
                        notifier,
                        config=SimpleNamespace(report_language="zh", market_review_region="cn"),
                        send_notification=True,
                        trigger_source="cli",
                    )

                self.assertEqual(result, "## 今日大盘\n\n复盘正文")
                db = DatabaseManager.get_instance()
                with db.get_session() as session:
                    rows = session.query(AnalysisHistory).all()
                    self.assertEqual(len(rows), 1)
                    row = rows[0]
                    self.assertTrue(row.query_id.startswith("market_review_"))
                    context_snapshot = json.loads(row.context_snapshot)
                    notification_runs = context_snapshot["diagnostics"]["notification_runs"]
                    self.assertEqual(notification_runs[-1]["status"], "success")
                    self.assertTrue(notification_runs[-1]["success"])
            finally:
                reset_run_diagnostic_context(token)
                DatabaseManager.reset_instance()
                Config._instance = None
                if old_db_path is None:
                    os.environ.pop("DATABASE_PATH", None)
                else:
                    os.environ["DATABASE_PATH"] = old_db_path


    def _build_20260729_conflicting_payload(self) -> tuple[dict, dict]:
        fixture = json.loads(
            (Path(__file__).parent / "fixtures" / "a_share_review_20260729.json").read_text(encoding="utf-8")
        )
        indices = [
            {
                "code": item["symbol"],
                "name": item["name"],
                "current": item["current"],
                "change_pct": item["change_pct"],
                "ma5": item["current"] + 4,
                "ma10": item["current"] + 6,
                "ma20": item["current"] + 8,
                "current_source": "runtime_snapshot",
            }
            for item in fixture["indices"]
        ]
        themes = [
            {
                "theme": "CPO",
                "classification": "mainline_candidate",
                "actionable": True,
                "change_pct": 3.2,
                "position_cap_pct": 10,
                "confirmation": "板块强于沪深300且容量核心不破当日低点",
                "invalidation": "板块转负或容量核心跌破当日低点",
            },
            {"theme": "PCB", "classification": "observation_only", "actionable": False, "change_pct": -3.1},
            {"theme": "半导体", "classification": "observation_only", "actionable": False, "change_pct": -2.4},
            {"theme": "消费链", "classification": "rotation_or_defense", "actionable": False, "change_pct": 4.8},
        ]
        risk = {
            "raw_heat_score": 72,
            "raw_heat_label": "强势",
            "risk_state": "red",
            "position_mode": "defense_only",
            "position_cap_pct": 10,
            "triggered_gates": ["growth_index_below_short_ma", "external_tech_negative", "external_tech_veto"],
            "gate_evidence": ["成长指数位于MA5/MA10下方", "外围半导体显著下跌"],
        }
        snapshot = {
            "version": "a-share-market-review-snapshot-v1",
            "trade_date": fixture["trade_date"],
            "generated_at": fixture["generated_at"],
            "indices": indices,
            "breadth": {**fixture["market_stats"], "limit_up_count": None, "limit_down_count": None},
            "turnover": {"total_amount": 19300, "market_scope": "沪深口径"},
            "sentiment": fixture["canonical_sentiment"],
            "external_context": {
                "status": "ok",
                "as_of": "2026-07-29",
                "nasdaq_change_pct": -2.1,
                "semiconductor_proxy_change_pct": -4.5,
            },
            "theme_evidence": {"candidates": themes, "actionable_directions": themes[:1]},
            "stock_candidates": [
                {
                    "code": "603221",
                    "name": "爱丽家居",
                    "role": "emotion_thermometer",
                    "trade_eligibility": "observation_only",
                    "trigger_type": "无交易触发",
                    "confirmation_conditions": ["出现充分换手"],
                    "invalidation": "情绪高度下降",
                    "position_cap_pct": 0,
                }
            ],
            "risk_assessment": risk,
            "data_quality": {"status": "ok", "missing_fields": [], "contaminated_fields": [], "errors": []},
        }
        payload = {
            "version": 1,
            "kind": "market_review",
            "region": "cn",
            "language": "zh",
            "title": "2026-07-29 A股大盘复盘",
            "date": fixture["trade_date"],
            "data_date": fixture["trade_date"],
            "markdown_report": fixture["legacy_conflicting_report"],
            "sections": [{"key": "overview", "title": "Overview", "markdown": fixture["legacy_conflicting_report"]}],
            "normalized_review_snapshot": snapshot,
            "risk_assessment": risk,
            "a_share_evidence": {
                "sentiment_structure": fixture["canonical_sentiment"],
                "theme_candidates": themes,
                "stock_candidates": snapshot["stock_candidates"],
                "data_quality": {"status": "ok", "missing_fields": [], "contaminated_fields": [], "errors": []},
            },
            "market_light": {
                "region": "cn",
                "trade_date": fixture["trade_date"],
                "status": "red",
                "risk_state": "red",
                "score": 72,
                "label": "防守/不开高风险新仓",
                "temperature_label": "强势",
                "position_cap_pct": 10,
            },
        }
        return payload, fixture

    def test_20260729_consistency_guard_replaces_conflicting_markdown(self) -> None:
        from src.services.market_review_consistency import ensure_market_review_consistency

        payload, _ = self._build_20260729_conflicting_payload()
        guarded = ensure_market_review_consistency(payload)
        markdown = guarded["markdown_report"]

        self.assertEqual(guarded["consistency_validation"]["status"], "fallback")
        self.assertIn("涨停 81 家、炸板 14 家、跌停 9 家", markdown)
        self.assertIn("炸板率 14.74%", markdown)
        self.assertIn("中位数 +0.42%", markdown)
        self.assertNotIn("涨停83家", markdown)
        self.assertNotIn("仍未提供昨日涨停溢价", markdown)
        self.assertNotIn("可进攻", markdown)
        self.assertIn("风险状态：red", markdown)
        self.assertIn("组合仓位上限：10%", markdown)
        self.assertIn("CPO", markdown)
        self.assertIn("PCB", markdown)
        self.assertIn("半导体", markdown)
        self.assertIn("爱丽家居 | observation_only | 无交易触发", markdown)
        self.assertIn("A股当日新开仓受 T+1 约束", markdown)

    def test_20260729_payload_markdown_and_notification_share_guarded_risk_state(self) -> None:
        payload, fixture = self._build_20260729_conflicting_payload()
        notifier = self._make_notifier()
        market_analyzer = MagicMock()
        market_analyzer.run_daily_review_with_snapshot.return_value = SimpleNamespace(
            report=fixture["legacy_conflicting_report"],
            market_light_snapshot=payload["market_light"],
            structured_payload=payload,
        )

        with patch.object(market_review_module, "MarketAnalyzer", return_value=market_analyzer):
            result = run_market_review(
                notifier,
                config=SimpleNamespace(report_language="zh", market_review_region="cn"),
                send_notification=True,
                save_report_file=False,
                persist_history=False,
                return_structured=True,
            )

        self.assertIsInstance(result, market_review_module.MarketReviewRunResult)
        self.assertEqual(result.market_review_payload["risk_assessment"]["risk_state"], "red")
        self.assertIn("风险状态：red", result.report)
        notification_markdown = notifier.send.call_args.args[0]
        self.assertIn("风险状态：red", notification_markdown)
        self.assertNotIn("可进攻", notification_markdown)


if __name__ == "__main__":
    unittest.main()
