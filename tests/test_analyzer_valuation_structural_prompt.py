# -*- coding: utf-8 -*-
"""Tests for A-share valuation-reference and structural-risk prompt sections."""

import unittest
from unittest.mock import patch

try:
    import litellm  # noqa: F401
except ModuleNotFoundError:
    from tests.litellm_stub import ensure_litellm_stub

    ensure_litellm_stub()

from src.analyzer import GeminiAnalyzer


def _build_analyzer() -> GeminiAnalyzer:
    with patch.object(GeminiAnalyzer, "_init_litellm", return_value=None):
        return GeminiAnalyzer(
            skill_instructions="",
            default_skill_policy="",
            use_legacy_default_prompt=False,
        )


def _base_context(**overrides):
    context = {
        "code": "600519",
        "stock_name": "贵州茅台",
        "date": "2026-07-03",
        "today": {},
        "news_window_days": 3,
    }
    context.update(overrides)
    return context


class ValuationReferencePromptTestCase(unittest.TestCase):
    def test_valuation_reference_table_renders_with_profile_data(self) -> None:
        analyzer = _build_analyzer()
        context = _base_context(
            fundamental_context={
                "market": "cn",
                "valuation": {
                    "status": "ok",
                    "data": {
                        "pe_ratio": 22.0,
                        "pe_ttm": 21.5,
                        "pb": 6.8,
                        "pe_percentile_3y": 45.0,
                        "pe_percentile_5y": 38.2,
                        "pb_percentile_3y": 51.3,
                        "pb_percentile_5y": 47.9,
                        "history_as_of": "2026-07-02",
                        "industry_name": "白酒",
                        "industry_pe_median": 24.6,
                        "industry_pe_sample_count": 18,
                        "industry_pb_median": 4.2,
                    },
                },
            }
        )

        prompt = analyzer._format_prompt(context, "贵州茅台", news_context=None)

        self.assertIn("估值参照（历史分位与行业对比）", prompt)
        self.assertIn("PE 3年分位", prompt)
        self.assertIn("45.0%", prompt)
        self.assertIn("白酒", prompt)
        self.assertIn("行业PE中位数", prompt)
        self.assertIn("样本 18 只", prompt)
        self.assertIn("估值高低判断必须引用上表", prompt)
        self.assertNotIn("严禁凭经验或训练记忆", prompt)

    def test_valuation_reference_missing_forbids_valuation_judgement(self) -> None:
        analyzer = _build_analyzer()
        context = _base_context(
            fundamental_context={
                "market": "cn",
                "valuation": {"status": "partial", "data": {"pe_ratio": 22.0}},
            }
        )

        prompt = analyzer._format_prompt(context, "贵州茅台", news_context=None)

        self.assertIn("估值参照（历史分位与行业对比）", prompt)
        self.assertIn("本次未获取到估值参照数据", prompt)
        self.assertIn("严禁凭经验或训练记忆", prompt)

    def test_valuation_reference_missing_note_when_no_fundamental_context(self) -> None:
        analyzer = _build_analyzer()
        prompt = analyzer._format_prompt(_base_context(), "贵州茅台", news_context=None)

        self.assertIn("估值参照（历史分位与行业对比）", prompt)
        self.assertIn("严禁凭经验或训练记忆", prompt)

    def test_sections_skipped_for_non_cn_and_etf(self) -> None:
        analyzer = _build_analyzer()

        us_context = _base_context(code="AAPL", stock_name="Apple")
        prompt_us = analyzer._format_prompt(us_context, "Apple", news_context=None)
        self.assertNotIn("估值参照（历史分位与行业对比）", prompt_us)
        self.assertNotIn("结构性风险（A股制度性因素）", prompt_us)

        etf_context = _base_context(is_index_etf=True)
        prompt_etf = analyzer._format_prompt(etf_context, "沪深300ETF", news_context=None)
        self.assertNotIn("估值参照（历史分位与行业对比）", prompt_etf)
        self.assertNotIn("结构性风险（A股制度性因素）", prompt_etf)


class StructuralRiskPromptTestCase(unittest.TestCase):
    def test_structural_risk_table_renders(self) -> None:
        analyzer = _build_analyzer()
        context = _base_context(
            fundamental_context={
                "market": "cn",
                "structural_risk": {
                    "status": "ok",
                    "data": {
                        "restricted_release": {
                            "window_days": 90,
                            "event_count": 2,
                            "next_release_date": "2026-08-01",
                            "next_release_ratio_pct": 1.2,
                            "total_ratio_pct": 3.7,
                            "as_of": "2026-07-03",
                        },
                        "pledge": {
                            "pledge_ratio_pct": 12.3,
                            "in_table": True,
                            "as_of": "20260703",
                        },
                        "margin": {
                            "is_margin_target": True,
                            "margin_balance": 5.24e9,
                            "margin_balance_prev": 4.0e9,
                            "prev_date": "20260626",
                            "margin_balance_change_pct": 31.0,
                            "as_of": "20260703",
                        },
                    },
                },
            }
        )

        prompt = analyzer._format_prompt(context, "贵州茅台", news_context=None)

        self.assertIn("结构性风险（A股制度性因素）", prompt)
        self.assertIn("未来90天限售解禁", prompt)
        self.assertIn("2 次，最近 2026-08-01", prompt)
        self.assertIn("合计约占总股本 3.7%", prompt)
        self.assertIn("12.3%", prompt)
        self.assertIn("52.40 亿元", prompt)
        self.assertIn("+31.00%", prompt)
        self.assertIn("禁止编造解禁日期、质押比例或两融数据", prompt)

    def test_structural_risk_no_events_and_not_margin_target(self) -> None:
        analyzer = _build_analyzer()
        context = _base_context(
            fundamental_context={
                "market": "cn",
                "structural_risk": {
                    "status": "ok",
                    "data": {
                        "restricted_release": {
                            "window_days": 90,
                            "event_count": 0,
                            "next_release_date": None,
                            "total_ratio_pct": None,
                            "as_of": "2026-07-03",
                        },
                        "pledge": {"pledge_ratio_pct": None, "in_table": False, "as_of": "20260703"},
                        "margin": {"is_margin_target": False, "as_of": "20260703"},
                    },
                },
            }
        )

        prompt = analyzer._format_prompt(context, "贵州茅台", news_context=None)

        self.assertIn("数据源口径内无解禁事件", prompt)
        self.assertIn("未见于全市场质押统计", prompt)
        self.assertIn("非两融标的", prompt)

    def test_structural_risk_missing_note(self) -> None:
        analyzer = _build_analyzer()
        prompt = analyzer._format_prompt(_base_context(), "贵州茅台", news_context=None)

        self.assertIn("结构性风险（A股制度性因素）", prompt)
        self.assertIn("本次未获取到解禁/质押/两融数据", prompt)
        self.assertIn("不得声称“无解禁压力”“无质押风险”", prompt)


class FinancialQualityPromptTestCase(unittest.TestCase):
    def test_financial_table_includes_growth_and_ocf_ratio(self) -> None:
        analyzer = _build_analyzer()
        context = _base_context(
            fundamental_context={
                "market": "cn",
                "earnings": {
                    "data": {
                        "financial_report": {
                            "report_date": "2026-03-31",
                            "revenue": 1000.0,
                            "net_profit_parent": 100.0,
                            "operating_cash_flow": 85.0,
                            "ocf_to_net_profit_ratio": 0.85,
                            "roe": 15.0,
                        },
                        "dividend": {
                            "ttm_cash_dividend_per_share": 1.2,
                            "ttm_dividend_yield_pct": 2.4,
                            "ttm_event_count": 1,
                        },
                    }
                },
                "growth": {
                    "data": {"revenue_yoy": 12.5, "net_profit_yoy": 8.0},
                },
            }
        )

        prompt = analyzer._format_prompt(context, "贵州茅台", news_context=None)

        self.assertIn("净现比", prompt)
        self.assertIn("0.85", prompt)
        self.assertIn("营收同比", prompt)
        self.assertIn("12.5", prompt)
        self.assertIn("归母净利同比", prompt)


class SystemPromptValuationConstraintTestCase(unittest.TestCase):
    def test_system_prompt_requires_valuation_reference(self) -> None:
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                with patch.object(GeminiAnalyzer, "_init_litellm", return_value=None):
                    analyzer = GeminiAnalyzer(
                        skill_instructions="",
                        default_skill_policy="",
                        use_legacy_default_prompt=legacy,
                    )
                prompt = analyzer._get_analysis_system_prompt("zh", stock_code="600519")
                self.assertIn("估值高低判断必须引用输入中的“估值参照”数据", prompt)
                self.assertIn("禁止输出“估值合理/偏低/偏高”类结论", prompt)


if __name__ == "__main__":
    unittest.main()
