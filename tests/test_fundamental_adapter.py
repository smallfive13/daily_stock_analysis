# -*- coding: utf-8 -*-
"""
Tests for fundamental adapter helpers.
"""

import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_provider.fundamental_adapter import (
    AkshareFundamentalAdapter,
    _build_dividend_payload,
    _extract_latest_row,
    _parse_dividend_plan_to_per_share,
)


class TestFundamentalAdapter(unittest.TestCase):
    def test_parse_dividend_plan_to_per_share_supports_cn_patterns(self) -> None:
        self.assertAlmostEqual(_parse_dividend_plan_to_per_share("10派3元(含税)"), 0.3, places=6)
        self.assertAlmostEqual(_parse_dividend_plan_to_per_share("每10股派发2.5元"), 0.25, places=6)
        self.assertAlmostEqual(_parse_dividend_plan_to_per_share("每股派0.8元"), 0.8, places=6)
        self.assertIsNone(_parse_dividend_plan_to_per_share("仅送股，不现金分红"))

    def test_extract_latest_row_returns_none_when_code_mismatch(self) -> None:
        df = pd.DataFrame(
            {
                "股票代码": ["600000", "000001"],
                "值": [1, 2],
            }
        )
        row = _extract_latest_row(df, "600519")
        self.assertIsNone(row)

    def test_extract_latest_row_fallback_when_no_code_column(self) -> None:
        df = pd.DataFrame({"值": [1, 2]})
        row = _extract_latest_row(df, "600519")
        self.assertIsNotNone(row)
        self.assertEqual(row["值"], 1)

    def test_dragon_tiger_no_match_with_code_column_is_ok(self) -> None:
        adapter = AkshareFundamentalAdapter()
        df = pd.DataFrame(
            {
                "股票代码": ["600000"],
                "日期": ["2026-01-01"],
            }
        )
        with patch.object(adapter, "_call_df_candidates", return_value=(df, "stock_lhb_stock_statistic_em", [])):
            result = adapter.get_dragon_tiger_flag("600519")
        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["is_on_list"])
        self.assertEqual(result["recent_count"], 0)

    def test_dragon_tiger_match_is_ok(self) -> None:
        adapter = AkshareFundamentalAdapter()
        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        df = pd.DataFrame(
            {
                "股票代码": ["600519"],
                "日期": [today],
            }
        )
        with patch.object(adapter, "_call_df_candidates", return_value=(df, "stock_lhb_stock_statistic_em", [])):
            result = adapter.get_dragon_tiger_flag("600519")
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["is_on_list"])
        self.assertGreaterEqual(result["recent_count"], 1)

    def test_fundamental_bundle_includes_financial_report_and_dividend_payload(self) -> None:
        adapter = AkshareFundamentalAdapter()
        now = datetime.now()
        within_ttm = (now - timedelta(days=30)).strftime("%Y-%m-%d")
        future_day = (now + timedelta(days=10)).strftime("%Y-%m-%d")
        old_day = (now - timedelta(days=500)).strftime("%Y-%m-%d")
        fin_df = pd.DataFrame(
            {
                "股票代码": ["600519"],
                "报告期": [within_ttm],
                "营业总收入": [1000.0],
                "归母净利润": [300.0],
                "经营活动产生的现金流量净额": [500.0],
                "净资产收益率": [18.2],
                "营业收入同比": [12.0],
                "净利润同比": [9.5],
            }
        )
        forecast_df = pd.DataFrame({"股票代码": ["600519"], "预告": ["预增"]})
        quick_df = pd.DataFrame({"股票代码": ["600519"], "快报": ["快报摘要"]})
        dividend_df = pd.DataFrame(
            {
                "股票代码": ["600519", "600519", "600519", "600519"],
                "除息日": [within_ttm, within_ttm, future_day, old_day],
                "分配方案": ["10派3元(含税)", "10派3元(含税)", "10派5元", "10派1元"],
            }
        )

        with patch.object(
            adapter,
            "_call_df_candidates",
            side_effect=[
                (fin_df, "stock_financial_abstract", []),
                (forecast_df, "stock_yjyg_em", []),
                (quick_df, "stock_yjkb_em", []),
                (dividend_df, "stock_fhps_detail_em", []),
                (None, None, []),
                (None, None, []),
            ],
        ):
            result = adapter.get_fundamental_bundle("600519")

        financial_report = result["earnings"].get("financial_report", {})
        self.assertEqual(financial_report.get("report_date"), within_ttm)
        self.assertEqual(financial_report.get("revenue"), 1000.0)
        self.assertEqual(financial_report.get("net_profit_parent"), 300.0)
        self.assertEqual(financial_report.get("operating_cash_flow"), 500.0)
        self.assertEqual(financial_report.get("roe"), 18.2)

        dividend_payload = result["earnings"].get("dividend", {})
        events = dividend_payload.get("events", [])
        self.assertEqual(len(events), 2)  # duplicate + future day filtered
        self.assertEqual(dividend_payload.get("ttm_event_count"), 1)
        self.assertAlmostEqual(dividend_payload.get("ttm_cash_dividend_per_share"), 0.3, places=6)

    def test_build_dividend_payload_returns_empty_when_code_not_matched(self) -> None:
        now = datetime.now().strftime("%Y-%m-%d")
        df = pd.DataFrame(
            {
                "股票代码": ["000001"],
                "除息日": [now],
                "分配方案": ["10派3元(含税)"],
            }
        )

        payload = _build_dividend_payload(df, stock_code="600519")
        self.assertEqual(payload, {})

    def test_build_dividend_payload_skips_after_tax_plan(self) -> None:
        now = datetime.now().strftime("%Y-%m-%d")
        df = pd.DataFrame(
            {
                "股票代码": ["600519"],
                "除息日": [now],
                "分配方案": ["10派3元(税后)"],
            }
        )

        payload = _build_dividend_payload(df, stock_code="600519")
        self.assertEqual(payload, {})

    def test_build_dividend_payload_ttm_window_boundary(self) -> None:
        now = datetime.now()
        day_365 = (now - timedelta(days=365)).strftime("%Y-%m-%d")
        day_366 = (now - timedelta(days=366)).strftime("%Y-%m-%d")
        df = pd.DataFrame(
            {
                "股票代码": ["600519", "600519"],
                "除息日": [day_365, day_366],
                "分配方案": ["10派3元(含税)", "10派5元(含税)"],
            }
        )

        payload = _build_dividend_payload(df, stock_code="600519")
        self.assertEqual(payload.get("ttm_event_count"), 1)
        self.assertAlmostEqual(payload.get("ttm_cash_dividend_per_share"), 0.3, places=6)

    def test_fundamental_bundle_computes_ocf_to_net_profit_ratio(self) -> None:
        adapter = AkshareFundamentalAdapter()
        fin_df = pd.DataFrame(
            {
                "股票代码": ["600519"],
                "报告期": ["2026-03-31"],
                "营业总收入": [1000.0],
                "归母净利润": [100.0],
                "经营活动产生的现金流量净额": [80.0],
                "净资产收益率": [15.0],
            }
        )

        def dispatch(candidates):
            if candidates[0][0] == "stock_financial_abstract":
                return fin_df, "stock_financial_abstract", []
            return None, None, []

        with patch.object(adapter, "_call_df_candidates", side_effect=dispatch):
            bundle = adapter.get_fundamental_bundle("600519")

        report = bundle["earnings"]["financial_report"]
        self.assertAlmostEqual(report["ocf_to_net_profit_ratio"], 0.8, places=6)

    def test_fundamental_bundle_skips_ocf_ratio_when_profit_non_positive(self) -> None:
        adapter = AkshareFundamentalAdapter()
        fin_df = pd.DataFrame(
            {
                "股票代码": ["600519"],
                "报告期": ["2026-03-31"],
                "归母净利润": [-50.0],
                "经营活动产生的现金流量净额": [80.0],
            }
        )

        def dispatch(candidates):
            if candidates[0][0] == "stock_financial_abstract":
                return fin_df, "stock_financial_abstract", []
            return None, None, []

        with patch.object(adapter, "_call_df_candidates", side_effect=dispatch):
            bundle = adapter.get_fundamental_bundle("600519")

        report = bundle["earnings"]["financial_report"]
        self.assertIsNone(report["ocf_to_net_profit_ratio"])


class TestValuationProfile(unittest.TestCase):
    def _build_history_df(self) -> pd.DataFrame:
        # ~4.1 年日频序列：3 年窗口内 PE 全部 <= 当前值（分位应为 100），
        # 更早区间 PE 明显更高（5 年分位应 < 100），用于验证窗口切分。
        dates = pd.date_range(end="2026-06-30", periods=1500, freq="D")
        # 旧值段留在 3 年窗口（含边界共 1096 点）之外，保证 3 年分位恰为 100
        recent_len = 1120
        old_len = 1500 - recent_len
        pe = [50.0] * old_len + list(pd.Series(range(recent_len)) / (recent_len - 1) * 10 + 10)
        pb = [8.0] * old_len + list(pd.Series(range(recent_len)) / (recent_len - 1) * 2 + 1)
        return pd.DataFrame({"trade_date": dates, "pe": pe, "pe_ttm": pe, "pb": pb})

    def test_valuation_profile_percentiles_and_industry(self) -> None:
        adapter = AkshareFundamentalAdapter()
        hist_df = self._build_history_df()
        info_df = pd.DataFrame(
            {
                "item": ["总市值", "行业", "上市时间"],
                "value": ["2万亿", "白酒", "2001-08-27"],
            }
        )
        cons_df = pd.DataFrame(
            {
                "名称": ["A", "B", "C", "D", "E", "F"],
                "市盈率-动态": [10.0, 20.0, 30.0, 40.0, 50.0, -5.0],
                "市净率": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )

        def dispatch(candidates):
            func = candidates[0][0]
            if func == "stock_a_indicator_lg":
                return hist_df, func, []
            if func == "stock_individual_info_em":
                return info_df, func, []
            if func == "stock_board_industry_cons_em":
                return cons_df, func, []
            return None, None, []

        with patch.object(adapter, "_call_df_candidates", side_effect=dispatch):
            result = adapter.get_valuation_profile("600519")

        profile = result["profile"]
        self.assertEqual(result["status"], "partial")
        self.assertEqual(profile["pe_percentile_3y"], 100.0)
        self.assertLess(profile["pe_percentile_5y"], 100.0)
        self.assertEqual(profile["industry_name"], "白酒")
        self.assertAlmostEqual(profile["industry_pe_median"], 30.0, places=2)
        self.assertEqual(profile["industry_pe_sample_count"], 5)
        self.assertAlmostEqual(profile["industry_pb_median"], 3.5, places=2)
        self.assertEqual(profile["history_as_of"], "2026-06-30")
        self.assertTrue(any(entry.startswith("valuation_history:") for entry in result["source_chain"]))

    def test_valuation_profile_not_supported_when_all_sources_fail(self) -> None:
        adapter = AkshareFundamentalAdapter()
        with patch.object(adapter, "_call_df_candidates", return_value=(None, None, ["boom:Timeout"])):
            result = adapter.get_valuation_profile("600519")
        self.assertEqual(result["status"], "not_supported")
        self.assertEqual(result["profile"], {})
        self.assertIn("boom:Timeout", result["errors"])


class TestStructuralRisk(unittest.TestCase):
    def test_structural_risk_release_pledge_and_margin(self) -> None:
        adapter = AkshareFundamentalAdapter()
        now = datetime.now()
        release_df = pd.DataFrame(
            {
                "解禁时间": [
                    (now - timedelta(days=10)).strftime("%Y-%m-%d"),
                    (now + timedelta(days=30)).strftime("%Y-%m-%d"),
                    (now + timedelta(days=200)).strftime("%Y-%m-%d"),
                ],
                "解禁股占总股本比例": [1.0, 2.5, 9.9],
            }
        )
        pledge_df = pd.DataFrame(
            {
                "股票代码": ["600519", "600000"],
                "质押比例": [12.3, 45.6],
            }
        )
        margin_latest_df = pd.DataFrame(
            {
                "标的证券代码": ["600519"],
                "融资余额": [5.0e9],
            }
        )
        margin_prev_df = pd.DataFrame(
            {
                "标的证券代码": ["600519"],
                "融资余额": [4.0e9],
            }
        )
        margin_calls = {"count": 0}

        def dispatch(candidates):
            func = candidates[0][0]
            if func == "stock_restricted_release_queue_em":
                return release_df, func, []
            if func == "stock_gpzy_pledge_ratio_em":
                return pledge_df, func, []
            if func == "stock_margin_detail_sse":
                margin_calls["count"] += 1
                if margin_calls["count"] == 1:
                    return margin_latest_df, func, []
                return margin_prev_df, func, []
            return None, None, []

        with patch.object(adapter, "_call_df_candidates", side_effect=dispatch):
            result = adapter.get_structural_risk("600519")

        self.assertEqual(result["status"], "partial")
        release = result["restricted_release"]
        self.assertEqual(release["event_count"], 1)
        self.assertEqual(release["next_release_date"], (now + timedelta(days=30)).date().isoformat())
        self.assertAlmostEqual(release["total_ratio_pct"], 2.5, places=4)

        pledge = result["pledge"]
        self.assertTrue(pledge["in_table"])
        self.assertAlmostEqual(pledge["pledge_ratio_pct"], 12.3, places=4)

        margin = result["margin"]
        self.assertTrue(margin["is_margin_target"])
        self.assertAlmostEqual(margin["margin_balance"], 5.0e9, places=2)
        self.assertAlmostEqual(margin["margin_balance_change_pct"], 25.0, places=2)

    def test_structural_risk_pledge_absent_and_not_margin_target(self) -> None:
        adapter = AkshareFundamentalAdapter()
        pledge_df = pd.DataFrame(
            {
                "股票代码": ["600000"],
                "质押比例": [45.6],
            }
        )
        margin_df = pd.DataFrame(
            {
                "标的证券代码": ["600000"],
                "融资余额": [1.0e9],
            }
        )

        def dispatch(candidates):
            func = candidates[0][0]
            if func == "stock_gpzy_pledge_ratio_em":
                return pledge_df, func, []
            if func == "stock_margin_detail_sse":
                return margin_df, func, []
            return None, None, []

        with patch.object(adapter, "_call_df_candidates", side_effect=dispatch):
            result = adapter.get_structural_risk("600519")

        self.assertEqual(result["status"], "partial")
        self.assertFalse(result["pledge"]["in_table"])
        self.assertIsNone(result["pledge"]["pledge_ratio_pct"])
        self.assertFalse(result["margin"]["is_margin_target"])
        self.assertEqual(result["restricted_release"], {})

    def test_structural_risk_not_supported_when_all_sources_fail(self) -> None:
        adapter = AkshareFundamentalAdapter()
        with patch.object(adapter, "_call_df_candidates", return_value=(None, None, [])):
            result = adapter.get_structural_risk("600519")
        self.assertEqual(result["status"], "not_supported")


if __name__ == "__main__":
    unittest.main()
