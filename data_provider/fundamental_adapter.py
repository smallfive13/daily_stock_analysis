# -*- coding: utf-8 -*-
"""
AkShare fundamental adapter (fail-open).

This adapter intentionally uses capability probing against multiple AkShare
endpoint candidates. It should never raise to caller; partial data is allowed.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

_DIVIDEND_KEYWORD_MAP: Dict[str, List[str]] = {
    "per_share": [
        "每股派息",
        "每股现金红利",
        "每股分红",
        "每股派现",
        "派现(元/股)",
        "派息(元/股)",
        "税前派息(元/股)",
        "现金分红(税前)",
    ],
    "plan_text": [
        "分配方案",
        "分红方案",
        "实施方案",
        "派息方案",
        "方案",
        "预案",
        "方案说明",
    ],
    "ex_dividend_date": ["除权除息日", "除息日", "除权日", "除权除息", "除息日期"],
    "record_date": ["股权登记日", "登记日"],
    "announce_date": ["公告日期", "公告日", "实施公告日", "预案公告日"],
    "report_date": ["报告期", "报告日期", "截止日期", "统计截止日期"],
}


def _safe_float(value: Any) -> Optional[float]:
    """Best-effort float conversion."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    s = str(value).strip().replace(",", "").replace("%", "")
    if not s:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    try:
        parsed = pd.to_datetime(value)
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    try:
        return parsed.to_pydatetime()
    except Exception:
        return None


def _normalize_code(raw: Any) -> str:
    s = _safe_str(raw).upper()
    if "." in s:
        s = s.split(".", 1)[0]
    s = re.sub(r"^(SH|SZ|BJ)", "", s)
    return s


def _pick_by_keywords(row: pd.Series, keywords: List[str]) -> Optional[Any]:
    """
    Return first non-empty row value whose column name contains any keyword.
    """
    for col in row.index:
        col_s = str(col)
        if any(k in col_s for k in keywords):
            val = row.get(col)
            if val is not None and str(val).strip() not in ("", "-", "nan", "None"):
                return val
    return None


def _parse_dividend_plan_to_per_share(plan_text: str) -> Optional[float]:
    """Parse per-share cash dividend from Chinese plan text."""
    text = _safe_str(plan_text)
    if not text:
        return None

    for pattern in (
        r"(?:每)?\s*10\s*股?\s*派(?:发)?\s*([0-9]+(?:\.[0-9]+)?)\s*元",
        r"10\s*派\s*([0-9]+(?:\.[0-9]+)?)\s*元",
    ):
        match = re.search(pattern, text)
        if match:
            parsed = _safe_float(match.group(1))
            if parsed is not None and parsed > 0:
                return parsed / 10.0

    match_per_share = re.search(r"每\s*股\s*派(?:发)?\s*([0-9]+(?:\.[0-9]+)?)\s*元", text)
    if match_per_share:
        parsed = _safe_float(match_per_share.group(1))
        if parsed is not None and parsed > 0:
            return parsed
    return None


def _extract_cash_dividend_per_share(row: pd.Series) -> Optional[float]:
    """Extract pre-tax cash dividend per share from a row."""
    plan_text = _safe_str(_pick_by_keywords(row, _DIVIDEND_KEYWORD_MAP["plan_text"]))
    # Keep pre-tax semantics; skip explicit after-tax plans unless pre-tax marker exists.
    if "税后" in plan_text and "税前" not in plan_text and "含税" not in plan_text:
        return None

    direct = _safe_float(_pick_by_keywords(row, _DIVIDEND_KEYWORD_MAP["per_share"]))
    if direct is not None and direct > 0:
        return direct
    return _parse_dividend_plan_to_per_share(plan_text)


def _filter_rows_by_code(df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    code_cols = [c for c in df.columns if any(k in str(c) for k in ("代码", "股票代码", "证券代码", "symbol", "ts_code"))]
    if not code_cols:
        return df

    target = _normalize_code(stock_code)
    for col in code_cols:
        try:
            series = df[col].astype(str).map(_normalize_code)
            filtered = df[series == target]
            if not filtered.empty:
                return filtered
        except Exception:
            continue
    return pd.DataFrame()


def _normalize_report_date(value: Any) -> Optional[str]:
    parsed = _safe_datetime(value)
    return parsed.date().isoformat() if parsed else None


def _percentile_rank(series: Any, current: float, min_points: int = 120) -> Optional[float]:
    """
    Percentile (0-100) of `current` within the positive values of a series.

    Returns None when the sample is too small to be statistically meaningful.
    """
    if series is None or current is None:
        return None
    try:
        values = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
        values = values[values > 0]
    except Exception:
        return None
    if len(values) < max(1, min_points):
        return None
    try:
        return round(float((values <= float(current)).mean() * 100.0), 1)
    except Exception:
        return None


def _recent_fridays(count: int) -> List[str]:
    """Most recent Fridays (inclusive of today if Friday), newest first, as YYYYMMDD."""
    today = datetime.now().date()
    offset = (today.weekday() - 4) % 7
    friday = today - timedelta(days=offset)
    return [(friday - timedelta(weeks=i)).strftime("%Y%m%d") for i in range(max(1, count))]


def _recent_weekdays_desc(count: int, before: Optional[datetime] = None) -> List[str]:
    """Most recent weekdays strictly before `before` (default now), newest first, as YYYYMMDD."""
    anchor = (before or datetime.now()).date()
    days: List[str] = []
    cursor = anchor - timedelta(days=1)
    while len(days) < max(1, count):
        if cursor.weekday() < 5:
            days.append(cursor.strftime("%Y%m%d"))
        cursor -= timedelta(days=1)
    return days


def _has_code_column(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False
    return any(
        any(k in str(c) for k in ("代码", "股票代码", "证券代码", "symbol", "ts_code"))
        for c in df.columns
    )


def _build_dividend_payload(
    dividend_df: pd.DataFrame,
    stock_code: str,
    max_events: int = 5,
) -> Dict[str, Any]:
    work_df = _filter_rows_by_code(dividend_df, stock_code)
    if work_df.empty:
        return {}

    now_date = datetime.now().date()
    ttm_start_date = now_date - timedelta(days=365)
    dedupe_keys = set()
    events: List[Dict[str, Any]] = []

    for _, row in work_df.iterrows():
        if not isinstance(row, pd.Series):
            continue
        ex_dt = _safe_datetime(_pick_by_keywords(row, _DIVIDEND_KEYWORD_MAP["ex_dividend_date"]))
        record_dt = _safe_datetime(_pick_by_keywords(row, _DIVIDEND_KEYWORD_MAP["record_date"]))
        announce_dt = _safe_datetime(_pick_by_keywords(row, _DIVIDEND_KEYWORD_MAP["announce_date"]))
        event_dt = ex_dt or record_dt or announce_dt
        if event_dt is None:
            continue
        event_date = event_dt.date()
        if event_date > now_date:
            continue

        per_share = _extract_cash_dividend_per_share(row)
        if per_share is None or per_share <= 0:
            continue

        dedupe_key = (event_date.isoformat(), round(per_share, 6))
        if dedupe_key in dedupe_keys:
            continue
        dedupe_keys.add(dedupe_key)

        events.append(
            {
                "event_date": event_date.isoformat(),
                "ex_dividend_date": ex_dt.date().isoformat() if ex_dt else None,
                "record_date": record_dt.date().isoformat() if record_dt else None,
                "announcement_date": announce_dt.date().isoformat() if announce_dt else None,
                "cash_dividend_per_share": round(per_share, 6),
                "is_pre_tax": True,
            }
        )

    if not events:
        return {}

    events.sort(key=lambda item: item.get("event_date") or "", reverse=True)
    ttm_events: List[Dict[str, Any]] = []
    for item in events:
        event_dt = _safe_datetime(item.get("event_date"))
        if event_dt is None:
            continue
        event_date = event_dt.date()
        if ttm_start_date <= event_date <= now_date:
            ttm_events.append(item)

    return {
        "events": events[:max(1, max_events)],
        "ttm_event_count": len(ttm_events),
        "ttm_cash_dividend_per_share": (
            round(sum(float(item.get("cash_dividend_per_share") or 0.0) for item in ttm_events), 6)
            if ttm_events else None
        ),
        "coverage": "cash_dividend_pre_tax",
        "as_of": now_date.isoformat(),
    }


def _extract_latest_row(df: pd.DataFrame, stock_code: str) -> Optional[pd.Series]:
    """
    Select the most relevant row for the given stock.
    """
    if df is None or df.empty:
        return None

    code_cols = [c for c in df.columns if any(k in str(c) for k in ("代码", "股票代码", "证券代码", "ts_code", "symbol"))]
    target = _normalize_code(stock_code)
    if code_cols:
        for col in code_cols:
            try:
                series = df[col].astype(str).map(_normalize_code)
                matched = df[series == target]
                if not matched.empty:
                    return matched.iloc[0]
            except Exception:
                continue
        return None

    # Fallback: use latest row
    return df.iloc[0]


class AkshareFundamentalAdapter:
    """AkShare adapter for fundamentals, capital flow and dragon-tiger signals."""

    def _call_df_candidates(
        self,
        candidates: List[Tuple[str, Dict[str, Any]]],
    ) -> Tuple[Optional[pd.DataFrame], Optional[str], List[str]]:
        errors: List[str] = []
        try:
            import akshare as ak
        except Exception as exc:
            return None, None, [f"import_akshare:{type(exc).__name__}"]

        for func_name, kwargs in candidates:
            fn = getattr(ak, func_name, None)
            if fn is None:
                continue
            try:
                df = fn(**kwargs)
                if isinstance(df, pd.Series):
                    df = df.to_frame().T
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df, func_name, errors
            except Exception as exc:
                errors.append(f"{func_name}:{type(exc).__name__}")
                continue
        return None, None, errors

    def get_fundamental_bundle(self, stock_code: str) -> Dict[str, Any]:
        """
        Return normalized fundamental blocks from AkShare with partial tolerance.
        """
        result: Dict[str, Any] = {
            "status": "not_supported",
            "growth": {},
            "earnings": {},
            "institution": {},
            "source_chain": [],
            "errors": [],
        }

        # Financial indicators
        fin_df, fin_source, fin_errors = self._call_df_candidates([
            ("stock_financial_abstract", {"symbol": stock_code}),
            ("stock_financial_analysis_indicator", {"symbol": stock_code}),
            ("stock_financial_analysis_indicator", {}),
        ])
        result["errors"].extend(fin_errors)
        if fin_df is not None:
            row = _extract_latest_row(fin_df, stock_code)
            if row is not None:
                revenue_yoy = _safe_float(_pick_by_keywords(row, ["营业收入同比", "营收同比", "收入同比", "同比增长"]))
                profit_yoy = _safe_float(_pick_by_keywords(row, ["净利润同比", "净利同比", "归母净利润同比"]))
                roe = _safe_float(_pick_by_keywords(row, ["净资产收益率", "ROE", "净资产收益"]))
                gross_margin = _safe_float(_pick_by_keywords(row, ["毛利率"]))
                report_date = _normalize_report_date(_pick_by_keywords(row, _DIVIDEND_KEYWORD_MAP["report_date"]))
                revenue = _safe_float(_pick_by_keywords(row, ["营业总收入", "营业收入", "营收"]))
                net_profit_parent = _safe_float(_pick_by_keywords(row, ["归母净利润", "母公司股东净利润", "净利润"]))
                operating_cash_flow = _safe_float(
                    _pick_by_keywords(row, ["经营活动产生的现金流量净额", "经营现金流", "经营活动现金流"])
                )
                result["growth"] = {
                    "revenue_yoy": revenue_yoy,
                    "net_profit_yoy": profit_yoy,
                    "roe": roe,
                    "gross_margin": gross_margin,
                }
                # 净现比：同一报表行内取值，单位一致；净利润为非正时比值语义失真，不计算
                ocf_to_net_profit_ratio = None
                if (
                    operating_cash_flow is not None
                    and net_profit_parent is not None
                    and net_profit_parent > 0
                ):
                    ocf_to_net_profit_ratio = round(operating_cash_flow / net_profit_parent, 4)
                financial_report_payload = {
                    "report_date": report_date,
                    "revenue": revenue,
                    "net_profit_parent": net_profit_parent,
                    "operating_cash_flow": operating_cash_flow,
                    "ocf_to_net_profit_ratio": ocf_to_net_profit_ratio,
                    "roe": roe,
                }
                if any(v is not None for v in financial_report_payload.values()):
                    result["earnings"]["financial_report"] = financial_report_payload
                result["source_chain"].append(f"growth:{fin_source}")

        # Earnings forecast
        forecast_df, forecast_source, forecast_errors = self._call_df_candidates([
            ("stock_yjyg_em", {"symbol": stock_code}),
            ("stock_yjyg_em", {}),
            ("stock_yjbb_em", {"symbol": stock_code}),
            ("stock_yjbb_em", {}),
        ])
        result["errors"].extend(forecast_errors)
        if forecast_df is not None:
            row = _extract_latest_row(forecast_df, stock_code)
            if row is not None:
                result["earnings"]["forecast_summary"] = _safe_str(
                    _pick_by_keywords(row, ["预告", "业绩变动", "内容", "摘要", "公告"])
                )[:200]
                result["source_chain"].append(f"earnings_forecast:{forecast_source}")

        # Earnings quick report
        quick_df, quick_source, quick_errors = self._call_df_candidates([
            ("stock_yjkb_em", {"symbol": stock_code}),
            ("stock_yjkb_em", {}),
        ])
        result["errors"].extend(quick_errors)
        if quick_df is not None:
            row = _extract_latest_row(quick_df, stock_code)
            if row is not None:
                result["earnings"]["quick_report_summary"] = _safe_str(
                    _pick_by_keywords(row, ["快报", "摘要", "公告", "说明"])
                )[:200]
                result["source_chain"].append(f"earnings_quick:{quick_source}")

        # Dividend details (cash dividend, pre-tax)
        dividend_df, dividend_source, dividend_errors = self._call_df_candidates([
            ("stock_fhps_detail_em", {"symbol": stock_code}),
            ("stock_history_dividend_detail", {"symbol": stock_code, "indicator": "分红", "date": ""}),
            ("stock_dividend_cninfo", {"symbol": stock_code}),
        ])
        result["errors"].extend(dividend_errors)
        if dividend_df is not None:
            dividend_payload = _build_dividend_payload(dividend_df, stock_code, max_events=5)
            if dividend_payload:
                result["earnings"]["dividend"] = dividend_payload
                result["source_chain"].append(f"dividend:{dividend_source}")

        # Institution / top shareholders
        inst_df, inst_source, inst_errors = self._call_df_candidates([
            ("stock_institute_hold", {}),
            ("stock_institute_recommend", {}),
        ])
        result["errors"].extend(inst_errors)
        if inst_df is not None:
            row = _extract_latest_row(inst_df, stock_code)
            if row is not None:
                inst_change = _safe_float(_pick_by_keywords(row, ["增减", "变化", "变动", "持股变化"]))
                result["institution"]["institution_holding_change"] = inst_change
                result["source_chain"].append(f"institution:{inst_source}")

        top10_df, top10_source, top10_errors = self._call_df_candidates([
            ("stock_gdfx_top_10_em", {"symbol": stock_code}),
            ("stock_gdfx_top_10_em", {}),
            ("stock_zh_a_gdhs_detail_em", {"symbol": stock_code}),
            ("stock_zh_a_gdhs_detail_em", {}),
        ])
        result["errors"].extend(top10_errors)
        if top10_df is not None:
            row = _extract_latest_row(top10_df, stock_code)
            if row is not None:
                holder_change = _safe_float(_pick_by_keywords(row, ["增减", "变化", "持股变化", "变动"]))
                result["institution"]["top10_holder_change"] = holder_change
                result["source_chain"].append(f"top10:{top10_source}")

        has_content = bool(result["growth"] or result["earnings"] or result["institution"])
        result["status"] = "partial" if has_content else "not_supported"
        return result

    def get_capital_flow(self, stock_code: str, top_n: int = 5) -> Dict[str, Any]:
        """
        Return stock + sector capital flow.
        """
        result: Dict[str, Any] = {
            "status": "not_supported",
            "stock_flow": {},
            "sector_rankings": {"top": [], "bottom": []},
            "source_chain": [],
            "errors": [],
        }

        stock_df, stock_source, stock_errors = self._call_df_candidates([
            ("stock_individual_fund_flow", {"stock": stock_code}),
            ("stock_individual_fund_flow", {"symbol": stock_code}),
            ("stock_individual_fund_flow", {}),
            ("stock_main_fund_flow", {"symbol": stock_code}),
            ("stock_main_fund_flow", {}),
        ])
        result["errors"].extend(stock_errors)
        if stock_df is not None:
            row = _extract_latest_row(stock_df, stock_code)
            if row is not None:
                net_inflow = _safe_float(_pick_by_keywords(row, ["主力净流入", "净流入", "净额"]))
                inflow_5d = _safe_float(_pick_by_keywords(row, ["5日", "五日"]))
                inflow_10d = _safe_float(_pick_by_keywords(row, ["10日", "十日"]))
                result["stock_flow"] = {
                    "main_net_inflow": net_inflow,
                    "inflow_5d": inflow_5d,
                    "inflow_10d": inflow_10d,
                }
                result["source_chain"].append(f"capital_stock:{stock_source}")

        sector_df, sector_source, sector_errors = self._call_df_candidates([
            ("stock_sector_fund_flow_rank", {}),
            ("stock_sector_fund_flow_summary", {}),
        ])
        result["errors"].extend(sector_errors)
        if sector_df is not None:
            name_col = next((c for c in sector_df.columns if any(k in str(c) for k in ("板块", "行业", "名称", "name"))), None)
            flow_col = next((c for c in sector_df.columns if any(k in str(c) for k in ("净流入", "主力", "flow", "净额"))), None)
            if name_col and flow_col:
                work_df = sector_df[[name_col, flow_col]].copy()
                work_df[flow_col] = pd.to_numeric(work_df[flow_col], errors="coerce")
                work_df = work_df.dropna(subset=[flow_col])
                top_df = work_df.nlargest(top_n, flow_col)
                bottom_df = work_df.nsmallest(top_n, flow_col)
                result["sector_rankings"] = {
                    "top": [{"name": _safe_str(r[name_col]), "net_inflow": float(r[flow_col])} for _, r in top_df.iterrows()],
                    "bottom": [{"name": _safe_str(r[name_col]), "net_inflow": float(r[flow_col])} for _, r in bottom_df.iterrows()],
                }
                result["source_chain"].append(f"capital_sector:{sector_source}")

        has_content = bool(result["stock_flow"] or result["sector_rankings"]["top"] or result["sector_rankings"]["bottom"])
        result["status"] = "partial" if has_content else "not_supported"
        return result

    def get_dragon_tiger_flag(self, stock_code: str, lookback_days: int = 20) -> Dict[str, Any]:
        """
        Return dragon-tiger signal in lookback window.
        """
        result: Dict[str, Any] = {
            "status": "not_supported",
            "is_on_list": False,
            "recent_count": 0,
            "latest_date": None,
            "source_chain": [],
            "errors": [],
        }

        df, source, errors = self._call_df_candidates([
            ("stock_lhb_stock_statistic_em", {}),
            ("stock_lhb_detail_em", {}),
            ("stock_lhb_jgmmtj_em", {}),
        ])
        result["errors"].extend(errors)
        if df is None:
            return result

        # Try code filter
        code_cols = [c for c in df.columns if any(k in str(c) for k in ("代码", "股票代码", "证券代码"))]
        target = _normalize_code(stock_code)
        matched = pd.DataFrame()
        for col in code_cols:
            try:
                series = df[col].astype(str).map(_normalize_code)
                cur = df[series == target]
                if not cur.empty:
                    matched = cur
                    break
            except Exception:
                continue
        if matched.empty:
            result["source_chain"].append(f"dragon_tiger:{source}")
            result["status"] = "ok" if code_cols else "partial"
            return result

        date_col = next((c for c in matched.columns if any(k in str(c) for k in ("日期", "上榜", "交易日", "time"))), None)
        parsed_dates: List[datetime] = []
        if date_col is not None:
            for val in matched[date_col].astype(str).tolist():
                try:
                    parsed_dates.append(pd.to_datetime(val).to_pydatetime())
                except Exception:
                    continue
        now = datetime.now()
        start = now - timedelta(days=max(1, lookback_days))
        recent_dates = [d for d in parsed_dates if start <= d <= now]

        result["is_on_list"] = bool(recent_dates)
        result["recent_count"] = len(recent_dates) if recent_dates else int(len(matched))
        result["latest_date"] = max(recent_dates).date().isoformat() if recent_dates else (
            max(parsed_dates).date().isoformat() if parsed_dates else None
        )
        result["status"] = "ok"
        result["source_chain"].append(f"dragon_tiger:{source}")
        return result

    def get_valuation_profile(self, stock_code: str) -> Dict[str, Any]:
        """
        Return valuation reference data: PE/PB historical percentiles and
        industry medians, so the LLM never judges valuation without anchors.

        Percentiles use the stock's own PE-TTM/PB daily history (positive
        values only). Industry medians come from East Money industry-board
        constituents; note their PE is the dynamic metric, recorded in
        `industry_pe_metric` to keep 口径 explicit.
        """
        result: Dict[str, Any] = {
            "status": "not_supported",
            "profile": {},
            "source_chain": [],
            "errors": [],
        }
        code = _normalize_code(stock_code)
        profile: Dict[str, Any] = {}

        hist_df, hist_source, hist_errors = self._call_df_candidates([
            ("stock_a_indicator_lg", {"symbol": code}),
            ("stock_a_lg_indicator", {"symbol": code}),
        ])
        result["errors"].extend(hist_errors)
        if hist_df is not None:
            date_col = next(
                (c for c in hist_df.columns if any(k in str(c).lower() for k in ("trade_date", "date", "日期"))),
                None,
            )
            lower_cols = {str(c).strip().lower(): c for c in hist_df.columns}
            pe_col = lower_cols.get("pe_ttm") or lower_cols.get("pe")
            pb_col = lower_cols.get("pb")
            if date_col is not None and (pe_col is not None or pb_col is not None):
                try:
                    work = hist_df.copy()
                    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
                    work = work.dropna(subset=[date_col]).sort_values(date_col)
                except Exception:
                    work = pd.DataFrame()
                if not work.empty:
                    latest = work.iloc[-1]
                    # Anchor windows on the data's own latest date to avoid clock drift.
                    anchor = latest[date_col]
                    current_pe = _safe_float(latest.get(pe_col)) if pe_col is not None else None
                    current_pb = _safe_float(latest.get(pb_col)) if pb_col is not None else None
                    if current_pe is not None and current_pe > 0:
                        profile["pe_ttm"] = round(current_pe, 2)
                    if current_pb is not None and current_pb > 0:
                        profile["pb"] = round(current_pb, 2)
                    for label, days in (("3y", 365 * 3), ("5y", 365 * 5)):
                        window = work[work[date_col] >= anchor - pd.Timedelta(days=days)]
                        if pe_col is not None and current_pe is not None and current_pe > 0:
                            profile[f"pe_percentile_{label}"] = _percentile_rank(window[pe_col], current_pe)
                        if pb_col is not None and current_pb is not None and current_pb > 0:
                            profile[f"pb_percentile_{label}"] = _percentile_rank(window[pb_col], current_pb)
                    try:
                        profile["history_as_of"] = anchor.date().isoformat()
                    except Exception:
                        pass
                    result["source_chain"].append(f"valuation_history:{hist_source}")

        info_df, _info_source, info_errors = self._call_df_candidates([
            ("stock_individual_info_em", {"symbol": code}),
        ])
        result["errors"].extend(info_errors)
        industry_name = ""
        if info_df is not None and len(info_df.columns) >= 2:
            item_col, value_col = list(info_df.columns)[0], list(info_df.columns)[1]
            for _, row in info_df.iterrows():
                if "行业" in _safe_str(row.get(item_col)):
                    industry_name = _safe_str(row.get(value_col))
                    break

        if industry_name:
            cons_df, cons_source, cons_errors = self._call_df_candidates([
                ("stock_board_industry_cons_em", {"symbol": industry_name}),
            ])
            result["errors"].extend(cons_errors)
            if cons_df is not None:
                profile["industry_name"] = industry_name
                pe_cons_col = next((c for c in cons_df.columns if "市盈率" in str(c)), None)
                pb_cons_col = next((c for c in cons_df.columns if "市净率" in str(c)), None)
                if pe_cons_col is not None:
                    pes = pd.to_numeric(cons_df[pe_cons_col], errors="coerce").dropna()
                    pes = pes[pes > 0]
                    if len(pes) >= 5:
                        profile["industry_pe_median"] = round(float(pes.median()), 2)
                        profile["industry_pe_sample_count"] = int(len(pes))
                        # East Money constituent tables expose dynamic PE, not TTM.
                        profile["industry_pe_metric"] = "pe_dynamic_positive_only"
                if pb_cons_col is not None:
                    pbs = pd.to_numeric(cons_df[pb_cons_col], errors="coerce").dropna()
                    pbs = pbs[pbs > 0]
                    if len(pbs) >= 5:
                        profile["industry_pb_median"] = round(float(pbs.median()), 2)
                result["source_chain"].append(f"valuation_industry:{cons_source}")

        result["profile"] = {k: v for k, v in profile.items() if v is not None}
        result["status"] = "partial" if result["profile"] else "not_supported"
        return result

    def get_structural_risk(self, stock_code: str, horizon_days: int = 90) -> Dict[str, Any]:
        """
        Return A-share structural risk data: upcoming restricted-share
        releases (解禁), share-pledge ratio (质押), and margin balance (两融).
        Fail-open: any sub-block may be missing; never raises.
        """
        result: Dict[str, Any] = {
            "status": "not_supported",
            "restricted_release": {},
            "pledge": {},
            "margin": {},
            "source_chain": [],
            "errors": [],
        }
        code = _normalize_code(stock_code)

        # -- 限售解禁（未来 horizon_days 天） --
        release_df, release_source, release_errors = self._call_df_candidates([
            ("stock_restricted_release_queue_em", {"symbol": code}),
            ("stock_restricted_release_queue_sina", {"symbol": code}),
        ])
        result["errors"].extend(release_errors)
        if release_df is not None:
            work = _filter_rows_by_code(release_df, code) if _has_code_column(release_df) else release_df
            date_col = next((c for c in work.columns if "解禁" in str(c) and any(k in str(c) for k in ("时间", "日期"))), None)
            ratio_col = next(
                (c for c in work.columns if "总股本" in str(c) and any(k in str(c) for k in ("占", "比例", "比率"))),
                None,
            )
            if date_col is not None:
                now_date = datetime.now().date()
                horizon_end = now_date + timedelta(days=max(1, horizon_days))
                events: List[Dict[str, Any]] = []
                for _, row in work.iterrows():
                    parsed = _safe_datetime(row.get(date_col))
                    if parsed is None:
                        continue
                    event_date = parsed.date()
                    if now_date <= event_date <= horizon_end:
                        ratio = _safe_float(row.get(ratio_col)) if ratio_col is not None else None
                        events.append({"date": event_date.isoformat(), "ratio_pct": ratio})
                events.sort(key=lambda item: item["date"])
                ratios = [e["ratio_pct"] for e in events if e["ratio_pct"] is not None]
                result["restricted_release"] = {
                    "window_days": max(1, horizon_days),
                    "event_count": len(events),
                    "next_release_date": events[0]["date"] if events else None,
                    "next_release_ratio_pct": events[0]["ratio_pct"] if events else None,
                    "total_ratio_pct": round(sum(ratios), 4) if ratios else None,
                    "as_of": now_date.isoformat(),
                }
                result["source_chain"].append(f"restricted_release:{release_source}")

        # -- 股权质押比例（周频数据，回溯最近几个周五） --
        for pledge_date in _recent_fridays(4):
            pledge_df, pledge_source, pledge_errors = self._call_df_candidates([
                ("stock_gpzy_pledge_ratio_em", {"date": pledge_date}),
            ])
            result["errors"].extend(pledge_errors)
            if pledge_df is None or not _has_code_column(pledge_df):
                continue
            matched = _filter_rows_by_code(pledge_df, code)
            if not matched.empty:
                ratio = _safe_float(_pick_by_keywords(matched.iloc[0], ["质押比例", "质押率"]))
                result["pledge"] = {
                    "pledge_ratio_pct": ratio,
                    "in_table": True,
                    "as_of": pledge_date,
                }
            else:
                # Whole-market table loaded but the stock is absent: usually no
                # pledge record that week; keep the distinction explicit.
                result["pledge"] = {
                    "pledge_ratio_pct": None,
                    "in_table": False,
                    "as_of": pledge_date,
                }
            result["source_chain"].append(f"pledge:{pledge_source}")
            break

        # -- 融资融券余额（按交易所路由；北交所等不支持则跳过） --
        margin_fn = None
        if code.startswith(("6", "9")):
            margin_fn = "stock_margin_detail_sse"
        elif code.startswith(("0", "3")):
            margin_fn = "stock_margin_detail_szse"
        if margin_fn is not None:
            latest_balance: Optional[float] = None
            latest_date: Optional[str] = None
            for margin_date in _recent_weekdays_desc(6):
                margin_df, margin_source, margin_errors = self._call_df_candidates([
                    (margin_fn, {"date": margin_date}),
                ])
                result["errors"].extend(margin_errors)
                if margin_df is None or not _has_code_column(margin_df):
                    continue
                matched = _filter_rows_by_code(margin_df, code)
                if matched.empty:
                    result["margin"] = {"is_margin_target": False, "as_of": margin_date}
                else:
                    latest_balance = _safe_float(_pick_by_keywords(matched.iloc[0], ["融资余额"]))
                    latest_date = margin_date
                    result["margin"] = {
                        "is_margin_target": True,
                        "margin_balance": latest_balance,
                        "as_of": margin_date,
                    }
                result["source_chain"].append(f"margin:{margin_source}")
                break

            if latest_balance is not None and latest_balance > 0 and latest_date is not None:
                try:
                    anchor = datetime.strptime(latest_date, "%Y%m%d")
                except ValueError:
                    anchor = None
                if anchor is not None:
                    # 约 5 个交易日前的余额，用于观察杠杆资金变化方向
                    for prev_date in _recent_weekdays_desc(3, before=anchor - timedelta(days=6)):
                        prev_df, _prev_source, prev_errors = self._call_df_candidates([
                            (margin_fn, {"date": prev_date}),
                        ])
                        result["errors"].extend(prev_errors)
                        if prev_df is None or not _has_code_column(prev_df):
                            continue
                        prev_matched = _filter_rows_by_code(prev_df, code)
                        if prev_matched.empty:
                            break
                        prev_balance = _safe_float(_pick_by_keywords(prev_matched.iloc[0], ["融资余额"]))
                        if prev_balance is not None and prev_balance > 0:
                            result["margin"]["margin_balance_prev"] = prev_balance
                            result["margin"]["prev_date"] = prev_date
                            result["margin"]["margin_balance_change_pct"] = round(
                                (latest_balance - prev_balance) / prev_balance * 100.0, 2
                            )
                        break

        has_content = bool(result["restricted_release"] or result["pledge"] or result["margin"])
        result["status"] = "partial" if has_content else "not_supported"
        return result
