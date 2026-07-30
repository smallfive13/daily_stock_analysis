# -*- coding: utf-8 -*-
"""Deterministic consistency guard for normalized A-share market reviews."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from typing import Any, Dict, Iterable, List


_ACTIONABLE_THEME_CLASSES = {"confirmed_mainline", "mainline_candidate"}
_AGGRESSIVE_TERMS = ("强势，可进攻", "可进攻", "正常仓", "重仓", "积极加仓", "aggressive buy", "risk-on")
_BUY_TERMS = ("建议买入", "可买", "低吸", "弱转强买点", "回封买入", "加仓", "buy now", "buy on")
_NEGATIONS = ("不", "不得", "禁止", "避免", "仅观察", "无交易触发", "not ", "do not", "avoid")


def ensure_market_review_consistency(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a validated payload, replacing contradictory CN Markdown when needed."""

    result = copy.deepcopy(dict(payload))
    if str(result.get("region") or "").lower() != "cn":
        return result
    snapshot = result.get("normalized_review_snapshot")
    if not isinstance(snapshot, Mapping) or not snapshot:
        return result

    markdown = str(result.get("markdown_report") or "")
    errors = validate_market_review_consistency(result, markdown)
    if errors:
        _append_quality_errors(result, errors)
        snapshot = result.get("normalized_review_snapshot") or snapshot
        safe_markdown = render_safe_a_share_review(snapshot)
        result["markdown_report"] = safe_markdown
        result["sections"] = []
        result["title"] = f"{snapshot.get('trade_date') or result.get('date') or ''} A股大盘复盘".strip()
    result["consistency_validation"] = {
        "status": "fallback" if errors else "ok",
        "fallback_used": bool(errors),
        "errors": errors,
    }
    return result


def validate_market_review_consistency(
    payload: Mapping[str, Any],
    markdown: str,
) -> List[str]:
    snapshot = payload.get("normalized_review_snapshot")
    if not isinstance(snapshot, Mapping):
        return []
    errors: List[str] = []
    sentiment = snapshot.get("sentiment") if isinstance(snapshot.get("sentiment"), Mapping) else {}
    risk = snapshot.get("risk_assessment") if isinstance(snapshot.get("risk_assessment"), Mapping) else {}
    quality = snapshot.get("data_quality") if isinstance(snapshot.get("data_quality"), Mapping) else {}
    theme_evidence = snapshot.get("theme_evidence") if isinstance(snapshot.get("theme_evidence"), Mapping) else {}
    candidates = [item for item in (theme_evidence.get("candidates") or []) if isinstance(item, Mapping)]
    stocks = [item for item in (snapshot.get("stock_candidates") or []) if isinstance(item, Mapping)]

    metric_patterns = {
        "limit_up_count": (r"(?<!昨日)涨停\s*[:：]?\s*(\d+)\s*家", r"limit-up\s*[:：]?\s*(\d+)"),
        "broken_board_count": (r"炸板\s*[:：]?\s*(\d+)\s*家", r"failed boards?\s*[:：]?\s*(\d+)"),
        "limit_down_count": (r"跌停\s*[:：]?\s*(\d+)\s*家", r"limit-down\s*[:：]?\s*(\d+)"),
    }
    for field, patterns in metric_patterns.items():
        expected = sentiment.get(field)
        if expected is None:
            continue
        observed = _matched_ints(markdown, patterns)
        if any(value != int(expected) for value in observed):
            errors.append(f"metric_conflict:{field}:expected={int(expected)}:observed={sorted(set(observed))}")

    premium = sentiment.get("previous_limit_premium_median_pct")
    if premium is not None and re.search(
        r"(?:昨日涨停溢价.{0,18}(?:未提供|缺失|不可用)|(?:未提供|缺失|不可用).{0,18}昨日涨停溢价)",
        markdown,
    ):
        errors.append("provided_metric_described_missing:previous_limit_premium_median_pct")

    risk_state = str(risk.get("risk_state") or "yellow")
    if risk_state != "green" and any(_contains_positive_term(markdown, term) for term in _AGGRESSIVE_TERMS):
        errors.append(f"risk_language_conflict:risk_state={risk_state}")

    market_light = payload.get("market_light")
    if isinstance(market_light, Mapping):
        light_state = str(market_light.get("risk_state") or market_light.get("status") or "")
        if light_state and light_state != risk_state:
            errors.append(f"risk_state_mismatch:market_light={light_state}:snapshot={risk_state}")

    actionable = [item for item in candidates if item.get("classification") in _ACTIONABLE_THEME_CLASSES]
    if len(actionable) > 3:
        errors.append(f"too_many_actionable_themes:{len(actionable)}")
    if str(quality.get("status") or "unknown") != "ok" and re.search(
        r"(?:确定性第一主线|确认主线|confirmed mainline)",
        markdown,
        flags=re.IGNORECASE,
    ):
        errors.append("mainline_upgraded_with_incomplete_data")

    for stock in stocks:
        if stock.get("trade_eligibility") != "observation_only":
            continue
        identity = str(stock.get("name") or stock.get("code") or "").strip()
        if identity and _identity_has_buy_instruction(markdown, identity):
            errors.append(f"observation_only_buy_instruction:{identity}")

    index_rows = [item for item in (snapshot.get("indices") or []) if isinstance(item, Mapping)]
    covered_indices = [item for item in index_rows if item.get("change_pct") is not None]
    if len(covered_indices) < 5 and re.search(r"主要指数平均涨跌幅|average major-index change", markdown, re.IGNORECASE):
        errors.append(f"invalid_major_index_average:coverage={len(covered_indices)}")

    trade_date = str(snapshot.get("trade_date") or "")
    generated_date = str(snapshot.get("generated_at") or "")[:10]
    allowed_dates = {value for value in (trade_date, generated_date) if value}
    markdown_dates = set(re.findall(r"\b20\d{2}-\d{2}-\d{2}\b", markdown))
    unexpected_dates = sorted(markdown_dates - allowed_dates)
    if unexpected_dates:
        errors.append(f"date_mismatch:{unexpected_dates}")
    return _dedupe(errors)


def render_safe_a_share_review(snapshot: Mapping[str, Any]) -> str:
    trade_date = str(snapshot.get("trade_date") or "N/A")
    generated_at = str(snapshot.get("generated_at") or "N/A")
    breadth = snapshot.get("breadth") if isinstance(snapshot.get("breadth"), Mapping) else {}
    turnover = snapshot.get("turnover") if isinstance(snapshot.get("turnover"), Mapping) else {}
    sentiment = snapshot.get("sentiment") if isinstance(snapshot.get("sentiment"), Mapping) else {}
    risk = snapshot.get("risk_assessment") if isinstance(snapshot.get("risk_assessment"), Mapping) else {}
    quality = snapshot.get("data_quality") if isinstance(snapshot.get("data_quality"), Mapping) else {}
    external = snapshot.get("external_context") if isinstance(snapshot.get("external_context"), Mapping) else {}
    theme_evidence = snapshot.get("theme_evidence") if isinstance(snapshot.get("theme_evidence"), Mapping) else {}
    candidates = [item for item in (theme_evidence.get("candidates") or []) if isinstance(item, Mapping)]
    actionable = [item for item in candidates if item.get("classification") in _ACTIONABLE_THEME_CLASSES][:3]
    downgraded = [item for item in candidates if item not in actionable]
    stocks = [item for item in (snapshot.get("stock_candidates") or []) if isinstance(item, Mapping)][:10]
    missing = [str(value) for value in (quality.get("missing_fields") or [])]
    contaminated = [str(value) for value in (quality.get("contaminated_fields") or [])]
    quality_errors = [str(value) for value in (quality.get("errors") or [])]
    triggered_gates = [str(value) for value in (risk.get("triggered_gates") or [])]

    lines = [
        f"## {trade_date} A股大盘复盘",
        "",
        "### 一、数据口径与失败项",
        f"- 交易日：{trade_date}；生成时间：{generated_at}。",
        (
            "- 市场宽度：上涨 {up} 家、下跌 {down} 家、平盘 {flat} 家。".format(
                up=_value_or_na(breadth.get("up_count")),
                down=_value_or_na(breadth.get("down_count")),
                flat=_value_or_na(breadth.get("flat_count")),
            )
        ),
        (
            "- 日期化情绪口径：涨停 {limit_up} 家、炸板 {broken} 家、跌停 {limit_down} 家，"
            "炸板率 {broken_ratio}；昨日涨停溢价样本 {sample} 只，均值 {premium_mean}，中位数 {premium_median}；"
            "最高 {height} 板。"
        ).format(
            limit_up=_value_or_na(sentiment.get("limit_up_count")),
            broken=_value_or_na(sentiment.get("broken_board_count")),
            limit_down=_value_or_na(sentiment.get("limit_down_count")),
            broken_ratio=_format_ratio(sentiment.get("broken_ratio")),
            sample=_value_or_na(sentiment.get("previous_limit_sample_count")),
            premium_mean=_format_pct(sentiment.get("previous_limit_premium_mean_pct")),
            premium_median=_format_pct(sentiment.get("previous_limit_premium_median_pct")),
            height=_value_or_na(sentiment.get("highest_consecutive_board")),
        ),
        f"- 成交额：{_value_or_na(turnover.get('total_amount'))} 亿元；范围：{turnover.get('market_scope') or '未标记'}。",
        f"- 数据质量：{quality.get('status') or 'unknown'}；缺失：{_join_or_none(missing)}；不可用：{_join_or_none(contaminated)}。",
        "",
        "### 二、市场热度与风险状态",
        f"- 市场热度：{risk.get('raw_heat_score', 'N/A')}/100（{risk.get('raw_heat_label') or 'unknown'}）。",
        f"- 风险状态：{risk.get('risk_state') or 'yellow'}；仓位模式：{risk.get('position_mode') or 'confirmation_trial'}；组合仓位上限：{risk.get('position_cap_pct', 30)}%。",
        "- 热度描述普涨与活跃度，风险状态决定能否开仓；普涨修复不等于趋势修复。",
        "",
        "### 三、指数、旧主线与外围风险",
        "| 指数 | 涨跌幅 | 现价 | MA5 | MA10 | MA20 | 当前来源 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in snapshot.get("indices") or []:
        if not isinstance(item, Mapping):
            continue
        lines.append(
            "| {name} | {change} | {current} | {ma5} | {ma10} | {ma20} | {source} |".format(
                name=item.get("name") or item.get("code") or "N/A",
                change=_format_pct(item.get("change_pct")),
                current=_format_number(item.get("current")),
                ma5=_format_number(item.get("ma5")),
                ma10=_format_number(item.get("ma10")),
                ma20=_format_number(item.get("ma20")),
                source=item.get("current_source") or "unknown",
            )
        )
    lines.extend(
        [
            f"- 外围科技：{external.get('status') or 'missing'}；纳指 {_format_pct(external.get('nasdaq_change_pct'))}；半导体代理 {_format_pct(external.get('semiconductor_proxy_change_pct'))}；时间 {external.get('as_of') or 'N/A'}。",
            "- 科技分支按证据分别处理：" + _technology_branch_summary(candidates),
            "- 已触发门槛：" + _join_or_none(triggered_gates),
            "",
            "### 四、不能买什么",
        ]
    )
    if "external_tech_veto" in triggered_gates:
        lines.append("- 不买外围科技否决下的高位科技反抽，不把 CPO 单分支修复外推为 PCB、半导体同步修复。")
    if any(stock.get("trade_eligibility") == "observation_only" for stock in stocks):
        lines.append("- 不买一字或严重缩量高标；这些标的只作情绪温度计，无交易触发。")
    lines.extend(
        [
            "- 不买缺少容量核心、只有单日排行或后排扩散证据的方向。",
            "",
            "### 五、可操作方向",
        ]
    )
    if actionable:
        for index, item in enumerate(actionable, 1):
            lines.append(
                f"{index}. {item.get('theme') or 'N/A'}：{item.get('classification')}；确认={item.get('confirmation') or 'N/A'}；失效={item.get('invalidation') or 'N/A'}；方向仓位上限 {item.get('position_cap_pct', 0)}%。"
            )
    else:
        lines.append("- 当前没有通过结构化门槛的可操作方向，保持板块级观察。")
    if downgraded:
        lines.append(
            "- 降级观察："
            + "；".join(
                f"{item.get('theme') or 'N/A'}({item.get('classification') or 'observation_only'})"
                for item in downgraded[:8]
            )
            + "。"
        )
    lines.extend(
        [
            "",
            "### 六、转为试错或进攻的确认条件",
            "- 五个主要指数均有同交易日日线，创业板指与科创50收盘重新站上 MA5 和 MA10。",
            "- 日期化炸板率不高于30%，昨日涨停溢价中位数不低于0%。",
            "- 可操作方向的前排完成换手，容量核心不跌破参考交易日低点且板块继续强于沪深300。",
            "- 纳指与半导体代理不再触发显著负反馈，`external_tech_veto` 解除。",
            "",
            "### 七、核心观察票与交易资格",
            "| 角色 | 标的 | 资格 | 触发 | 确认 | 失效 | 个股仓位上限 |",
            "| --- | --- | --- | --- | --- | --- | ---: |",
        ]
    )
    if stocks:
        for item in stocks:
            conditions = item.get("confirmation_conditions") or []
            lines.append(
                "| {role} | {identity} | {eligibility} | {trigger} | {confirmation} | {invalidation} | {cap}% |".format(
                    role=item.get("role") or item.get("category") or "watchlist",
                    identity=" ".join(str(value) for value in (item.get("code"), item.get("name")) if value),
                    eligibility=item.get("trade_eligibility") or "conditional",
                    trigger=item.get("trigger_type") or "N/A",
                    confirmation="；".join(str(value) for value in conditions) or item.get("validation") or "N/A",
                    invalidation=item.get("invalidation_level") or item.get("invalidation") or "N/A",
                    cap=item.get("position_cap_pct", 0),
                )
            )
    else:
        lines.append("| 板块观察 | 暂无可验证观察票 | observation_only | 无交易触发 | 等待结构化个股证据 | N/A | 0% |")
    lines.extend(
        [
            "",
            "### 八、仓位模式与仓位上限",
            f"- 新开仓按 {risk.get('position_mode') or 'confirmation_trial'} 执行，组合仓位不超过 {risk.get('position_cap_pct', 30)}%；单票不得超过表内上限。",
            "- A股当日新开仓受 T+1 约束，入场前必须接受当日无法卖出的流动性风险；已有持仓处置与新开仓计划分开执行。",
            "",
            "### 九、风险提示与未验证项",
            f"- 风险门槛证据：{_join_or_none(risk.get('gate_evidence') or [])}。",
            f"- 未验证或失败项：{_join_or_none(missing + contaminated + quality_errors)}。",
            "- 建议仅供参考，不构成投资建议。",
        ]
    )
    return "\n".join(lines).strip()


def _append_quality_errors(payload: Dict[str, Any], errors: List[str]) -> None:
    snapshot = payload.get("normalized_review_snapshot")
    if isinstance(snapshot, dict):
        quality = snapshot.setdefault("data_quality", {})
        existing = quality.setdefault("errors", [])
        quality["errors"] = _dedupe([str(value) for value in existing] + errors)
        if quality.get("status") == "ok":
            quality["status"] = "partial"
    evidence = payload.get("a_share_evidence")
    if isinstance(evidence, dict):
        quality = evidence.setdefault("data_quality", {})
        existing = quality.setdefault("errors", [])
        quality["errors"] = _dedupe([str(value) for value in existing] + errors)
        if quality.get("status") == "ok":
            quality["status"] = "partial"


def _matched_ints(markdown: str, patterns: Iterable[str]) -> List[int]:
    values: List[int] = []
    for pattern in patterns:
        values.extend(int(value) for value in re.findall(pattern, markdown, flags=re.IGNORECASE))
    return values


def _contains_positive_term(text: str, term: str) -> bool:
    lowered = text.lower()
    target = term.lower()
    start = 0
    while True:
        index = lowered.find(target, start)
        if index < 0:
            return False
        prefix = lowered[max(0, index - 14):index]
        if not any(negation.lower() in prefix for negation in _NEGATIONS):
            return True
        start = index + len(target)


def _identity_has_buy_instruction(text: str, identity: str) -> bool:
    for match in re.finditer(re.escape(identity), text):
        window = text[max(0, match.start() - 20): min(len(text), match.end() + 100)]
        if any(_contains_positive_term(window, term) for term in _BUY_TERMS):
            return True
    return False


def _technology_branch_summary(candidates: List[Mapping[str, Any]]) -> str:
    branches = []
    for item in candidates:
        name = str(item.get("theme") or "")
        if any(token in name.upper() for token in ("CPO", "PCB")) or "半导体" in name:
            branches.append(
                f"{name} 涨跌幅 {_format_pct(item.get('change_pct'))}、分类 {item.get('classification') or 'observation_only'}"
            )
    return "；".join(branches) + "。" if branches else "当前缺少 CPO、PCB、半导体分支证据，不作整体同向判断。"


def _format_ratio(value: Any) -> str:
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "N/A"


def _format_pct(value: Any) -> str:
    try:
        return f"{float(value):+.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def _format_number(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "N/A"


def _value_or_na(value: Any) -> Any:
    return "N/A" if value is None else value


def _join_or_none(values: Iterable[Any]) -> str:
    normalized = [str(value) for value in values if str(value).strip()]
    return "、".join(normalized) if normalized else "无"


def _dedupe(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result
