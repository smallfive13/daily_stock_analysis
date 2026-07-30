# -*- coding: utf-8 -*-
"""Build structured A-share review evidence from DSA-native providers."""

from __future__ import annotations

from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.schemas.a_share_review import (
    AShareExternalTechContext,
    AShareIndexTrend,
    AShareReviewDataQuality,
    AShareReviewEvidence,
    AShareReviewRiskRule,
    AShareReviewSource,
    AShareSentimentStructure,
    AShareStockCandidate,
    AShareThemeCandidate,
    dump_a_share_review_model,
)
from src.services.market_hotspot_service import MarketHotspotService

CN_INDEX_TREND_TARGETS = (
    {"symbol": "000001", "name": "上证指数", "aliases": ("000001", "sh000001", "上证指数")},
    {"symbol": "399001", "name": "深证成指", "aliases": ("399001", "sz399001", "深证成指")},
    {"symbol": "399006", "name": "创业板指", "aliases": ("399006", "sz399006", "创业板指")},
    {"symbol": "000688", "name": "科创50", "aliases": ("000688", "sh000688", "科创50")},
    {"symbol": "000300", "name": "沪深300", "aliases": ("000300", "sh000300", "沪深300")},
)

ACTIONABLE_THEME_CLASSES = frozenset({"confirmed_mainline", "mainline_candidate"})
THEME_GROUP_ALIASES = {
    "化学制品": "化工链",
    "化学原料": "化工链",
    "基础化工": "化工链",
    "乳品": "消费链",
    "饮料乳品": "消费链",
    "休闲食品": "消费链",
    "一般零售": "消费链",
}

POOL_DATASETS = {
    "limit_up": "limit_up_pool",
    "broken_board": "broken_board_pool",
    "previous_limit": "previous_limit_pool",
    "limit_down": "limit_down_pool",
}


def compute_index_key_levels(bars: List[Dict[str, Any]], current: Any = None) -> Dict[str, Any]:
    """Compute MA5/MA10/MA20 and 20-day range from ascending daily bars."""
    if not bars or len(bars) < 20:
        return {}

    recent_bars = bars[-20:]
    try:
        closes = [float(bar["close"]) for bar in recent_bars]
        highs = [float(bar["high"]) for bar in recent_bars]
        lows = [float(bar["low"]) for bar in recent_bars]
    except (KeyError, TypeError, ValueError):
        return {}

    ma5 = mean(closes[-5:])
    ma10 = mean(closes[-10:])
    ma20 = mean(closes)
    current_value = _safe_float(current)
    return {
        "ma5": round(ma5, 2),
        "ma10": round(ma10, 2),
        "ma20": round(ma20, 2),
        "dist_ma5_pct": _distance_pct(current_value, ma5),
        "dist_ma10_pct": _distance_pct(current_value, ma10),
        "dist_ma20_pct": _distance_pct(current_value, ma20),
        "high_20d": round(max(highs), 2),
        "low_20d": round(min(lows), 2),
    }


def aggregate_limit_up_pool(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Keep the legacy limit-up structure contract from normalized pool rows."""
    if not rows:
        return {}

    industry_counts: Dict[str, int] = {}
    max_consecutive_boards = 0
    max_boards_stock = ""
    total_break_count = 0
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        industry = _text(row.get("industry"))
        if industry:
            industry_counts[industry] = industry_counts.get(industry, 0) + 1
        boards = _safe_int(row.get("consecutive_boards")) or 0
        if boards > max_consecutive_boards:
            max_consecutive_boards = boards
            max_boards_stock = _text(row.get("name"))
        total_break_count += _safe_int(row.get("break_count")) or 0

    return {
        "total": len(rows),
        "industry_distribution": [
            {"industry": name, "count": count}
            for name, count in sorted(industry_counts.items(), key=lambda item: (-item[1], item[0]))[:8]
        ],
        "max_consecutive_boards": max_consecutive_boards,
        "max_boards_stock": max_boards_stock,
        "total_break_count": total_break_count,
    }


class AShareReviewEvidenceService:
    """Collect and compress A-share review evidence without invoking an LLM."""

    def __init__(self, fetcher_manager: Any, *, pool_limit: int = 200) -> None:
        self.fetcher_manager = fetcher_manager
        self.pool_limit = max(20, int(pool_limit or 200))
        self.hotspot_service = MarketHotspotService(fetcher_manager=fetcher_manager)

    def build(
        self,
        *,
        trade_date: str,
        indices: Sequence[Any],
        sector_rankings: Mapping[str, Any],
        concept_rankings: Mapping[str, Any],
        market_snapshot: Optional[Mapping[str, Any]] = None,
        global_indices: Sequence[Any] = (),
    ) -> Dict[str, Any]:
        pools: Dict[str, List[Dict[str, Any]]] = {}
        pool_available: Dict[str, bool] = {}
        sources: List[AShareReviewSource] = []
        errors: List[str] = []
        missing_fields: List[str] = []

        for pool_type, dataset in POOL_DATASETS.items():
            rows, available, pool_sources, pool_errors = self._fetch_pool(pool_type, trade_date)
            pools[pool_type] = rows
            pool_available[pool_type] = available
            sources.extend(pool_sources)
            errors.extend(pool_errors)
            if not available:
                missing_fields.append(dataset)

        index_trend, index_sources, index_errors = self._build_index_trend(
            indices,
            trade_date=trade_date,
        )
        sources.extend(index_sources)
        errors.extend(index_errors)
        covered_index_codes = {item.code for item in index_trend}
        for target in CN_INDEX_TREND_TARGETS:
            if target["symbol"] not in covered_index_codes:
                missing_fields.append(f"index_trend:{target['symbol']}")
        for item in index_trend:
            if item.status != "ok":
                missing_fields.append(f"index_trend_as_of:{item.code}")

        hotspot_payload = self.hotspot_service.get_hotspots(
            market="cn",
            trade_date=trade_date,
            limit=10,
            sector_rankings=dict(sector_rankings),
            concept_rankings=dict(concept_rankings),
        )
        hotspot_quality = hotspot_payload.get("data_quality") if isinstance(hotspot_payload, Mapping) else None
        if isinstance(hotspot_quality, Mapping):
            for item in hotspot_quality.get("errors") or []:
                errors.append(f"market_hotspots: {item}")
            for item in hotspot_quality.get("missing_fields") or []:
                missing_fields.append(str(item))

        sentiment = self._build_sentiment(pools, pool_available)
        themes = self._build_theme_candidates(hotspot_payload, pools["limit_up"])
        stocks = self._build_stock_candidates(
            pools["limit_up"],
            sentiment.highest_consecutive_board or 0,
            trade_date=trade_date,
        )
        external_tech_context = self._build_external_tech_context(global_indices)
        sources.append(
            AShareReviewSource(
                provider=external_tech_context.source,
                dataset="external_tech_context",
                status=external_tech_context.status,
                source_role="canonical",
                as_of=external_tech_context.as_of,
                trade_date=_normalize_date_text(trade_date),
            )
        )
        if external_tech_context.status != "ok":
            missing_fields.append("external_tech_context")
        risk_rules = self._build_risk_rules(index_trend, sentiment, external_tech_context)
        risk_tags = [rule.code for rule in risk_rules if rule.triggered]
        contaminated_fields = self._detect_contaminated_fields(indices, market_snapshot or {})

        has_evidence = bool(index_trend or themes or stocks or pool_available.get("limit_up"))
        if not has_evidence:
            status = "unknown"
        elif missing_fields or errors or contaminated_fields:
            status = "partial"
        else:
            status = "ok"

        evidence = AShareReviewEvidence(
            status=status,
            trade_date=trade_date,
            index_trend=index_trend,
            sentiment_structure=sentiment,
            theme_candidates=themes,
            stock_candidates=stocks,
            external_tech_context=external_tech_context,
            risk_rules=risk_rules,
            risk_tags=risk_tags,
            data_quality=AShareReviewDataQuality(
                status=status,
                missing_fields=_dedupe(missing_fields),
                contaminated_fields=_dedupe(contaminated_fields),
                sources=sources,
                errors=_dedupe(errors),
            ),
        )
        return dump_a_share_review_model(evidence)

    def _fetch_pool(
        self,
        pool_type: str,
        trade_date: str,
    ) -> Tuple[List[Dict[str, Any]], bool, List[AShareReviewSource], List[str]]:
        dataset = POOL_DATASETS[pool_type]
        fetch_with_meta = getattr(self.fetcher_manager, "get_limit_event_pool_with_meta", None)
        if callable(fetch_with_meta):
            try:
                result = fetch_with_meta(pool_type=pool_type, date=trade_date, n=self.pool_limit)
            except Exception as exc:
                return (
                    [],
                    False,
                    [AShareReviewSource(dataset=dataset, status="failed", message=str(exc), failure_reason=str(exc))],
                    [f"{dataset}: {type(exc).__name__}: {exc}"],
                )
            parsed = self._parse_pool_meta_result(dataset, result)
            if parsed is not None:
                return parsed

        if pool_type == "limit_up":
            legacy_fetch = getattr(self.fetcher_manager, "get_limit_up_pool", None)
            if callable(legacy_fetch):
                try:
                    rows = legacy_fetch(date=trade_date, n=self.pool_limit)
                except Exception as exc:
                    return (
                        [],
                        False,
                        [AShareReviewSource(dataset=dataset, status="failed", message=str(exc), failure_reason=str(exc))],
                        [f"{dataset}: {type(exc).__name__}: {exc}"],
                    )
                if isinstance(rows, list):
                    status = "ok" if rows else "empty"
                    return rows, True, [AShareReviewSource(dataset=dataset, status=status, source_role="canonical", trade_date=_normalize_date_text(trade_date))], []

        return [], False, [AShareReviewSource(dataset=dataset, status="missing")], []

    @staticmethod
    def _parse_pool_meta_result(
        dataset: str,
        result: Any,
    ) -> Optional[Tuple[List[Dict[str, Any]], bool, List[AShareReviewSource], List[str]]]:
        if not isinstance(result, tuple) or len(result) != 3:
            return None
        raw_rows, raw_chain, raw_error = result
        rows = [dict(row) for row in raw_rows if isinstance(row, Mapping)] if isinstance(raw_rows, list) else []
        sources: List[AShareReviewSource] = []
        available = False
        if isinstance(raw_chain, list):
            for item in raw_chain:
                if not isinstance(item, Mapping):
                    continue
                status = _text(item.get("status")) or "unknown"
                if status in {"ok", "empty"}:
                    available = True
                sources.append(
                    AShareReviewSource(
                        provider=_text(item.get("provider")) or "dsa",
                        dataset=dataset,
                        status=status,
                        message=_text(item.get("error")) or None,
                        source_role="canonical" if status in {"ok", "empty"} else "diagnostic",
                        failure_reason=_text(item.get("error")) or None,
                    )
                )
        errors = [f"{dataset}: {raw_error}"] if raw_error and not available else []
        if not sources:
            sources.append(AShareReviewSource(dataset=dataset, status="ok" if rows else "missing"))
        return rows, available or bool(rows), sources, errors

    def _build_index_trend(
        self,
        indices: Sequence[Any],
        *,
        trade_date: str,
    ) -> Tuple[List[AShareIndexTrend], List[AShareReviewSource], List[str]]:
        matched = self._match_indices(indices)
        rows: List[AShareIndexTrend] = []
        sources: List[AShareReviewSource] = []
        errors: List[str] = []
        fetch_history = getattr(self.fetcher_manager, "get_index_daily_history", None)
        if not callable(fetch_history):
            return rows, [AShareReviewSource(dataset="index_daily_history", status="missing")], errors

        for target in CN_INDEX_TREND_TARGETS:
            index = matched.get(target["symbol"])
            try:
                bars = fetch_history(target["symbol"], days=30)
            except Exception as exc:
                errors.append(f"index_daily_history:{target['symbol']}: {type(exc).__name__}: {exc}")
                sources.append(
                    AShareReviewSource(
                        dataset=f"index_daily_history:{target['symbol']}",
                        status="failed",
                        message=str(exc),
                    )
                )
                continue
            dated_bars = _bars_through_trade_date(bars or [], trade_date)
            last_bar = dated_bars[-1] if dated_bars else {}
            history_as_of = _bar_date(last_bar)
            runtime_as_of = _normalize_date_text(
                _value(index, "as_of") or _value(index, "date") or _value(index, "trade_date")
            )
            target_trade_date = _normalize_date_text(trade_date)
            runtime_current = _safe_float(_value(index, "current"))
            historical_current = _safe_float(last_bar.get("close")) if isinstance(last_bar, Mapping) else None
            if index is not None and runtime_current is not None and runtime_as_of == target_trade_date:
                current = runtime_current
                current_source = "runtime_snapshot"
                as_of = runtime_as_of
                row_status = "ok"
            elif historical_current is not None:
                current = historical_current
                current_source = "historical_close_fallback"
                as_of = history_as_of
                row_status = "ok" if current is not None and history_as_of == target_trade_date else "partial"
            else:
                current = runtime_current
                current_source = "runtime_snapshot_unverified"
                as_of = runtime_as_of
                row_status = "partial"

            levels = compute_index_key_levels(dated_bars, current=current)
            sources.append(
                AShareReviewSource(
                    provider=_text(last_bar.get("provider")) or "dsa",
                    dataset=f"index_daily_history:{target['symbol']}",
                    status="ok" if levels else "empty",
                    source_role="canonical",
                    as_of=as_of,
                    trade_date=target_trade_date,
                )
            )
            if not levels:
                continue
            rows.append(
                AShareIndexTrend(
                    code=target["symbol"],
                    name=target["name"],
                    current=current,
                    change_pct=(
                        _safe_float(_value(index, "change_pct"))
                        if current_source == "runtime_snapshot"
                        else _bar_change_pct(dated_bars)
                    ),
                    as_of=as_of,
                    trade_date=target_trade_date,
                    status=row_status,
                    current_source=current_source,
                    **levels,
                )
            )
        return rows, sources, errors

    @staticmethod
    def _match_indices(indices: Sequence[Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for target in CN_INDEX_TREND_TARGETS:
            aliases = tuple(str(alias).lower() for alias in target["aliases"])
            for index in indices or []:
                code = _text(_value(index, "code")).lower()
                name = _text(_value(index, "name")).lower()
                if any(alias and (code.endswith(alias) or name == alias) for alias in aliases):
                    result[target["symbol"]] = index
                    break
        return result

    @staticmethod
    def _build_sentiment(
        pools: Mapping[str, List[Dict[str, Any]]],
        available: Mapping[str, bool],
    ) -> AShareSentimentStructure:
        limit_up = pools.get("limit_up") or []
        broken = pools.get("broken_board") or []
        previous = pools.get("previous_limit") or []
        limit_down = pools.get("limit_down") or []
        legacy = aggregate_limit_up_pool(limit_up)
        premiums = [value for value in (_safe_float(row.get("change_pct")) for row in previous) if value is not None]
        one_price_rows = [row for row in limit_up if _is_one_price_like(row)]
        ladder = sorted(
            [
                {
                    "code": _text(row.get("code")),
                    "name": _text(row.get("name")),
                    "industry": _text(row.get("industry")),
                    "consecutive_boards": _safe_int(row.get("consecutive_boards")) or 0,
                    "limit_stat": _text(row.get("limit_stat")),
                    "first_limit_time": _text(row.get("first_limit_time")),
                    "break_count": _safe_int(row.get("break_count")) or 0,
                }
                for row in limit_up
                if (_safe_int(row.get("consecutive_boards")) or 0) >= 2
            ],
            key=lambda item: (-item["consecutive_boards"], item["first_limit_time"] or "999999"),
        )[:20]
        total_attempts = len(limit_up) + len(broken)
        return AShareSentimentStructure(
            limit_up_count=len(limit_up),
            broken_board_count=len(broken) if available.get("broken_board") else None,
            broken_ratio=(
                round(len(broken) / total_attempts, 4) if available.get("broken_board") and total_attempts else None
            ),
            limit_down_count=len(limit_down) if available.get("limit_down") else None,
            previous_limit_sample_count=len(premiums),
            previous_limit_premium_mean_pct=round(mean(premiums), 2) if premiums else None,
            previous_limit_premium_median_pct=round(median(premiums), 2) if premiums else None,
            highest_consecutive_board=legacy.get("max_consecutive_boards") or None,
            highest_board_stock=legacy.get("max_boards_stock") or None,
            one_price_like_count=len(one_price_rows),
            one_price_like_ratio=round(len(one_price_rows) / len(limit_up), 4) if limit_up else None,
            total_break_count=legacy.get("total_break_count") or 0,
            industry_distribution=legacy.get("industry_distribution") or [],
            multi_board_ladder=ladder,
        )

    @staticmethod
    def _build_theme_candidates(
        hotspot_payload: Any,
        limit_up_rows: List[Dict[str, Any]],
    ) -> List[AShareThemeCandidate]:
        active_themes = hotspot_payload.get("active_themes") if isinstance(hotspot_payload, Mapping) else []
        lagging_themes = hotspot_payload.get("lagging_themes") if isinstance(hotspot_payload, Mapping) else []
        theme_state: Dict[str, Dict[str, Any]] = {}
        for item in list(active_themes or []) + list(lagging_themes or []):
            if not isinstance(item, Mapping):
                continue
            name = _text(item.get("name"))
            if not name:
                continue
            grouped_name = THEME_GROUP_ALIASES.get(name, name)
            state = theme_state.setdefault(
                grouped_name,
                {
                    "name": grouped_name,
                    "sources": [],
                    "rank": None,
                    "change_pct": None,
                    "strength_score": 0.0,
                    "stocks": [],
                    "rows": [],
                    "subthemes": [],
                },
            )
            source = _text(item.get("source")) or "unknown"
            if source not in state["sources"]:
                state["sources"].append(source)
            if name not in state["subthemes"]:
                state["subthemes"].append(name)
            rank = _safe_int(item.get("rank"))
            if rank is not None and (state["rank"] is None or rank < state["rank"]):
                state["rank"] = rank
            change_pct = _safe_float(item.get("change_pct"))
            if change_pct is not None and (
                state["change_pct"] is None or change_pct > state["change_pct"]
            ):
                state["change_pct"] = change_pct
            state["strength_score"] = max(
                state["strength_score"],
                _safe_float(item.get("strength_score")) or 0.0,
            )

        for row in limit_up_rows:
            industry = _text(row.get("industry"))
            if not industry:
                continue
            grouped_name = THEME_GROUP_ALIASES.get(industry, industry)
            state = theme_state.setdefault(
                grouped_name,
                {
                    "name": grouped_name,
                    "sources": ["limit_up_industry"],
                    "rank": None,
                    "change_pct": None,
                    "strength_score": 0.0,
                    "stocks": [],
                    "rows": [],
                    "subthemes": [],
                },
            )
            if "limit_up_industry" not in state["sources"]:
                state["sources"].append("limit_up_industry")
            if industry not in state["subthemes"]:
                state["subthemes"].append(industry)
            label = " ".join(part for part in (_text(row.get("code")), _text(row.get("name"))) if part)
            if label and label not in state["stocks"]:
                state["stocks"].append(label)
            state["rows"].append(row)

        candidates: List[AShareThemeCandidate] = []
        for state in theme_state.values():
            rows = state["rows"]
            limit_up_count = len(rows)
            max_height = max((_safe_int(row.get("consecutive_boards")) or 0 for row in rows), default=0)
            one_price_count = sum(1 for row in rows if _is_one_price_like(row))
            capacity_core_present = any((_amount_yi(row.get("amount")) or 0) >= 50 for row in rows)
            tradeable_front_present = any(
                (_safe_int(row.get("consecutive_boards")) or 0) >= 2
                and not _is_low_liquidity_board(row)
                for row in rows
            )
            total_break_count = sum(_safe_int(row.get("break_count")) or 0 for row in rows)
            score = min(limit_up_count * 8, 32)
            score += min(max_height * 8, 24)
            score += min(float(state["strength_score"]) * 0.28, 28)
            score += max(0, 16 - one_price_count * 3 - min(total_break_count, 8))
            score = round(score, 2)

            if (
                score >= 68
                and limit_up_count >= 2
                and max_height >= 2
                and capacity_core_present
                and tradeable_front_present
            ):
                classification = "confirmed_mainline"
            elif score >= 55 and limit_up_count >= 2 and tradeable_front_present:
                classification = "mainline_candidate"
            elif state["change_pct"] is not None and state["change_pct"] < 0:
                classification = "observation_only"
            elif limit_up_count >= 2 or score >= 38:
                classification = "diffusion"
            elif limit_up_count >= 1 or (state["change_pct"] or 0) > 0:
                classification = "rotation_or_defense"
            else:
                classification = "observation_only"

            positive = []
            negative = []
            if state["change_pct"] is not None:
                positive.append(f"板块涨幅 {state['change_pct']:+.2f}%")
            if limit_up_count:
                positive.append(f"涨停 {limit_up_count} 家")
            if max_height >= 2:
                positive.append(f"最高 {max_height} 连板")
            if not rows:
                negative.append("仅有板块排行，缺少涨停扩散证据")
            if one_price_count:
                negative.append(f"疑似一字/无换手 {one_price_count} 家")
            if total_break_count:
                negative.append(f"封板过程开板 {total_break_count} 次")
            if not capacity_core_present:
                negative.append("缺少50亿元成交额以上的容量核心证据")
            if rows and not tradeable_front_present:
                negative.append("前排主要由一字或严重缩量板贡献，不具备交易资格")

            actionable = classification in ACTIONABLE_THEME_CLASSES
            candidates.append(
                AShareThemeCandidate(
                    theme=state["name"],
                    classification=classification,
                    total_score=score,
                    rank_source="+".join(state["sources"]) or "unknown",
                    rank=state["rank"],
                    change_pct=state["change_pct"],
                    limit_up_count=limit_up_count,
                    max_board_height=max_height,
                    representative_stocks=state["stocks"][:8],
                    subthemes=state["subthemes"][:6],
                    capacity_core_present=capacity_core_present,
                    tradeable_front_present=tradeable_front_present,
                    actionable=actionable,
                    position_cap_pct=(20 if classification == "confirmed_mainline" else 10 if actionable else 0),
                    positive_evidence=positive,
                    negative_evidence=negative,
                    confirmation="次日板块涨幅保持为正且强于沪深300，前排有换手承接，容量核心不跌破当日低点",
                    invalidation="板块转负且弱于沪深300，或前排断板无修复、容量核心跌破当日低点",
                )
            )
        candidates.sort(key=lambda item: (item.total_score, item.limit_up_count, item.max_board_height), reverse=True)
        actionable_seen = 0
        for candidate in candidates:
            if candidate.classification not in ACTIONABLE_THEME_CLASSES:
                continue
            actionable_seen += 1
            if actionable_seen <= 3:
                continue
            candidate.classification = "diffusion" if candidate.limit_up_count else "observation_only"
            candidate.actionable = False
            candidate.position_cap_pct = 0
            candidate.negative_evidence.append("可操作方向已达三个上限，本方向降级观察")
        return candidates[:12]

    @staticmethod
    def _build_stock_candidates(
        limit_up_rows: List[Dict[str, Any]],
        highest_board: int,
        *,
        trade_date: str,
    ) -> List[AShareStockCandidate]:
        candidates: List[AShareStockCandidate] = []
        for row in limit_up_rows:
            code = _text(row.get("code"))
            name = _text(row.get("name"))
            if not code and not name:
                continue
            boards = _safe_int(row.get("consecutive_boards")) or 0
            amount_yi = _amount_yi(row.get("amount"))
            observation_only = _is_low_liquidity_board(row)
            if boards >= 2 and boards == highest_board:
                category = "emotion_leader"
                buy_point = "分歧后的换手回封或超预期弱转强，不做一致追高"
            elif amount_yi is not None and amount_yi >= 50:
                category = "capacity_core"
                buy_point = "板块共振后的回踩承接或放量突破确认"
            elif boards >= 2:
                category = "front_row"
                buy_point = "前排分歧回封，确认同题材高标未出现负反馈"
            else:
                category = "low_position_extension"
                buy_point = "题材确认后的低位首板换手承接"

            risk_tags: List[str] = []
            if _is_one_price_like(row):
                risk_tags.append("one_price_like_heuristic")
            turnover = _safe_float(row.get("turnover_rate"))
            if turnover is not None and turnover <= 1:
                risk_tags.append("severely_low_turnover")
            break_count = _safe_int(row.get("break_count")) or 0
            if break_count >= 2:
                risk_tags.append("multiple_board_breaks")
            if turnover is not None and turnover >= 30:
                risk_tags.append("high_turnover")

            if observation_only:
                role = "emotion_thermometer"
                trade_eligibility = "observation_only"
                observation_reason = "一字或严重缩量，当前只用于观察情绪高度和板块反馈"
                buy_point = "仅作情绪温度计，不提供买入触发"
                confirmation_conditions = ["次日出现充分换手", "同题材容量核心维持承接"]
                position_cap_pct = 0
            else:
                role = category
                trade_eligibility = "conditional"
                observation_reason = "仅在板块与个股确认条件同时满足后进入交易候选"
                confirmation_conditions = [
                    "次日竞价不过度透支且开盘后有换手承接",
                    "同题材前排与容量核心未出现负反馈",
                ]
                position_cap_pct = 5

            score = boards * 10 + min((amount_yi or 0) / 10, 15)
            score += max(_safe_float(row.get("change_pct")) or 0, 0)
            candidates.append(
                AShareStockCandidate(
                    category=category,
                    code=code,
                    name=name,
                    themes=[_text(row.get("industry"))] if _text(row.get("industry")) else [],
                    change_pct=_safe_float(row.get("change_pct")),
                    amount_yi=amount_yi,
                    consecutive_boards=boards,
                    buy_point_type=buy_point,
                    validation="次日竞价不过度透支，开盘后有换手承接且同题材前排不掉队",
                    invalidation="高开回落收不住、炸板后无回封，或同题材前排先出现负反馈",
                    role=role,
                    trade_eligibility=trade_eligibility,
                    observation_reason=observation_reason,
                    trigger_type="换手与板块共振确认" if not observation_only else "无交易触发",
                    trigger_level=None,
                    confirmation_conditions=confirmation_conditions,
                    invalidation_level=_safe_float(row.get("low")),
                    position_cap_pct=position_cap_pct,
                    risk_tags=risk_tags,
                    price_reference_date=_normalize_date_text(trade_date),
                    score=round(score, 2),
                )
            )
        candidates.sort(key=lambda item: (item.score, item.consecutive_boards, item.amount_yi or 0), reverse=True)
        return candidates[:20]

    @staticmethod
    def _build_external_tech_context(
        global_indices: Sequence[Any],
    ) -> AShareExternalTechContext:
        nasdaq_change: Optional[float] = None
        semiconductor_change: Optional[float] = None
        nasdaq_as_of: Optional[str] = None
        semiconductor_as_of: Optional[str] = None
        key_symbols: List[Dict[str, Any]] = []
        providers: List[str] = []
        for item in global_indices or []:
            code = _text(_value(item, "code"))
            name = _text(_value(item, "name"))
            identity = f"{code} {name}".lower()
            change_pct = _safe_float(_value(item, "change_pct"))
            as_of = _normalize_date_text(
                _value(item, "as_of") or _value(item, "date") or _value(item, "trade_date")
            )
            provider = _text(_value(item, "provider")) or "dsa"
            if provider not in providers:
                providers.append(provider)
            is_nasdaq = any(token in identity for token in ("ixic", "ndx", "nasdaq", "纳斯达克"))
            is_semiconductor = any(token in identity for token in ("sox", "semiconductor", "半导体"))
            if not is_nasdaq and not is_semiconductor:
                continue
            key_symbols.append(
                {
                    "code": code,
                    "name": name,
                    "change_pct": change_pct,
                    "as_of": as_of,
                    "provider": provider,
                }
            )
            if is_nasdaq and nasdaq_change is None:
                nasdaq_change = change_pct
                nasdaq_as_of = as_of or None
            if is_semiconductor and semiconductor_change is None:
                semiconductor_change = change_pct
                semiconductor_as_of = as_of or None

        if (
            nasdaq_change is not None
            and semiconductor_change is not None
            and nasdaq_as_of
            and semiconductor_as_of
        ):
            status = "ok"
        elif nasdaq_change is not None or semiconductor_change is not None:
            status = "partial"
        else:
            status = "missing"
        return AShareExternalTechContext(
            as_of=max(
                (value for value in (nasdaq_as_of, semiconductor_as_of) if value),
                default=None,
            ),
            nasdaq_change_pct=nasdaq_change,
            semiconductor_proxy_change_pct=semiconductor_change,
            key_symbols=key_symbols,
            status=status,
            source="+".join(providers) or "global_market_indices",
        )

    @staticmethod
    def _build_risk_rules(
        index_trend: List[AShareIndexTrend],
        sentiment: AShareSentimentStructure,
        external_tech_context: AShareExternalTechContext,
    ) -> List[AShareReviewRiskRule]:
        broken_triggered = sentiment.broken_ratio is not None and sentiment.broken_ratio > 0.30
        premium_triggered = (
            sentiment.previous_limit_premium_median_pct is not None and sentiment.previous_limit_premium_median_pct < 0
        )
        weak_growth_indices = [
            item.name
            for item in index_trend
            if item.name in {"创业板指", "科创50"}
            and (
                (item.dist_ma5_pct is not None and item.dist_ma5_pct < 0)
                or (item.dist_ma10_pct is not None and item.dist_ma10_pct < 0)
            )
        ]
        one_price_triggered = sentiment.one_price_like_ratio is not None and sentiment.one_price_like_ratio >= 0.25
        external_missing = external_tech_context.status != "ok"
        external_negative = (
            external_tech_context.semiconductor_proxy_change_pct is not None
            and external_tech_context.semiconductor_proxy_change_pct <= -2.0
        ) or (
            external_tech_context.nasdaq_change_pct is not None
            and external_tech_context.nasdaq_change_pct <= -1.5
        )
        return [
            AShareReviewRiskRule(
                code="high_broken_board_ratio",
                triggered=broken_triggered,
                evidence=(
                    f"炸板率 {sentiment.broken_ratio:.1%}" if sentiment.broken_ratio is not None else "炸板池不可用"
                ),
                action="炸板率超过30%时降低接力仓位，不把涨停数量直接视为强势",
            ),
            AShareReviewRiskRule(
                code="negative_previous_limit_premium",
                triggered=premium_triggered,
                evidence=(
                    f"昨日涨停溢价中位数 {sentiment.previous_limit_premium_median_pct:+.2f}%"
                    if sentiment.previous_limit_premium_median_pct is not None
                    else "昨日涨停溢价不可用"
                ),
                action="溢价中位数为负时回避高位接力，等待弱转强确认",
            ),
            AShareReviewRiskRule(
                code="growth_index_below_short_ma",
                triggered=bool(weak_growth_indices),
                evidence=(
                    ("、".join(weak_growth_indices) + " 位于MA5或MA10下方")
                    if weak_growth_indices
                    else "成长指数短均线未触发"
                ),
                action="创业板或科创50跌破短均线时，下调高位科技线的默认优先级",
            ),
            AShareReviewRiskRule(
                code="one_price_concentration",
                triggered=one_price_triggered,
                evidence=(
                    f"疑似一字/无换手占比 {sentiment.one_price_like_ratio:.1%}"
                    if sentiment.one_price_like_ratio is not None
                    else "一字板启发式不可用"
                ),
                action="不可交易的一致强度只作情绪参考，不作为可执行买点",
            ),
            AShareReviewRiskRule(
                code="external_tech_context_missing",
                triggered=external_missing,
                evidence=(
                    "外围科技数据缺失"
                    if external_tech_context.status == "missing"
                    else "外围科技数据覆盖不完整"
                ),
                action="外围科技链未完整覆盖时，整体风险状态最高为黄色",
            ),
            AShareReviewRiskRule(
                code="external_tech_negative",
                triggered=external_negative,
                evidence=(
                    "纳指 {nasdaq}，半导体代理 {semi}".format(
                        nasdaq=(
                            f"{external_tech_context.nasdaq_change_pct:+.2f}%"
                            if external_tech_context.nasdaq_change_pct is not None
                            else "N/A"
                        ),
                        semi=(
                            f"{external_tech_context.semiconductor_proxy_change_pct:+.2f}%"
                            if external_tech_context.semiconductor_proxy_change_pct is not None
                            else "N/A"
                        ),
                    )
                ),
                action="外围科技显著负反馈且A股成长结构未修复时，触发新开仓否决",
            ),
        ]

    @staticmethod
    def _detect_contaminated_fields(
        indices: Sequence[Any],
        market_snapshot: Mapping[str, Any],
    ) -> List[str]:
        contaminated: List[str] = []
        if indices and all((_safe_float(_value(item, "current")) or 0) <= 0 for item in indices):
            contaminated.append("indices")
        breadth_values = [
            _safe_int(market_snapshot.get("up_count")) or 0,
            _safe_int(market_snapshot.get("down_count")) or 0,
            _safe_int(market_snapshot.get("flat_count")) or 0,
        ]
        if market_snapshot and sum(breadth_values) == 0:
            contaminated.append("breadth")
        return contaminated


def _value(item: Any, key: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)


def _distance_pct(current: Optional[float], moving_average: float) -> Optional[float]:
    if current is None or moving_average == 0:
        return None
    return round((current / moving_average - 1) * 100, 2)


def _is_one_price_like(row: Mapping[str, Any]) -> bool:
    if (_safe_int(row.get("break_count")) or 0) != 0:
        return False
    digits = "".join(character for character in _text(row.get("first_limit_time")) if character.isdigit())
    if not digits:
        return False
    digits = digits.zfill(6)[-6:]
    return digits <= "093000"


def _is_low_liquidity_board(row: Mapping[str, Any]) -> bool:
    if _is_one_price_like(row):
        return True
    turnover = _safe_float(row.get("turnover_rate"))
    return turnover is not None and turnover <= 1.0


def _normalize_date_text(value: Any) -> Optional[str]:
    digits = "".join(character for character in _text(value) if character.isdigit())
    if len(digits) < 8:
        return None
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"


def _bar_date(bar: Any) -> Optional[str]:
    if not isinstance(bar, Mapping):
        return None
    return _normalize_date_text(
        bar.get("date") or bar.get("trade_date") or bar.get("datetime") or bar.get("time")
    )


def _bars_through_trade_date(bars: Sequence[Any], trade_date: str) -> List[Dict[str, Any]]:
    target = _normalize_date_text(trade_date)
    normalized = [dict(bar) for bar in bars if isinstance(bar, Mapping)]
    if not target:
        return normalized
    parseable = [bar for bar in normalized if _bar_date(bar)]
    if not parseable:
        return normalized
    return sorted(
        (bar for bar in parseable if _bar_date(bar) <= target),
        key=lambda bar: _bar_date(bar) or "",
    )


def _bar_change_pct(bars: Sequence[Mapping[str, Any]]) -> Optional[float]:
    if len(bars) < 2:
        return None
    current = _safe_float(bars[-1].get("close"))
    previous = _safe_float(bars[-2].get("close"))
    if current is None or previous in (None, 0):
        return None
    return round((current / previous - 1) * 100, 2)


def _amount_yi(value: Any) -> Optional[float]:
    amount = _safe_float(value)
    if amount is None:
        return None
    return round(amount / 100_000_000, 2) if abs(amount) >= 1_000_000 else round(amount, 2)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            text = value.strip().replace(",", "")
            if not text:
                return None
            if text.endswith("%"):
                text = text[:-1].strip()
            return float(text)
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    number = _safe_float(value)
    return int(number) if number is not None else None


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _dedupe(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    for value in values:
        text = _text(value)
        if text and text not in result:
            result.append(text)
    return result
