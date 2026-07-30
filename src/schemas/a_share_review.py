# -*- coding: utf-8 -*-
"""Versioned A-share daily-review evidence shared by reports and agents."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

A_SHARE_REVIEW_EVIDENCE_SCHEMA_VERSION = "a-share-review-evidence-v2"
A_SHARE_REVIEW_STATUS = Literal["ok", "partial", "unknown", "not_supported"]


class AShareReviewSource(BaseModel):
    provider: str = "dsa"
    dataset: str
    status: str = "unknown"
    message: Optional[str] = None
    source_role: Literal["canonical", "fallback", "diagnostic"] = "diagnostic"
    as_of: Optional[str] = None
    trade_date: Optional[str] = None
    failure_reason: Optional[str] = None


class AShareReviewDataQuality(BaseModel):
    status: A_SHARE_REVIEW_STATUS = "unknown"
    missing_fields: List[str] = Field(default_factory=list)
    contaminated_fields: List[str] = Field(default_factory=list)
    sources: List[AShareReviewSource] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class AShareIndexTrend(BaseModel):
    code: str
    name: str
    current: Optional[float] = None
    change_pct: Optional[float] = None
    ma5: Optional[float] = None
    ma10: Optional[float] = None
    ma20: Optional[float] = None
    dist_ma5_pct: Optional[float] = None
    dist_ma10_pct: Optional[float] = None
    dist_ma20_pct: Optional[float] = None
    high_20d: Optional[float] = None
    low_20d: Optional[float] = None
    as_of: Optional[str] = None
    trade_date: Optional[str] = None
    status: str = "unknown"
    current_source: str = "unknown"


class AShareSentimentStructure(BaseModel):
    limit_up_count: int = 0
    broken_board_count: Optional[int] = None
    broken_ratio: Optional[float] = None
    limit_down_count: Optional[int] = None
    previous_limit_sample_count: int = 0
    previous_limit_premium_mean_pct: Optional[float] = None
    previous_limit_premium_median_pct: Optional[float] = None
    highest_consecutive_board: Optional[int] = None
    highest_board_stock: Optional[str] = None
    one_price_like_count: int = 0
    one_price_like_ratio: Optional[float] = None
    total_break_count: int = 0
    industry_distribution: List[dict] = Field(default_factory=list)
    multi_board_ladder: List[dict] = Field(default_factory=list)


class AShareThemeCandidate(BaseModel):
    theme: str
    classification: str
    total_score: float
    rank_source: str = "unknown"
    rank: Optional[int] = None
    change_pct: Optional[float] = None
    limit_up_count: int = 0
    max_board_height: int = 0
    representative_stocks: List[str] = Field(default_factory=list)
    subthemes: List[str] = Field(default_factory=list)
    capacity_core_present: bool = False
    tradeable_front_present: bool = False
    actionable: bool = False
    position_cap_pct: int = Field(default=0, ge=0, le=100)
    positive_evidence: List[str] = Field(default_factory=list)
    negative_evidence: List[str] = Field(default_factory=list)
    confirmation: str
    invalidation: str


class AShareStockCandidate(BaseModel):
    category: str
    code: str
    name: str
    themes: List[str] = Field(default_factory=list)
    change_pct: Optional[float] = None
    amount_yi: Optional[float] = None
    consecutive_boards: int = 0
    buy_point_type: str
    validation: str
    invalidation: str
    role: str = "watchlist"
    trade_eligibility: Literal["conditional", "observation_only"] = "conditional"
    observation_reason: str = ""
    trigger_type: str = "confirmation"
    trigger_level: Optional[float] = None
    confirmation_conditions: List[str] = Field(default_factory=list)
    invalidation_level: Optional[float] = None
    position_cap_pct: int = Field(default=0, ge=0, le=100)
    risk_tags: List[str] = Field(default_factory=list)
    price_reference_date: Optional[str] = None
    score: float = 0.0


class AShareReviewRiskRule(BaseModel):
    code: str
    triggered: bool = False
    evidence: str
    action: str


class AShareExternalTechContext(BaseModel):
    as_of: Optional[str] = None
    nasdaq_change_pct: Optional[float] = None
    semiconductor_proxy_change_pct: Optional[float] = None
    key_symbols: List[Dict[str, Any]] = Field(default_factory=list)
    status: Literal["ok", "partial", "missing"] = "missing"
    source: str = "global_market_indices"


class AShareRiskAssessment(BaseModel):
    raw_heat_score: int = Field(default=50, ge=0, le=100)
    raw_heat_label: str = "unknown"
    risk_state: Literal["red", "yellow", "green"] = "yellow"
    position_mode: str = "confirmation_trial"
    position_cap_pct: int = Field(default=30, ge=0, le=100)
    triggered_gates: List[str] = Field(default_factory=list)
    gate_evidence: List[str] = Field(default_factory=list)


class AShareReviewEvidence(BaseModel):
    schema_version: str = A_SHARE_REVIEW_EVIDENCE_SCHEMA_VERSION
    status: A_SHARE_REVIEW_STATUS = "unknown"
    market: str = "cn"
    trade_date: Optional[str] = None
    index_trend: List[AShareIndexTrend] = Field(default_factory=list)
    sentiment_structure: AShareSentimentStructure = Field(default_factory=AShareSentimentStructure)
    theme_candidates: List[AShareThemeCandidate] = Field(default_factory=list)
    stock_candidates: List[AShareStockCandidate] = Field(default_factory=list)
    external_tech_context: AShareExternalTechContext = Field(default_factory=AShareExternalTechContext)
    risk_rules: List[AShareReviewRiskRule] = Field(default_factory=list)
    risk_tags: List[str] = Field(default_factory=list)
    risk_assessment: AShareRiskAssessment = Field(default_factory=AShareRiskAssessment)
    data_quality: AShareReviewDataQuality = Field(default_factory=AShareReviewDataQuality)


def dump_a_share_review_model(model: BaseModel) -> dict:
    return model.model_dump(exclude_none=True)
