# -*- coding: utf-8 -*-
"""Versioned A-share daily-review evidence shared by reports and agents."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

A_SHARE_REVIEW_EVIDENCE_SCHEMA_VERSION = "a-share-review-evidence-v1"
A_SHARE_REVIEW_STATUS = Literal["ok", "partial", "unknown", "not_supported"]


class AShareReviewSource(BaseModel):
    provider: str = "dsa"
    dataset: str
    status: str = "unknown"
    message: Optional[str] = None


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
    ma5: Optional[float] = None
    ma10: Optional[float] = None
    ma20: Optional[float] = None
    dist_ma5_pct: Optional[float] = None
    dist_ma10_pct: Optional[float] = None
    dist_ma20_pct: Optional[float] = None
    high_20d: Optional[float] = None
    low_20d: Optional[float] = None


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
    risk_tags: List[str] = Field(default_factory=list)
    score: float = 0.0


class AShareReviewRiskRule(BaseModel):
    code: str
    triggered: bool = False
    evidence: str
    action: str


class AShareReviewEvidence(BaseModel):
    schema_version: str = A_SHARE_REVIEW_EVIDENCE_SCHEMA_VERSION
    status: A_SHARE_REVIEW_STATUS = "unknown"
    market: str = "cn"
    trade_date: Optional[str] = None
    index_trend: List[AShareIndexTrend] = Field(default_factory=list)
    sentiment_structure: AShareSentimentStructure = Field(default_factory=AShareSentimentStructure)
    theme_candidates: List[AShareThemeCandidate] = Field(default_factory=list)
    stock_candidates: List[AShareStockCandidate] = Field(default_factory=list)
    risk_rules: List[AShareReviewRiskRule] = Field(default_factory=list)
    risk_tags: List[str] = Field(default_factory=list)
    data_quality: AShareReviewDataQuality = Field(default_factory=AShareReviewDataQuality)


def dump_a_share_review_model(model: BaseModel) -> dict:
    return model.model_dump(exclude_none=True)
