# -*- coding: utf-8 -*-
"""Structured Market Light snapshot schema."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


MarketRegion = Literal["cn", "hk", "us", "jp", "kr"]
MarketLightStatus = Literal["green", "yellow", "red"]
MarketLightDataQuality = Literal["ok", "partial", "unavailable"]
MARKET_LIGHT_REGIONS = frozenset(("cn", "hk", "us"))


class MarketLightDimension(BaseModel):
    """A single Market Light scoring dimension."""

    score: int = Field(ge=0, le=100)
    available: bool


class MarketLightDimensions(BaseModel):
    """Canonical Market Light dimension scores."""

    breadth: MarketLightDimension
    index: MarketLightDimension
    limit: MarketLightDimension


class MarketLightSnapshot(BaseModel):
    """Structured Market Light snapshot persisted and consumed by alerts."""

    region: MarketRegion
    trade_date: str
    status: MarketLightStatus
    score: int = Field(ge=0, le=100)
    label: str
    temperature_label: str
    reasons: list[str]
    guidance: str
    dimensions: MarketLightDimensions
    data_quality: MarketLightDataQuality
    market_heat_score: int = Field(default=50, ge=0, le=100)
    raw_heat_score: int = Field(default=50, ge=0, le=100)
    raw_heat_label: str = "unknown"
    risk_state: MarketLightStatus = "yellow"
    position_mode: str = "confirmation_trial"
    position_cap_pct: int = Field(default=30, ge=0, le=100)
    triggered_gates: list[str] = Field(default_factory=list)
    gate_evidence: list[str] = Field(default_factory=list)
