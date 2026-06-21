"""Vendor adapters. FREE-tier first:
  - yfinance_source : free, anonymous daily OHLCV backbone (Yahoo Finance).
  - stooq_source    : free CSV daily bars (independent cross-check; includes some delisted).
  - sec_edgar       : free, official fundamentals (company facts) for value/quality factors.

All return typed pandas objects so the rest of the lake is vendor-agnostic. See
docs/data_sources.md for free-tier limits and the survivorship-bias caveat.
"""
