"""Outcome labels from posted transactions, following the handoff's self-pay definition.

A candidate self-pay payment is TXN Type = Payment, Payment Type = Patient Payment and
Exclude Transaction = Include. NULL classifications never qualify. Promises to pay,
adjustments and insurance payments are not patient cash.

Horizon and minimum amount are still pending business definitions with Salud; the
defaults here are the team's working assumption and are passed explicitly.
"""

from __future__ import annotations

import pandas as pd

from . import schema as S


def self_pay_payments(receipts: pd.DataFrame) -> pd.DataFrame:
    """Rows of the receipts table that count as patient cash."""
    mask = (
        (receipts["salud_TXNType"] == S.TXN_PAYMENT)
        & (receipts["salud_PaymentType"] == S.PAY_PATIENT)
        & (receipts["salud_ExcludeTransaction"] == S.TXN_INCLUDE)
    )
    return receipts.loc[mask.fillna(False)]


def account_outcomes(
    accounts: pd.DataFrame,
    receipts: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon_days: int = 90,
    min_amount: float = 0.01,
) -> pd.DataFrame:
    """One row per account: did self-pay cash arrive in (cutoff, cutoff + horizon]?

    Receipts are aggregated to account grain on their own (never via contact joins), so a
    call that covers several accounts cannot duplicate payment totals.
    """
    end = cutoff + pd.Timedelta(days=horizon_days)
    pay = self_pay_payments(receipts)
    window = pay[(pay["salud_PostDate"] > cutoff) & (pay["salud_PostDate"] <= end)]
    agg = window.groupby("salud_ReceiptsId").agg(
        paid_amount=("salud_Amount", "sum"),
        first_payment_date=("salud_PostDate", "min"),
    )
    out = accounts[["salud_gphpvaccountsId"]].merge(
        agg, left_on="salud_gphpvaccountsId", right_index=True, how="left"
    )
    out["paid_amount"] = out["paid_amount"].fillna(0.0)
    out["paid_any"] = out["paid_amount"] >= min_amount
    out["days_to_first_payment"] = (out["first_payment_date"] - cutoff).dt.days
    return out
