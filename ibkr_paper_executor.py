#!/usr/bin/env python3
"""
IBKR paper executor experiment (safety-first).

Consumes execution_engine CSV output and prepares:
1) BUY paper-intent queue
2) MANUAL_REVIEW queue
3) Run summary artifacts

Default is dry-run only. No live routing is allowed.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from ibkr_connector import IBKRConnector, LIVE_PORTS, PAPER_PORTS, confirm_paper_session
from risk_manager import size_position

logger = logging.getLogger(__name__)

LIVE_BLOCKED_PORTS = LIVE_PORTS
PAPER_ALLOWED_PORTS = PAPER_PORTS
ALLOWED_DECISIONS = {"BUY", "MANUAL_REVIEW", "SKIP"}


@dataclass(frozen=True)
class PaperExecutorConfig:
    tws_host: str = "127.0.0.1"
    tws_port: int = 7497
    client_id: int = 91
    account: str = ""


def _read_config_defaults() -> PaperExecutorConfig:
    try:
        import config  # type: ignore

        return PaperExecutorConfig(
            tws_host=str(getattr(config, "TWS_HOST", "127.0.0.1")),
            tws_port=int(getattr(config, "TWS_PORT", 7497)),
            # Never config.CLIENT_ID: that's the all-day watcher's id
            client_id=int(getattr(config, "PAPER_EXECUTOR_CLIENT_ID", 91)),
            account=str(getattr(config, "IBKR_ACCOUNT", "")),
        )
    except Exception:
        return PaperExecutorConfig()


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def _to_float(value) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_entry_price(row: pd.Series) -> Optional[float]:
    for col in ("Entry_Price", "Price", "Current_Price", "entry_price"):
        if col in row.index and not pd.isna(row[col]):
            return _to_float(row[col])
    return None


def _normalize_symbol_value(raw_value) -> tuple[Optional[str], Optional[str]]:
    if raw_value is None or pd.isna(raw_value):
        return None, "MISSING_SYMBOL"

    symbol_raw = str(raw_value).strip()
    if not symbol_raw:
        return None, "MISSING_SYMBOL"

    symbol = symbol_raw.upper()
    if symbol in {"NAN", "NONE", "NULL"}:
        return None, f"INVALID_SYMBOL:{symbol_raw}"
    return symbol, None


def _normalize_decision_value(raw_value) -> tuple[Optional[str], str]:
    if raw_value is None or pd.isna(raw_value):
        return None, "<NULL>"

    normalized = str(raw_value).strip().upper()
    if not normalized:
        return None, "<BLANK>"
    if normalized not in ALLOWED_DECISIONS:
        return None, normalized
    return normalized, normalized


def _append_reason(existing_reason, addition: str) -> str:
    base = "" if existing_reason is None or pd.isna(existing_reason) else str(existing_reason).strip()
    return addition if not base else f"{base}; {addition}"


def _classify_decisions(decisions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    routed_rows = []
    unclassified_rows = 0
    invalid_symbol_rows = 0

    for _, row in decisions.iterrows():
        row_dict = row.to_dict()

        symbol_normalized, symbol_issue = _normalize_symbol_value(row_dict.get("Symbol"))
        decision_normalized, decision_display = _normalize_decision_value(row_dict.get("Decision"))

        reason = row_dict.get("Reason")
        if decision_normalized is None:
            unclassified_rows += 1
            reason = _append_reason(reason, f"UNCLASSIFIED_DECISION:{decision_display}")
            decision_normalized = "MANUAL_REVIEW"

        if symbol_issue:
            invalid_symbol_rows += 1
            reason = _append_reason(reason, symbol_issue)
            if decision_normalized == "BUY":
                decision_normalized = "MANUAL_REVIEW"

        row_dict["Decision"] = decision_normalized
        row_dict["Reason"] = reason
        row_dict["Symbol"] = symbol_normalized or ""
        routed_rows.append(row_dict)

    routed = pd.DataFrame(routed_rows)
    buy_df = routed[(routed["Decision"] == "BUY") & (routed["Symbol"].str.len() > 0)].copy()
    manual_df = routed[routed["Decision"] == "MANUAL_REVIEW"].copy()
    skip_df = routed[routed["Decision"] == "SKIP"].copy()
    metrics = {
        "unclassified_rows": int(unclassified_rows),
        "invalid_symbol_rows": int(invalid_symbol_rows),
        "buy_rows_validation_ready": int(len(buy_df)),
    }
    return buy_df, manual_df, skip_df, metrics


def _build_buy_intents(df_buy: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    rows = []
    excluded_missing_symbol = 0
    for _, row in df_buy.iterrows():
        symbol, symbol_issue = _normalize_symbol_value(row.get("Symbol"))
        if symbol_issue:
            excluded_missing_symbol += 1
            continue

        stop_pct = _to_float(row.get("Stop_Pct"))
        risk_pct_target = _to_float(row.get("Risk_Pct_Target"))
        position_cap_pct = _to_float(row.get("Position_Cap_Pct"))
        entry_price = _resolve_entry_price(row)

        sizing_status = "RISK_METADATA_ONLY"
        position_pct = None
        estimated_position_dollars = None
        estimated_shares = None
        if entry_price and entry_price > 0 and position_cap_pct is not None:
            # Size by risk with the one shared rule (risk_manager.size_position): risk % / stop %,
            # capped by the position cap.
            if stop_pct and stop_pct > 0 and risk_pct_target is not None:
                position_pct, _ = size_position(risk_pct_target, stop_pct, position_cap_pct)
                sizing_status = "RISK_SIZED"
            else:
                position_pct = position_cap_pct
                sizing_status = "POSITION_CAP_ONLY_NO_STOP"
            estimated_position_dollars = f"<account_equity>*{position_pct / 100:.4f}"
            estimated_shares = f"({estimated_position_dollars})/{entry_price:.4f}"

        rows.append(
            {
                "Symbol": symbol,
                "Intent_Action": "BUY",
                "Intent_Type": "PAPER_ONLY",
                "Entry_Type": row.get("Entry_Type", ""),
                "Entry_Price": entry_price,
                "Stop_Pct": stop_pct,
                "Risk_Pct_Target": risk_pct_target,
                "Position_Cap_Pct": position_cap_pct,
                "Sector_Cap_Pct": _to_float(row.get("Sector_Cap_Pct")),
                "Max_New_Entries_Day": _to_float(row.get("Max_New_Entries_Day")),
                "Max_Open_Risk_Pct": _to_float(row.get("Max_Open_Risk_Pct")),
                "Needs_Manual_Permission": bool(row.get("Needs_Manual_Permission", False)),
                "Sizing_Status": sizing_status,
                "Position_Pct_Of_Equity": round(position_pct, 4) if position_pct is not None else None,
                "Estimated_Position_Dollars": estimated_position_dollars,
                "Estimated_Shares_Expr": estimated_shares,
                "Route": "PAPER_SIMULATED_INTENT",
            }
        )
    return pd.DataFrame(rows), excluded_missing_symbol


def _build_manual_review_queue(df_manual: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Symbol",
        "Decision",
        "Reason",
        "Entry_Type",
        "Stop_Pct",
        "Needs_Manual_Permission",
        "Earnings_Blackout",
    ]
    available = [c for c in cols if c in df_manual.columns]
    return df_manual[available].copy()


def _validate_safety_for_submit(config: PaperExecutorConfig, submit_paper: bool) -> None:
    if not submit_paper:
        return

    if config.tws_port in LIVE_BLOCKED_PORTS:
        raise PermissionError(
            f"Live IBKR port {config.tws_port} is blocked for this experiment. Use paper port (7497/4002)."
        )

    if config.tws_port not in PAPER_ALLOWED_PORTS:
        raise PermissionError(
            f"Port {config.tws_port} is not an approved paper port ({sorted(PAPER_ALLOWED_PORTS)})."
        )

    if config.account and not str(config.account).upper().startswith("DU"):
        raise PermissionError(
            f"Account '{config.account}' is not a paper account (must start with 'DU'). Live routing is blocked."
        )


def _confirm_paper_account_with_broker(config: PaperExecutorConfig) -> tuple[str, list[str]]:
    connector = IBKRConnector()
    if not connector.connect_and_run(config.tws_host, config.tws_port, config.client_id):
        raise ConnectionError("Failed to connect to IBKR for broker account confirmation.")

    try:
        # Every account in the session must be paper (a mixed DU + U session is refused)
        reported_accounts = confirm_paper_session(connector, config.tws_port)

        if config.account:
            requested = config.account.strip().upper()
            if requested not in reported_accounts:
                raise PermissionError(
                    f"Requested account '{requested}' not in broker-reported accounts: {reported_accounts}"
                )
            effective_account = requested
        else:
            paper_accounts = [acct for acct in reported_accounts if acct.startswith("DU")]
            if not paper_accounts:
                raise PermissionError(
                    f"Broker-reported accounts are not paper accounts: {reported_accounts}"
                )
            effective_account = paper_accounts[0]

        if not effective_account.startswith("DU"):
            raise PermissionError(
                f"Effective account '{effective_account}' is not a paper account. Live routing is blocked."
            )
        return effective_account, reported_accounts
    finally:
        connector.disconnect_and_stop()


def _validate_with_ibkr(df_buy_intents: pd.DataFrame, config: PaperExecutorConfig) -> pd.DataFrame:
    connector = IBKRConnector()
    validated = df_buy_intents.copy()
    validated["IBKR_Validated"] = False
    validated["IBKR_Validation_Reason"] = "Not checked"

    if validated.empty:
        return validated

    if not connector.connect_and_run(config.tws_host, config.tws_port, config.client_id):
        validated["IBKR_Validation_Reason"] = "TWS paper connection failed"
        return validated

    try:
        for idx, row in validated.iterrows():
            symbol = str(row.get("Symbol", "")).strip().upper()
            if not symbol:
                validated.at[idx, "IBKR_Validation_Reason"] = "Missing symbol"
                continue

            details = connector.fetch_contract_details(symbol=symbol, timeout=8)
            if details is None:
                validated.at[idx, "IBKR_Validation_Reason"] = "Contract details unavailable"
                continue

            validated.at[idx, "IBKR_Validated"] = True
            validated.at[idx, "IBKR_Validation_Reason"] = "Paper intent validated via contract details"
    finally:
        connector.disconnect_and_stop()

    return validated


def parse_args() -> argparse.Namespace:
    defaults = _read_config_defaults()
    parser = argparse.ArgumentParser(
        description="Prepare IBKR paper trade intents from execution_engine decision CSV."
    )
    parser.add_argument("--input", required=True, help="Path to execution_engine output CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for intent/review/summary artifacts.")
    parser.add_argument(
        "--submit-paper",
        action="store_true",
        help="Validate BUY intents against the paper account (contract details only; places no "
             "orders, blocks live ports and non-DU accounts).",
    )
    parser.add_argument("--tws-host", default=defaults.tws_host, help="IBKR TWS/Gateway host.")
    parser.add_argument("--tws-port", type=int, default=defaults.tws_port, help="IBKR TWS/Gateway port.")
    parser.add_argument("--client-id", type=int, default=defaults.client_id, help="IBKR API client id.")
    parser.add_argument(
        "--account",
        default=defaults.account,
        help="Paper account id (must start with DU for --submit-paper).",
    )
    return parser.parse_args()


def main() -> int:
    _setup_logging()
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    decisions = pd.read_csv(input_path)
    required = {"Symbol", "Decision", "Reason"}
    missing = sorted(required - set(decisions.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    config = PaperExecutorConfig(
        tws_host=args.tws_host,
        tws_port=args.tws_port,
        client_id=args.client_id,
        account=str(args.account or "").strip(),
    )
    _validate_safety_for_submit(config=config, submit_paper=bool(args.submit_paper))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # --submit-paper validates contracts against the paper account; no order is ever sent
    mode_label = "paper_validate" if args.submit_paper else "dry_run"

    buy_df, manual_df, skip_df, classification_metrics = _classify_decisions(decisions)

    effective_account = config.account
    broker_reported_accounts: list[str] = []
    if args.submit_paper:
        effective_account, broker_reported_accounts = _confirm_paper_account_with_broker(config=config)

    buy_intents, excluded_missing_symbol = _build_buy_intents(buy_df)
    if args.submit_paper:
        buy_intents = _validate_with_ibkr(df_buy_intents=buy_intents, config=config)
        buy_intents["Submit_Mode"] = "VALIDATED_NOT_SUBMITTED"
    else:
        buy_intents["IBKR_Validated"] = False
        buy_intents["IBKR_Validation_Reason"] = "Dry-run mode (default)"
        buy_intents["Submit_Mode"] = "DRY_RUN"

    manual_review_queue = _build_manual_review_queue(manual_df)

    buy_path = output_dir / f"paper_executor_{ts}_{mode_label}_BUY_INTENTS.csv"
    review_path = output_dir / f"paper_executor_{ts}_{mode_label}_MANUAL_REVIEW_QUEUE.csv"
    summary_path = output_dir / f"paper_executor_{ts}_{mode_label}_RUN_SUMMARY.json"

    buy_intents.to_csv(buy_path, index=False)
    manual_review_queue.to_csv(review_path, index=False)

    summary = {
        "timestamp": ts,
        "mode": mode_label,
        "input_file": str(input_path),
        "artifacts": {
            "buy_intents_csv": str(buy_path),
            "manual_review_queue_csv": str(review_path),
            "run_summary_json": str(summary_path),
        },
        "counts": {
            "total_rows": int(len(decisions)),
            "buy_rows": int(len(buy_df)),
            "manual_review_rows": int(len(manual_df)),
            "skip_rows": int(len(skip_df)),
            "buy_intents_ready": int(len(buy_intents)),
            "unclassified_rows": int(classification_metrics["unclassified_rows"]),
            "invalid_symbol_rows": int(classification_metrics["invalid_symbol_rows"]),
            "buy_rows_missing_symbol_excluded": int(excluded_missing_symbol),
        },
        "safety": {
            "dry_run_default": True,
            "submit_paper_requested": bool(args.submit_paper),
            "live_routing_blocked": True,
            "account": effective_account,
            "broker_reported_accounts": broker_reported_accounts,
            "tws_host": config.tws_host,
            "tws_port": config.tws_port,
        },
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Paper executor run complete")
    logger.info(f"BUY intents artifact: {buy_path}")
    logger.info(f"MANUAL_REVIEW queue: {review_path}")
    logger.info(f"Summary JSON: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
