"""
Scan Display — end-of-scan terminal view (rich)

Renders the scanner's tiers as colored panels: actionable entries first,
then leaders waiting, watchlist, extended names, and data warnings.
Display only: reads the scanner's results, never changes them.
"""

from datetime import datetime
from typing import Optional

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import config

try:
    # Execution engine's stop cap, so the notes match what would actually be routed
    from execution_engine import ExecutionPolicy
    EXEC_MAX_STOP_PCT: Optional[float] = ExecutionPolicy().max_stop_pct
except ImportError:
    EXEC_MAX_STOP_PCT = None

ACCENT, GOOD, WARN, BAD, DIM = "bright_cyan", "green3", "gold1", "red1", "grey58"
SCORE_MAX = 110            # calculate_priority_rating's true maximum
IBD_UNIVERSE_STALE_DAYS = 7  # the IBD data table is a weekly export
WATCH_ROWS = 10


# ═══════════════════════════════════════════════════════════════════════════
# CELL FORMATTERS
# ═══════════════════════════════════════════════════════════════════════════
def _money(value) -> Text:
    return Text("—", style=DIM) if value is None else Text(f"{value:,.2f}")


def _signed_pct(value) -> Text:
    return Text("n/a", style=BAD) if value is None else Text(f"{value:+.0f}%")


def _risk(stop_pct) -> Text:
    """Stop distance, colored by size: ≤3% green, ≤6% yellow, wider or missing red."""
    if stop_pct is None:
        return Text("n/a", style=BAD)
    style = GOOD if stop_pct <= 3 else WARN if stop_pct <= 6 else BAD
    return Text(f"{stop_pct:.1f}%", style=style)


def _score_bar(score, width: int = 8) -> Text:
    score = score or 0
    filled = round(width * min(score, SCORE_MAX) / SCORE_MAX)
    return (Text("█" * filled, style=ACCENT) + Text("░" * (width - filled), style=DIM)
            + Text(f" {score:.0f}", style=ACCENT))


def _stop_adr(stock) -> Text:
    stop_pct, adr = stock.get('stop_loss_pct'), stock.get('adr') or 0
    if stop_pct is None or adr <= 0:
        return Text("n/a", style=DIM)
    return Text(f"{stop_pct / adr:.2f}×")


def _vs_200d(stock) -> Text:
    """Distance above the 200-day; yellow when it's that far up with no VCP base."""
    dist = stock.get('distance_from_200ma')
    if dist is None:
        return Text("n/a", style=DIM)
    return Text(f"{dist:+.0f}%", style=WARN if stock.get('extended_no_base') else "")


def _base(stock) -> Text:
    """Base number since the stage low, then VCP contractions and distance from the pivot.

    Base number: #1-2 plain, #3 yellow, #4+ red (late-stage bases fail more often).
    VCP: green within ±5% of the pivot (Minervini's buy range), yellow more than
    5% past it (broken out, extended from the buy point), dim further below.
    """
    number = stock.get('base_number')
    if number:
        label = Text(f"#{number} ", style=BAD if number >= 4 else WARN if number == 3 else "")
    else:
        label = Text("#– ", style=DIM)
    if not stock.get('has_real_vcp'):
        return label
    n = stock.get('vcp_num_contractions') or 0
    dist = stock.get('vcp_pivot_distance_pct') or 0
    style = GOOD if -5 <= dist <= 5 else WARN if dist > 5 else DIM
    return label + Text(f"VCP {n}c {dist:+.0f}%", style=style)


def _no_base_note(stock) -> Optional[Text]:
    if not stock.get('extended_no_base'):
        return None
    return Text(f"no base, 200d {stock['distance_from_200ma']:+.0f}%", style=WARN)


def _entry_notes(stock) -> Text:
    """Short flags worth seeing before acting on an entry."""
    notes = []
    late = _no_base_note(stock)
    if late:
        notes.append(late)
    eps_rating = stock.get('ibd_eps_rating')
    if eps_rating is not None:
        min_rating = getattr(config, 'TIER_BUY_MIN_EPS_RATING', 80)  # scanner's BUY threshold
        notes.append(Text(f"EPS {eps_rating:.0f}", style=WARN if eps_rating < min_rating else DIM))
    elif stock.get('eps_yoy') is None:
        notes.append(Text("EPS n/a", style=DIM))
    acc_dist = stock.get('ibd_acc_dist') or ''
    if acc_dist[:1] in ('D', 'E'):  # institutions net selling
        notes.append(Text(f"A/D {acc_dist}", style=WARN))
    stop_pct = stock.get('stop_loss_pct')
    if EXEC_MAX_STOP_PCT is not None and stop_pct is not None and stop_pct > EXEC_MAX_STOP_PCT:
        notes.append(Text(f"over {EXEC_MAX_STOP_PCT:g}% exec cap", style=WARN))
    if stock.get('is_power_stock'):
        notes.append(Text("power stock", style=GOOD))
    return Text(" · ", style=DIM).join(notes) if notes else Text("")


def _section(body, title: str, subtitle: str, color: str) -> Panel:
    return Panel(body, title=f"[bold {color}]{title}[/]", subtitle=f"[{DIM}]{subtitle}[/]",
                 title_align="left", subtitle_align="right", border_style=color, padding=(0, 1))


# ═══════════════════════════════════════════════════════════════════════════
# SECTIONS
# ═══════════════════════════════════════════════════════════════════════════
def _header(scanner, scan_type: str) -> Panel:
    ts = scanner.scan_timestamp or datetime.now()
    top = Table.grid(expand=True)
    top.add_column(ratio=1)
    top.add_column(justify="right")
    top.add_row(Text("MOMENTUM SCAN", style=f"bold {ACCENT}"),
                Text(f"{ts:%a %Y-%m-%d · %H:%M} · {scan_type.replace('_', ' ')}", style=DIM))

    regime = scanner.market_regime
    if regime:
        color = BAD if scanner.watch_only_mode else GOOD
        meaning = ("watch-only: elite leaders are the only possible entries" if scanner.watch_only_mode
                   else f"new positions allowed · size ×{regime.position_size_multiplier:g}")
        regime_line = Text.assemble((f"● {regime.status}", f"bold {color}"), (f"   {meaning}", DIM))
    else:
        regime_line = Text("● REGIME UNKNOWN   detection failed: watch-only as a precaution", style=f"bold {WARN}")

    counts = getattr(scanner, 'scan_counts', {}) or {}
    issues = getattr(scanner, 'scan_issues', {}) or {}
    health = Text.assemble((str(counts.get('scanned', '?')), "bold"), (" scanned   ", DIM),
                           (str(counts.get('passed', '?')), "bold"), (" passed filters", DIM))
    no_data = len(issues.get('no_data', []))
    if no_data:
        health += Text.assemble(("   ⚠ ", WARN), (str(no_data), f"bold {WARN}"), (" no data", DIM))
    stale = _ibd_staleness(scanner)
    if stale:
        days, label = stale
        health += Text.assemble((f"   ⚠ {label}", f"bold {WARN}"), (f" {days} days old", DIM))
    elif getattr(scanner, 'ibd_universe_date', None):
        health += Text(f"   IBD data {scanner.ibd_universe_date:%b %d}", style=DIM)

    return Panel(Group(top, Text(""), regime_line, health), box=box.HEAVY, border_style=ACCENT, padding=(0, 2))


def _entries_table(stocks, entry_key: str) -> Table:
    t = Table(box=box.SIMPLE_HEAVY, expand=True, header_style=f"bold {DIM}", pad_edge=False)
    for col, justify in [("Symbol", "left"), ("Entry", "left"), ("Base·pivot", "left"), ("Price", "right"),
                         ("Stop", "right"), ("Risk", "right"), ("Stop/ADR", "right"), ("RS", "right"),
                         ("YTD", "right"), ("Score", "left")]:
        t.add_column(col, justify=justify, no_wrap=True)
    t.add_column("Notes", ratio=1, overflow="fold")
    for s in stocks:
        t.add_row(Text(s['symbol'], style=f"bold {GOOD}"), s.get(entry_key) or "—", _base(s),
                  _money(s.get('current_price')), _money(s.get('stop_loss')), _risk(s.get('stop_loss_pct')),
                  _stop_adr(s), f"{s.get('rs_rating') or 0:.0f}", _signed_pct(s.get('ytd_return')),
                  _score_bar(s.get('priority_rating')), _entry_notes(s))
    return t


def _actionable(scanner) -> Panel:
    elite = scanner.elite_buy_results
    regular = [] if scanner.watch_only_mode else scanner.buy_results
    total = len(elite) + len(regular)
    if not total:
        why = ("market regime blocks new entries and no elite leader has a clean entry"
               if scanner.watch_only_mode else "no setup passed every BUY check today")
        return _section(Text(f"Nothing actionable: {why}.", style=DIM), "⚡ ACTIONABLE NOW · 0",
                        "elite leaders and BUY setups with a clean entry", DIM)
    parts = []
    if elite:
        parts.append(_entries_table(elite, 'elite_entry_setup'))
    if regular:
        parts.append(Text("BUY setups", style=f"bold {GOOD}"))
        parts.append(_entries_table(regular, 'setup_type'))
    return _section(Group(*parts), f"⚡ ACTIONABLE NOW · {total}",
                    "elite leaders and BUY setups with a clean entry", GOOD)


def _blocked_buys(scanner) -> Optional[Panel]:
    """BUY setups that exist but can't be taken because the regime is watch-only."""
    if not scanner.watch_only_mode or not scanner.buy_results:
        return None
    return _section(_entries_table(scanner.buy_results, 'setup_type'),
                    f"⛔ BUY SETUPS BLOCKED BY REGIME · {len(scanner.buy_results)}",
                    "would be actionable in a healthier market", WARN)


def _rs_leaders(scanner) -> Optional[Panel]:
    """Technical leaders traded without the earnings check, at reduced size."""
    leaders = getattr(scanner, 'rs_leader_results', [])
    if not leaders:
        return None
    mult = leaders[0].get('size_multiplier', 1.0)
    size = f"{mult:g}× size"
    if scanner.watch_only_mode:
        return _section(_entries_table(leaders, 'setup_type'), f"🏃 RS LEADERS · BLOCKED BY REGIME · {len(leaders)}",
                        f"no earnings check · {size} when the market allows", WARN)
    return _section(_entries_table(leaders, 'setup_type'), f"🏃 RS LEADERS · {len(leaders)}",
                    f"technical strength, no earnings check · {size}", GOOD)


def _waiting(scanner) -> Optional[Panel]:
    holds = scanner.elite_hold_results
    if not holds:
        return None
    t = Table(box=None, expand=True, show_header=False, pad_edge=False)
    for justify in ("left", "right", "right", "right", "left"):
        t.add_column(justify=justify, no_wrap=True)
    t.add_column(ratio=1, overflow="fold")
    for s in holds:
        style = BAD if s.get('stop_status') == 'NO_STOP' else DIM
        reason = Text(s.get('hold_reason', ''), style=style)
        late = _no_base_note(s)
        if late:
            reason += Text(" · ", style=DIM) + late
        t.add_row(Text(s['symbol'], style=f"bold {WARN}"), _money(s.get('current_price')),
                  f"RS {s.get('rs_rating') or 0:.0f}", _signed_pct(s.get('ytd_return')), _base(s), reason)
    return _section(t, f"📌 LEADERS WAITING · {len(holds)}", "strong, but no entry today", WARN)


def _watchlist(scanner) -> Optional[Panel]:
    watch = scanner.watch_results
    if not watch:
        return None
    t = Table(box=box.SIMPLE, expand=True, header_style=f"bold {DIM}", pad_edge=False, row_styles=["", "on grey11"])
    # Rows are already in priority order, so no rank column; narrow bars keep reasons on one line
    for col, justify in [("Symbol", "left"), ("Price", "right"), ("RS", "right"), ("YTD", "right"),
                         ("ATR×", "right"), ("200d", "right"), ("Base·pivot", "left"), ("Stop", "right"),
                         ("Score", "left")]:
        t.add_column(col, justify=justify, no_wrap=True)
    t.add_column("Why not a buy", ratio=1, overflow="fold")
    for s in watch[:WATCH_ROWS]:
        reason = s.get('watch_reason') or s.get('moved_reason') or ''
        style = BAD if s.get('stop_status') == 'NO_STOP' else ""
        t.add_row(Text(s['symbol'], style="bold"), _money(s.get('current_price')),
                  f"{s.get('rs_rating') or 0:.0f}", _signed_pct(s.get('ytd_return')),
                  f"{s.get('atr_multiple_from_50ma') or 0:.1f}", _vs_200d(s), _base(s), _risk(s.get('stop_loss_pct')),
                  _score_bar(s.get('priority_rating'), width=5), Text(reason, style=style))
    if len(watch) > WATCH_ROWS:
        # A caption, not a row, so the note doesn't widen the Symbol column
        t.caption = f"⋯ {len(watch) - WATCH_ROWS} more in the CSV"
        t.caption_style = DIM
        t.caption_justify = "left"
    return _section(t, f"📋 WATCHLIST · {len(watch)}", "why each one isn't a buy", DIM)


def _extended(scanner) -> Panel:
    ext = scanner.extended_results
    if not ext:
        body = Text("None", style=DIM)
    else:
        body = Text("\n").join(
            Text.assemble((f"{s['symbol']:<6}", "bold"),
                          (f"{s.get('atr_multiple_from_50ma') or 0:.1f} ATR× above 50-day · ", DIM), _vs_200d(s),
                          (" vs 200d", DIM))
            for s in ext)
    return Panel(body, title=f"[bold]🚀 EXTENDED · {len(ext)}[/]", title_align="left", border_style=DIM)


def _warnings(scanner) -> Optional[Panel]:
    lines = []
    issues = getattr(scanner, 'scan_issues', {}) or {}
    labels = getattr(scanner, '_STAGE_LABELS', {})
    error_stages = getattr(scanner, '_ERROR_STAGES', ())
    days = getattr(scanner, 'secdef_days', {}) or {}
    for stage in error_stages:
        found = issues.get(stage, [])
        if stage == 'no_data':
            # Error 200 (IBKR doesn't know the ticker) gets its own line with the quarantine count
            secdef = [(s, r) for s, r in found if str(r).startswith('no_secdef')]
            found = [(s, r) for s, r in found if not str(r).startswith('no_secdef')]
            if secdef:
                n_days = getattr(scanner, 'quarantine_days', None) or _quarantine_days()
                names = ", ".join(f"{s} (day {days.get(s, 1)} of {n_days})" for s, _ in secdef[:5]) + \
                    (f" +{len(secdef) - 5}" if len(secdef) > 5 else "")
                lines.append(Text.assemble(("⚠ ", WARN), (f"No security definition ({len(secdef)}): ", "bold"), names))
                lines.append(Text("   delisted or renamed? auto-quarantined after that many scan days, "
                                  "or add to symbol_overrides.csv", style=DIM))
        if not found:
            continue
        names = ", ".join(sym for sym, _ in found[:5]) + (f" +{len(found) - 5}" if len(found) > 5 else "")
        lines.append(Text.assemble(("⚠ ", WARN), (f"{labels.get(stage, stage)} ({len(found)}): ", "bold"), names))
        lines.append(Text(f"   first error: {found[0][1][:80]}", style=DIM))

    for o in _overrides_due():
        lines.append(Text.assemble(("⚠ ", WARN), ("Override review due: ", "bold"),
                                   f"{o.symbol} {o.action} (review_after {o.review_after}): {o.reason}"))

    stale = _ibd_staleness(scanner)
    if stale and getattr(scanner, 'ibd_universe_date', None):
        lines.append(Text.assemble(("⚠ ", WARN), (f"IBD data is {stale[0]} days old", "bold")))
        lines.append(Text("   RS and the BUY EPS check use it: re-export the IBD data table", style=DIM))
    elif not getattr(scanner, 'ibd_universe_date', None):
        lines.append(Text.assemble(("⚠ ", WARN), ("No IBD universe export", "bold")))
        lines.append(Text("   No IBD ratings or EPS estimates: save Data/ibd_universe_YYYY-MM-DD.csv", style=DIM))

    regime = scanner.market_regime
    for w in (regime.warnings if regime else []):
        lines.append(Text.assemble(("⚠ ", WARN), w))

    # Known-dead names skipped on purpose: shown dim, never silent, and not a warning by itself
    info = []
    dropped = getattr(scanner, 'hygiene_dropped', []) or []
    if dropped:
        n_q = sum(1 for _, r in dropped if r.startswith('auto-quarantined'))
        names = ", ".join(sym for sym, _ in dropped[:10]) + (f" +{len(dropped) - 10}" if len(dropped) > 10 else "")
        info.append(Text(f"ℹ Skipped {len(dropped) - n_q} known dead/renamed symbols (symbol_overrides.csv)"
                         f" + {n_q} auto-quarantined: {names}", style=DIM))
    timeouts = issues.get('timeout', [])
    if timeouts:
        info.append(Text(f"ℹ IBKR timeouts (transient, not delisted) ({len(timeouts)}): "
                         + ", ".join(sym for sym, _ in timeouts[:5]), style=DIM))

    if not lines:
        if info:
            return Panel(Text("\n").join([Text("✓ No data problems", style=GOOD)] + info),
                         title=f"[bold {GOOD}]CHECK BEFORE TRADING[/]", title_align="left", border_style=GOOD)
        return Panel(Text("✓ No data problems", style=GOOD), title=f"[bold {GOOD}]CHECK BEFORE TRADING[/]",
                     title_align="left", border_style=GOOD)
    return Panel(Text("\n").join(lines + info), title=f"[bold {WARN}]CHECK BEFORE TRADING[/]",
                 title_align="left", border_style=WARN)


def _quarantine_days() -> int:
    try:
        import symbol_hygiene
        return symbol_hygiene.QUARANTINE_DAYS
    except Exception:                                            # noqa: BLE001
        return 3


def _overrides_due() -> list:
    """symbol_overrides.csv entries whose review_after date has passed (never raises)."""
    try:
        import symbol_hygiene
        return symbol_hygiene.review_due(symbol_hygiene.load_overrides())
    except Exception:                                            # noqa: BLE001
        return []


def _ibd_staleness(scanner) -> Optional[tuple]:
    """(age in days, label) when the IBD universe export is stale, else None."""
    universe_date = getattr(scanner, 'ibd_universe_date', None)
    if universe_date:
        days = (datetime.now().date() - universe_date).days
        return (days, "IBD data") if days > IBD_UNIVERSE_STALE_DAYS else None
    return None


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
def render_scan(scanner, scan_type: str = "afterhours", export_prefix: Optional[str] = None,
                console: Optional[Console] = None) -> None:
    """Print the end-of-scan view for a finished MomentumScanner run."""
    console = console or Console()
    console.print(_header(scanner, scan_type))
    console.print(_actionable(scanner))
    for panel in (_rs_leaders(scanner), _blocked_buys(scanner), _waiting(scanner), _watchlist(scanner)):
        if panel is not None:
            console.print(panel)

    bottom = Table.grid(expand=True, padding=(0, 1))
    bottom.add_column(ratio=2)
    bottom.add_column(ratio=5)
    bottom.add_row(_extended(scanner), _warnings(scanner))
    console.print(bottom)
    if export_prefix:
        console.print(Text(f"  Files → {export_prefix}_*.csv", style=DIM))
