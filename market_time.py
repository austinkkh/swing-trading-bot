"""
Market clock — every time gate runs on US/Eastern, whatever the Mac's time zone.

The bot's rules are written in exchange time (9:30 open, 10-minute opening range,
16:00 close). Comparing them with the local clock breaks as soon as the Mac isn't on
Eastern time, so time-gated code calls now_et() instead of datetime.now().
"""

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")

# NYSE full-day closures (update yearly: nyse.com/markets/hours-calendars)
NYSE_HOLIDAYS = {
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3), date(2026, 5, 25),
    date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26), date(2026, 12, 25),
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26), date(2027, 5, 31),
    date(2027, 6, 18), date(2027, 7, 5), date(2027, 9, 6), date(2027, 11, 25), date(2027, 12, 24),
}


# NYSE 13:00 ET early closes (same source; verify each year). A daily bar is complete after these.
NYSE_EARLY_CLOSES = {
    date(2026, 11, 27), date(2026, 12, 24),
    date(2027, 11, 26),
}
REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)


def market_close_time(d: date) -> time:
    """Regular-session close on a trading day (13:00 on early-close days, else 16:00)."""
    return EARLY_CLOSE if d in NYSE_EARLY_CLOSES else REGULAR_CLOSE


def in_regular_session(ts: datetime) -> bool:
    """True while a regular session is open at naive-ET time ts (today's daily bar still forming)."""
    return is_trading_day(ts.date()) and REGULAR_OPEN <= ts.time() < market_close_time(ts.date())


def now_et() -> datetime:
    """Current US/Eastern time as a naive datetime (compares directly with naive ET times)."""
    return datetime.now(ET).replace(tzinfo=None)


def calendar_covers(d: date) -> bool:
    """True while NYSE_HOLIDAYS lists d's year. Past the table every weekday reads as a trading day,
    so the order modules refuse to run (fail closed) until the table is extended."""
    return d.year <= max(h.year for h in NYSE_HOLIDAYS)


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in NYSE_HOLIDAYS


def bar_lag_minutes(last_bar_start: datetime, bar_minutes: float, now: datetime) -> float:
    """How far the data trails the clock: minutes from the newest bar's END to now.
    <= 0 means that bar is still forming (real-time data); ~15 means 15-minute delayed data."""
    end = last_bar_start + pd.Timedelta(minutes=bar_minutes)
    return round((now - end).total_seconds() / 60, 1)


def parse_ibkr_times(values) -> pd.Series:
    """
    IBKR bar dates -> naive US/Eastern datetimes. Handles '20260724 09:30:00 US/Eastern'
    (intraday, with a zone), '20260724 09:30:00', '20260724' (daily) and datetime objects.
    A plain pd.to_datetime can't parse the zone suffix, which silently broke ORB detection.
    """
    s = pd.Series(values)
    if not len(s) or not isinstance(s.iloc[0], str):
        return pd.to_datetime(s)
    parts = s.str.strip().str.split(' ')
    stamp = parts.str[:2].str.join(' ')
    zone = parts.str[2]
    fmt = '%Y%m%d %H:%M:%S' if parts.str.len().iloc[0] >= 2 else '%Y%m%d'
    out = pd.to_datetime(stamp if fmt != '%Y%m%d' else parts.str[0], format=fmt)
    if zone.notna().any():
        converted = [
            ts.replace(tzinfo=ZoneInfo(z)).astimezone(ET).replace(tzinfo=None) if isinstance(z, str) else ts
            for ts, z in zip([t.to_pydatetime() for t in out], zone)
        ]
        out = pd.Series(pd.to_datetime(converted), index=s.index)
    return out
