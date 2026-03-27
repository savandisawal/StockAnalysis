"""NSE India holiday calendar using exchange_calendars."""

from datetime import date, datetime

import exchange_calendars as xcals

# NSE India calendar identifier in exchange_calendars
_NSE_CALENDAR = xcals.get_calendar("XBOM")  # BSE/NSE share the same holidays


def is_market_open(check_date: date | None = None) -> bool:
    """Check if NSE is open on a given date."""
    check_date = check_date or date.today()
    ts = datetime(check_date.year, check_date.month, check_date.day)
    return _NSE_CALENDAR.is_session(ts)


def last_trading_day(before_date: date | None = None) -> date:
    """Get the most recent trading day on or before the given date."""
    before_date = before_date or date.today()
    ts = datetime(before_date.year, before_date.month, before_date.day)

    if _NSE_CALENDAR.is_session(ts):
        return before_date

    prev = _NSE_CALENDAR.previous_close(ts)
    return prev.date()


def next_trading_day(after_date: date | None = None) -> date:
    """Get the next trading day after the given date."""
    after_date = after_date or date.today()
    ts = datetime(after_date.year, after_date.month, after_date.day)

    if _NSE_CALENDAR.is_session(ts):
        nxt = _NSE_CALENDAR.next_open(ts)
    else:
        # Find next open from this date
        nxt = _NSE_CALENDAR.next_open(ts)

    return nxt.normalize().date()


def trading_days_between(start: date, end: date) -> list[date]:
    """Get all trading days in a date range (inclusive)."""
    start_ts = datetime(start.year, start.month, start.day)
    end_ts = datetime(end.year, end.month, end.day)
    sessions = _NSE_CALENDAR.sessions_in_range(start_ts, end_ts)
    return [s.date() for s in sessions]
