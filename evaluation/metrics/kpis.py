import pandas as pd

def ctr(events: pd.DataFrame) -> float:
    """Click-through rate."""
    recs = len(events[events["event_type"] == "recommendation"])
    clicks = len(events[events["event_type"] == "click"])
    return clicks / recs if recs > 0 else 0.0


def conversion_rate(events: pd.DataFrame) -> float:
    """Conversion rate: purchases / recommendations."""
    recs = len(events[events["event_type"] == "recommendation"])
    purchases = len(events[events["event_type"] == "purchase"])
    return purchases / recs if recs > 0 else 0.0


def avg_revenue(events: pd.DataFrame) -> float:
    return events[events["event_type"] == "purchase"]["revenue"].mean()
