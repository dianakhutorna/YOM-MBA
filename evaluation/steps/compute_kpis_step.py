from zenml import step
import pandas as pd
from metrics.kpis import ctr, conversion_rate, avg_revenue

@step
def compute_kpis_step(events: pd.DataFrame) -> dict:
    return {
        "ctr": ctr(events),
        "conversion_rate": conversion_rate(events),
        "avg_revenue": avg_revenue(events),
    }
