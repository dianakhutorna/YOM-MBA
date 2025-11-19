from zenml import step
import pandas as pd

@step
def load_events_step(path: str = "evaluation/data/raw/events.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df
