from zenml import pipeline

from steps.load_events_step import load_events_step
from steps.compute_kpis_step import compute_kpis_step

@pipeline
def evaluation_pipeline():
    events = load_events_step()
    results = compute_kpis_step(events)
    return results
