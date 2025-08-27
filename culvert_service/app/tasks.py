import os
import uuid
from pathlib import Path
from .celery_app import celery
from .culvert_analysis import CulvertAnalysisSystem

# Define the base output directory relative to this file
# This makes it easier to manage paths within the container
OUTPUTS_BASE_DIR = Path('/home/appuser/app/outputs')

@celery.task(bind=True)
def run_analysis_task(self, county: str, state: str):
    """
    Celery task to run the full culvert analysis.
    Updates its state periodically to allow for progress tracking.
    """
    job_id = self.request.id
    output_dir = OUTPUTS_BASE_DIR / str(job_id)
    os.makedirs(output_dir, exist_ok=True)

    def update_status(message: str):
        self.update_state(state='PROGRESS', meta={'status': message})

    try:
        cas = CulvertAnalysisSystem()

        update_status(f'Setting location: {county}, {state}')
        cas.set_location_by_county_state(county, state)
        if cas.bbox is None:
            raise ValueError(f"Could not determine bounding box for {county}, {state}.")

        update_status('Collecting culvert data...')
        cas.collect_culvert_data()

        update_status('Collecting stream gauge data...')
        cas.collect_stream_gauge_data()

        update_status('Collecting flood event data...')
        cas.collect_flood_event_data()

        update_status('Performing proximity analysis...')
        cas.proximity_analysis(buffer_distance=5000)

        update_status('Assessing hydrologic risk...')
        cas.hydrologic_risk_assessment()

        update_status('Analyzing transportation impact...')
        cas.transportation_impact_analysis()

        update_status('Training failure prediction model...')
        cas.train_failure_prediction_model()

        update_status('Generating synthetic flood scenarios...')
        cas.generate_synthetic_flood_scenarios(n_scenarios=5)

        update_status('Generating interactive map...')
        m = cas.create_interactive_map()
        map_path = None
        if m:
            map_path = output_dir / "culvert_analysis_map.html"
            m.save(map_path)

        update_status('Generating summary report...')
        report = cas.generate_report()

        update_status('Saving data artifacts...')
        cas.save_data(output_dir=str(output_dir))

        result_payload = {
            'status': 'Complete',
            'report': report,
            'outputs': {
                'map': f"/analysis/{job_id}/map" if map_path else None,
                'culverts': f"/analysis/{job_id}/data/culverts.geojson",
                'gauges': f"/analysis/{job_id}/data/gauges.geojson",
                'flood_events': f"/analysis/{job_id}/data/flood_events.geojson",
            }
        }

        return result_payload

    except Exception as e:
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        return {'status': 'FAILURE', 'error': str(e)}
