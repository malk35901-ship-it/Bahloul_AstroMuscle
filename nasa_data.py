"""NASA open-data provenance used as research context, never as substitute EMG labels."""

NASA_OPEN_DATA = (
    {"name": "Flight Crew Physiological Data for Crew State Monitoring", "url": "https://data.nasa.gov/dataset/flight-crew-physiological-data-for-crew-state-monitoring", "use": "Open physiological time-series dataset collected with pilot/copilot pairs in a flight simulator; listed by NASA as ML-training data.", "limitation": "Not astronaut EMG and not used to train, score, or validate this dashboard."},
    {"name": "NASA OSDR: human skeletal-muscle disuse / bed-rest studies", "url": "https://visualization.genelab.nasa.gov/data/", "use": "NASA's public space-biology repository includes human skeletal-muscle and bed-rest studies such as OSD-195.", "limitation": "These are biological-expression studies, not session-level surface-EMG reference records."},
    {"name": "ISS muscle lab-on-chip dataset", "url": "https://data.nasa.gov/dataset/microgravity-accelerates-skeletal-muscle-degeneration-functional-and-transcriptomic-insigh", "use": "Public NASA dataset on electrically stimulated bioengineered muscle tissue under microgravity.", "limitation": "Cell/tissue data cannot be treated as astronaut physiology or used to set virtual EMS actions."},
)
