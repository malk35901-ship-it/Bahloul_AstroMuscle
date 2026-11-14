# Astra EMS · Virtual Operations Console

An astronaut muscle-monitoring dashboard with a transparent hybrid architecture:

`local EMG features → baseline-relative signal model → condition class → policy agent → virtual suit/bed state`

## Non-negotiable boundary

**SIMULATION ONLY.** The virtual EMS agent cannot activate a physical bed, suit, electrodes, or other hardware. This repository contains no device driver, hardware API, stimulation settings, waveform, schedule, or treatment command.

The condition classes (`Stable`, `Watch`, `Elevated change flag`) are dashboard research labels, not diagnoses. The model is a transparent, untrained baseline-relative scoring rule. Its confidence number represents data sufficiency (number of longitudinal comparable observations), **not** predictive or clinical accuracy.

## NASA open-data provenance

The dashboard exposes these NASA public sources in its “NASA open-data provenance and scope” panel:

- [Flight Crew Physiological Data for Crew State Monitoring](https://data.nasa.gov/dataset/flight-crew-physiological-data-for-crew-state-monitoring) — NASA’s open pilot/copilot physiological time-series and ML dataset. It is not astronaut EMG, so it is not used to train or score this application.
- [NASA Open Science Data Repository](https://visualization.genelab.nasa.gov/data/) — public space-biology data, including human skeletal-muscle/bed-rest studies such as OSD-195. These are not session-level surface-EMG references.
- [ISS muscle lab-on-chip dataset](https://data.nasa.gov/dataset/microgravity-accelerates-skeletal-muscle-degeneration-functional-and-transcriptomic-insigh) — electrically stimulated bioengineered muscle tissue data in microgravity. It is not astronaut physiological data and is not used to set virtual actions.

The local signal records created by `seed_demo.py` are generated records for exercising the dashboard, clearly separated from the NASA research-source context.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python seed_demo.py --reset
streamlit run app.py
```

Run checks with:

```powershell
pytest -q
```

## Project layout

- `app.py` — Streamlit operations console, body-region map, trends, metrics, and provenance panel
- `decision_engine.py` — interpretable change score and virtual policy agent
- `database.py` — local records and immutable-in-spirit virtual decision history
- `nasa_data.py` — NASA public-data provenance and scope statements
- `seed_demo.py` — reproducible local dashboard records plus virtual decision history

## Limitations

No open NASA source located in this work provides correctly matched astronaut surface-EMG records for training or validating this model. Therefore no accuracy, sensitivity, specificity, efficacy, or clinical claim is made. The data-quality metric is coverage only, and all decisions remain virtual software states.
