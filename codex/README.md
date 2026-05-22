# Codex Cloud Environment

Use this repository with a Codex web cloud environment configured as follows:

- Base image: `universal`
- Python version: `3.11.9`
- Setup script: `bash codex/setup.sh`
- Maintenance script: `bash codex/maintenance.sh`

Recommended environment variables:

- `MPLBACKEND=Agg`
- `PYTHONUNBUFFERED=1`
- `NNKNN_DEVICE=cpu`

Quick validation commands inside a task:

- `python codex/smoke_test.py --mode imports`
- `python codex/smoke_test.py --mode train`
