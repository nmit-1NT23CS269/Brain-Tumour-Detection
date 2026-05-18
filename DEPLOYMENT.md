# Deployment Guide

## Local deployment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

1. Push the project to a Git repository.
2. Ensure `requirements.txt` is committed.
3. Set the entrypoint to `app.py`.
4. Upload or mount the trained model and required artifacts.

## Server deployment

Use a Windows or Linux host with:

- Python `3.10` or `3.11`
- TensorFlow `2.15.x`
- Enough RAM for transfer-learning inference and training

Run with:

```powershell
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## Production notes

- Replace the demo OTP flow with a real SMS provider.
- Back up `brain_tumour.db`.
- Keep trained model files under `models/exports/`.
- Store sensitive configuration in environment variables.
