# Setup Instructions

## 1. Install Python

Use Python `3.10` or `3.11`.

## 2. Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Launch the Streamlit app

```powershell
streamlit run app.py
```

## 5. Open the app

Visit `http://localhost:8501`.

## Notes

- The uploaded dataset archive is already extracted under `data/raw/archive`.
- If TensorFlow install fails, verify that you are using Python `3.10` or `3.11`.
