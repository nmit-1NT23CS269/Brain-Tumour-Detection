"""Reusable Streamlit theme helpers."""

from __future__ import annotations

import streamlit as st


def inject_theme() -> None:
    """Inject a medical dashboard theme into Streamlit."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=Cinzel:wght@600&display=swap');
        :root {
          --bg-primary:#07111f;
          --bg-secondary:#0d1b2c;
          --bg-card:#12253a;
          --border:#244865;
          --text-primary:#edf6ff;
          --text-secondary:#93aec8;
          --accent:#00a6fb;
          --accent-2:#00d4a5;
          --danger:#ff6b6b;
          --warning:#fcbf49;
          --success:#6ede8a;
        }
        html, body, [class*="css"] {
          font-family:'IBM Plex Sans', sans-serif !important;
          background: radial-gradient(circle at top, #102846 0%, #07111f 55%) !important;
          color:var(--text-primary) !important;
        }
        .app-card {
          background: linear-gradient(180deg, rgba(18,37,58,.94), rgba(10,22,36,.96));
          border:1px solid var(--border);
          border-radius:18px;
          padding:1.1rem 1.2rem;
          box-shadow:0 12px 32px rgba(0,0,0,.16);
          margin-bottom:1rem;
        }
        .hero-title {
          font-family:'Cinzel', serif;
          font-size:2.4rem;
          letter-spacing:.06em;
        }
        .hero-subtitle {
          color:var(--text-secondary);
        }
        .metric-tile {
          background:rgba(7,17,31,.55);
          border:1px solid rgba(36,72,101,.8);
          border-radius:16px;
          padding:.9rem 1rem;
          text-align:center;
        }
        .metric-value {
          font-size:1.9rem;
          font-weight:700;
        }
        .metric-label {
          font-size:.8rem;
          letter-spacing:.08em;
          text-transform:uppercase;
          color:var(--text-secondary);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
