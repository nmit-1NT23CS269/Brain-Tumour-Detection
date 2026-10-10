# NeuroScan AI Contribution Guidelines

This document outlines team file ownership and git workflow guidelines to enable parallel development between backend engine updates and separate frontend integration without merge conflicts or breaking changes.

---

## 1. File Ownership Split

To preserve modular boundaries and prevent conflicts during parallel development:

| Module / Path | Owner | Description |
| :--- | :--- | :--- |
| `backend/` | Backend Team | Scan orchestration & prediction pipeline service logic |
| `model/` & `models/` | Backend Team | Deep learning model wrappers, architecture definitions, & inference |
| `authentication/` | Backend Team | User authentication, password hashing, and OTP verification logic |
| `database/` | Backend Team | SQLite schema, migration scripts, & database ORM/data access methods |
| `explainability/` | Backend Team | Grad-CAM heatmap generation and spatial localization algorithms |
| `segmentation/` | Backend Team | U-Net tumour segmentation inference & heuristic fallbacks |
| `preprocessing/` | Backend Team | MRI image quality gating, OOD checks, resizing, and normalization |
| `reports/` | Backend Team | ReportLab PDF generation engines and documentation report generators |
| `api/` | Backend Team | FastAPI REST API endpoints, Pydantic schemas, and JWT middleware |
| `docs/` | Shared | API specs (`docs/INTEGRATION.md`) and mock JSON payloads |
| `config.py` | Backend Team | Central environment variable configuration & directory paths |
| `frontend/` | Frontend Team | Reusable UI components, themes, and design design tokens |
| `app.py` | Shared | Streamlit main application runner (UI views managed by Frontend Team) |

---

## 2. Git Branching & Pull Request Workflow

### **Branch Naming Conventions**
- `v1-working` (Tag): Baseline working snapshot before backend decoupling.
- `feature/backend-ready`: Active development branch for UI-free backend engine & FastAPI layer.
- `feature/frontend`: Active development branch for teammate's frontend framework integration.
- `feature/<short-description>`: Individual task feature branches.

### **Workflow Rules**
1. **Commit Hygiene:** Keep commits small, self-contained, and focused on single feature steps with descriptive commit messages.
2. **Never Break Baseline Behavior:** The main application entry point (`streamlit run app.py`) must remain 100% functional at the end of every step.
3. **Automated Testing:** Run `.venv\Scripts\python.exe -m pytest` before submitting a Pull Request. All test cases must pass 100% cleanly.
4. **Secrets Policy:** Never commit secrets, `.env` files, or production credentials to git. Use `.env.example` as a reference template.
