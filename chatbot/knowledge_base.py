"""Static Q&A knowledge base for the Mastishka assistant.

Each intent has:
- id: internal name
- patterns: example phrasings / keywords used for matching
- responses: the reply text (kept short, non-diagnostic)
"""

from __future__ import annotations

INTENTS: list[dict] = [
    {
        "id": "greeting",
        "patterns": ["hi", "hello", "hey", "namaste", "good morning", "good evening", "yo"],
        "responses": [
            "Hi, I'm Mastishka \U0001F9E0 — NeuroScan AI's assistant. Ask me about tumour types, "
            "how to use the app, or your latest scan result.",
        ],
    },
    {
        "id": "goodbye",
        "patterns": ["bye", "goodbye", "see you", "thanks bye", "exit", "quit"],
        "responses": ["Take care! Come back anytime you have a question about your scans."],
    },
    {
        "id": "thanks",
        "patterns": ["thank you", "thanks", "thx", "appreciate it", "thankyou"],
        "responses": ["You're welcome! Let me know if there's anything else about NeuroScan AI I can explain."],
    },
    {
        "id": "what_is_mastishka",
        "patterns": ["who are you", "what is mastishka", "what are you", "your name"],
        "responses": [
            "I'm Mastishka (\u092e\u0938\u094d\u0924\u093f\u0937\u094d\u0915, Hindi for 'brain') \U0001F9E0 — a simple rule-based "
            "assistant built into NeuroScan AI. I can answer basic questions about the platform, "
            "tumour classes, and your scan results. I'm not a diagnostic tool myself.",
        ],
    },
    {
        "id": "what_is_app",
        "patterns": [
            "what is neuroscan", "what does this app do", "what is this project",
            "what is this website", "what is this platform",
        ],
        "responses": [
            "NeuroScan AI is a Streamlit-based platform that analyses brain MRI scans using deep "
            "learning to flag possible tumour patterns (Glioma, Meningioma, Pituitary) or a Normal "
            "scan. It adds explainability (Grad-CAM), a segmentation overlay, a dashboard, and PDF "
            "reports. It's for education/research and is not a diagnostic device.",
        ],
    },
    {
        "id": "how_to_use",
        "patterns": [
            "how do i use", "how to scan", "how to upload", "how do i analyze",
            "steps to use", "how does it work", "workflow",
        ],
        "responses": [
            "1) Register/login (OTP verification). 2) Go to 'Upload & Scan' and choose an MRI image. "
            "3) Optionally add patient details. 4) Click 'Analyze MRI'. The app preprocesses the "
            "image, runs the model, shows Grad-CAM + segmentation, and lets you download a PDF report.",
        ],
    },
    {
        "id": "tumour_types",
        "patterns": [
            "tumour types", "tumor types", "what classes", "what categories",
            "glioma meningioma pituitary", "which tumours can it detect",
        ],
        "responses": [
            "The model classifies scans into 4 classes: **Glioma**, **Meningioma**, **Pituitary** "
            "tumour, and **Normal** (no tumour pattern detected). Ask me about any one of these "
            "specifically, e.g. 'what is glioma'.",
        ],
    },
    {
        "id": "glioma",
        "patterns": ["glioma", "what is glioma"],
        "responses": [
            "Glioma: a tumour arising from glial (support) cells in the brain or spinal cord. In "
            "this app it's treated as the most severe class — a Glioma flag suggests specialist "
            "neuro-oncology review is advised.",
        ],
    },
    {
        "id": "meningioma",
        "patterns": ["meningioma", "what is meningioma"],
        "responses": [
            "Meningioma: a tumour arising from the meninges, the membranes surrounding the brain "
            "and spinal cord. Most are slow-growing/benign, but the app still flags it for "
            "specialist correlation with contrast MRI.",
        ],
    },
    {
        "id": "pituitary",
        "patterns": ["pituitary", "what is pituitary"],
        "responses": [
            "Pituitary tumour: a growth in the pituitary gland at the base of the brain, which can "
            "affect hormone levels. The app suggests endocrine and neurosurgical review if flagged.",
        ],
    },
    {
        "id": "normal_class",
        "patterns": ["normal class", "what does normal mean", "no tumour"],
        "responses": [
            "'Normal' means the model did not detect a dominant tumour-like pattern in the scan. "
            "Routine clinical correlation is still recommended — a screening tool isn't a replacement "
            "for a radiologist.",
        ],
    },
    {
        "id": "confidence",
        "patterns": [
            "what is confidence", "confidence score", "confidence mean",
            "how confident", "confidence level",
        ],
        "responses": [
            "Confidence is the model's probability for its top predicted class. Roughly: \u226590% "
            "Very high, \u226575% High, \u226560% Moderate, below that Low (clinical review strongly "
            "advised). Higher confidence means the model is more certain, not that the finding is "
            "medically confirmed.",
        ],
    },
    {
        "id": "severity",
        "patterns": ["what is severity", "severity mean", "severity level"],
        "responses": [
            "Severity is a qualitative estimate combining the predicted class and confidence "
            "(e.g. Normal \u2192 None, Glioma at high confidence \u2192 Critical). It's a rough "
            "triage signal, not a clinical grading."
        ],
    },
    {
        "id": "gradcam",
        "patterns": ["grad-cam", "gradcam", "grad cam", "what is heatmap", "explainability"],
        "responses": [
            "Grad-CAM produces a heatmap over the MRI showing which regions most influenced the "
            "model's prediction. Warmer colours = higher influence. It's a way to make the model's "
            "reasoning more transparent, not a segmentation of the tumour itself.",
        ],
    },
    {
        "id": "segmentation",
        "patterns": ["segmentation", "segmentation overlay", "outline the tumour", "contour"],
        "responses": [
            "The segmentation overlay uses OpenCV contour detection on the preprocessed MRI to "
            "visually outline a candidate tumour region. It's a heuristic visual aid, not a "
            "clinically validated tumour boundary.",
        ],
    },
    {
        "id": "models",
        "patterns": [
            "which models", "what model", "models used", "model architecture",
            "vgg16", "resnet50", "efficientnet", "mobilenet",
        ],
        "responses": [
            "NeuroScan AI supports transfer learning with 4 backbones: VGG16, ResNet50, "
            "EfficientNetB0, and MobileNetV2. The best-performing trained model is auto-selected "
            "for inference; you can compare them on the Training Hub page.",
        ],
    },
    {
        "id": "accuracy",
        "patterns": ["accuracy", "how accurate", "how good is the model", "performance metrics"],
        "responses": [
            "Model accuracy/precision/recall/F1 depend on the training run and are shown on the "
            "'Training Hub' page once training has been done (`python model/train.py`). There's no "
            "single fixed number — check that page for the latest leaderboard.",
        ],
    },
    {
        "id": "pdf_report",
        "patterns": ["pdf report", "download report", "generate report", "export report"],
        "responses": [
            "After analysing a scan, click 'Download PDF report' on the Upload & Scan page. It "
            "includes the original MRI, preprocessed image, Grad-CAM, segmentation overlay, "
            "prediction, confidence, and an AI-generated summary.",
        ],
    },
    {
        "id": "dashboard",
        "patterns": ["dashboard", "analytics", "stats", "statistics"],
        "responses": [
            "The Dashboard page shows your total scans, tumour vs normal case counts, MRI quality "
            "breakdown, and recent scan history, once you're logged in and have run some scans.",
        ],
    },
    {
        "id": "login_help",
        "patterns": [
            "how to login", "how to register", "sign up", "otp", "forgot password",
            "can't login", "cannot log in", "account",
        ],
        "responses": [
            "Go to 'Register' to create an account with username/email/phone/password — you'll get "
            "a demo OTP shown on-screen to verify your phone. Then use 'Login' with your username, "
            "email, or phone plus your password.",
        ],
    },
    {
        "id": "quality",
        "patterns": ["mri quality", "image quality", "quality label", "blurry"],
        "responses": [
            "During preprocessing the app estimates MRI quality (e.g. Good/Fair/Poor) using image "
            "heuristics like sharpness and contrast. Lower quality scans should be treated with "
            "extra caution when interpreting results.",
        ],
    },
    {
        "id": "diagnosis_disclaimer",
        "patterns": [
            "is this a diagnosis", "can i trust this", "is this accurate enough",
            "should i believe this", "is this a real diagnosis", "medical advice",
        ],
        "responses": [
            "No — NeuroScan AI is for educational and research purposes only. It must not be used "
            "as a sole diagnostic system or a substitute for a qualified radiologist or clinician. "
            "Please always follow up with a medical professional.",
        ],
    },
    {
        "id": "dataset",
        "patterns": ["dataset", "training data", "how many images", "data source"],
        "responses": [
            "The model was trained on an MRI archive with 4 balanced classes (Glioma, Meningioma, "
            "Pituitary, Normal) split into Training and Testing sets (~1400 training / ~400 testing "
            "images per class in the reference dataset structure).",
        ],
    },
    {
        "id": "help",
        "patterns": ["help", "what can you do", "what can you answer", "options", "menu"],
        "responses": [
            "I can help with: what NeuroScan AI does, how to use it, the tumour classes, what "
            "confidence/severity/Grad-CAM/segmentation mean, models used, login/registration, and "
            "explaining your latest scan result. Just ask in plain language!",
        ],
    },
]

# Suggested quick-reply questions shown as buttons in the chat UI.
SUGGESTED_QUESTIONS: list[str] = [
    "What is NeuroScan AI?",
    "How do I scan an MRI?",
    "What does confidence mean?",
    "Explain my last result",
    "What is Grad-CAM?",
]

FALLBACK_RESPONSES: list[str] = [
    "I'm not sure about that one — I can only answer basic questions about NeuroScan AI (tumour "
    "types, how to use the app, confidence/severity/Grad-CAM, or your latest scan). Try rephrasing, "
    "or type 'help' to see what I can do.",
]
