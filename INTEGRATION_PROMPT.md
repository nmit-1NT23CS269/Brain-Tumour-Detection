# Task: Integrate the "Mastishka" AI chatbot into NeuroScan AI (Brain-Tumour-Detection)

You are working inside the repository `Brain-Tumour-Detection`
(https://github.com/nmit-1NT23CS269/Brain-Tumour-Detection), a Streamlit app
for MRI brain-tumour classification. This zip adds a small, self-contained,
offline, rule-based FAQ chatbot called **Mastishka** ("brain" in Hindi) that
answers basic questions about the app (tumour types, how to use it,
confidence/severity/Grad-CAM/segmentation, models used, login help, etc.)
and can explain the user's most recent scan result. It needs no API key and
makes no network calls — it's a keyword/typo-tolerant matcher over a static
knowledge base, which keeps the capstone project self-contained and free to
run.

## What's in this zip

```
chatbot/
  __init__.py          # package exports: get_response, SUGGESTED_QUESTIONS
  knowledge_base.py     # static Q&A intents + suggested quick questions
  mastishka.py           # matching engine (tokenize, score, fallback, scan-result explainer)
app_py_changes.diff     # unified diff of the required edits to app.py
INTEGRATION_PROMPT.md   # this file
```

## Steps to apply

1. **Copy the package.**
   Copy the `chatbot/` folder as-is into the root of the repository, so you
   end up with `Brain-Tumour-Detection/chatbot/__init__.py`,
   `chatbot/knowledge_base.py`, and `chatbot/mastishka.py` sitting alongside
   the existing `authentication/`, `backend/`, `database/`, `frontend/`,
   `model/`, `models/`, `preprocessing/`, `reports/`, `segmentation/`, and
   `utils/` packages.

2. **Apply the diff to `app.py`.**
   From the repo root:
   ```bash
   git apply app_py_changes.diff
   ```
   If `git apply` fails (e.g. because `app.py` has since diverged), apply
   the following changes to `app.py` by hand instead:

   a. Add an import near the other local imports:
      ```python
      from chatbot import SUGGESTED_QUESTIONS, get_response as mastishka_response
      ```

   b. In `render_sidebar()`, add a `"🧠 Mastishka AI": "mastishka"` entry to
      **both** the `pages` dict used when logged in and the one used when
      logged out (it should work with or without an account).

   c. In `page_upload()`, right after the prediction is saved
      (`prediction_id = _save_scan_result(...)`), store the raw result dict
      in session state so Mastishka can reference it later:
      ```python
      st.session_state["last_scan_result"] = result
      ```

   d. Add a new page function (place it near `page_about()`):
      ```python
      def page_mastishka() -> None:
          _hero("🧠 Mastishka Assistant", "Ask basic questions about NeuroScan AI, tumour types, or your latest scan")

          st.session_state.setdefault(
              "mastishka_chat",
              [
                  {
                      "role": "assistant",
                      "content": (
                          "Hi, I'm **Mastishka** 🧠 — your NeuroScan AI assistant. Ask me about "
                          "tumour types, how the app works, or your latest scan result. I run entirely "
                          "offline (no external AI API), so I stick to basic, pre-defined answers."
                      ),
                  }
              ],
          )

          _card_start()
          st.caption("Quick questions")
          quick_cols = st.columns(len(SUGGESTED_QUESTIONS))
          quick_pick = None
          for col, question in zip(quick_cols, SUGGESTED_QUESTIONS):
              if col.button(question, use_container_width=True, key=f"quick_{question}"):
                  quick_pick = question
          _card_end()

          _card_start()
          for turn in st.session_state["mastishka_chat"]:
              with st.chat_message(turn["role"]):
                  st.markdown(turn["content"])
          _card_end()

          # st.chat_input always docks to the bottom of the page regardless of
          # surrounding containers, so it's kept outside the card above.
          user_message = st.chat_input("Ask Mastishka a question...")
          incoming = quick_pick or user_message

          if incoming:
              st.session_state["mastishka_chat"].append({"role": "user", "content": incoming})
              reply = mastishka_response(incoming, scan_result=st.session_state.get("last_scan_result"))
              st.session_state["mastishka_chat"].append({"role": "assistant", "content": reply})
              st.rerun()

          if st.button("Clear conversation"):
              st.session_state["mastishka_chat"] = []
              st.rerun()
      ```

   e. In `main()`, register the route:
      ```python
      routes = {
          "home": page_home,
          "register": page_register,
          "login": page_login,
          "upload": page_upload,
          "dashboard": page_dashboard,
          "history": page_history,
          "training": page_training,
          "mastishka": page_mastishka,   # <-- add this line
          "about": page_about,
      }
      ```

3. **Sanity check.**
   ```bash
   python -m py_compile app.py chatbot/__init__.py chatbot/mastishka.py chatbot/knowledge_base.py
   streamlit run app.py
   ```
   Open the app, click "🧠 Mastishka AI" in the sidebar, try a quick-question
   button and a typed question (e.g. "what is glioma", "how do I upload a
   scan", "explain my result" after running a scan).

## Notes / things to verify after integration

- No new entries are needed in `requirements.txt` — the chatbot only uses
  the Python standard library (`difflib`, `re`, `random`) plus `streamlit`,
  which is already a dependency.
- `mastishka.get_response(message, scan_result=None, threshold=0.34)` is the
  only public function you need to call. `scan_result` should be the dict
  returned by `backend.analyze_uploaded_scan` (already stored at
  `st.session_state["last_scan_result"]` per step 2c above).
- The knowledge base lives entirely in `chatbot/knowledge_base.py` as a list
  of `{"id", "patterns", "responses"}` dicts (`INTENTS`) plus a
  `SUGGESTED_QUESTIONS` list for the quick-reply buttons. Extend it by
  appending more intents — no other code changes are required.
- If you later want smarter, open-ended answers instead of a fixed FAQ,
  swap the body of `get_response()` in `chatbot/mastishka.py` for a call to
  the Anthropic API (or another LLM), keeping the same function signature
  so `app.py` doesn't need to change.
