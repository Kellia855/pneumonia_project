"""Streamlit dashboard for the pneumonia X-ray classifier.

Every action goes through the FastAPI service (API_URL), so the dashboard shows exactly what the deployed
model does. Run locally:

    API_URL=http://localhost:8000 streamlit run app/ui.py
"""
import base64
import datetime
import html
import os
import random
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "reports" / "figures"
ICON_DIR = Path(__file__).resolve().parent / "assets" / "icons"
TEST_DIR = Path(os.getenv("DATA_DIR", ROOT / "data")) / "test"
CLASSES = ["NORMAL", "PNEUMONIA"]
COLORS = {"NORMAL": "#4C78A8", "PNEUMONIA": "#E45756"}
IMG_EXT = {".jpg", ".jpeg", ".png"}

_page_icon = ICON_DIR / "pneumonia.png"
st.set_page_config(page_title="Pneumonia X-ray Classifier",
                   page_icon=str(_page_icon) if _page_icon.exists() else None, layout="wide")


# ----------------------------------------------------------------------------- helpers
def api(method, path, timeout=30, **kwargs):
    """Call the API. Returns (json or None, error message or None, latency in ms)."""
    start = time.perf_counter()
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
        ms = (time.perf_counter() - start) * 1000
        if r.ok:
            return r.json(), None, ms
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        return None, f"{r.status_code}: {detail}", ms
    except requests.RequestException as exc:
        return None, f"API unreachable at {API_URL} ({type(exc).__name__})", (time.perf_counter() - start) * 1000


def fmt_duration(seconds):
    seconds = int(seconds or 0)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{days}d {hours}h {minutes}m" if days else f"{hours}h {minutes}m {seconds}s"


def pct(value):
    return "n/a" if value is None else f"{value:.1%}"


@st.cache_data(show_spinner=False)
def icon_uri(name):
    """An icon from app/assets/icons/ as an inline data URI (None if the file is missing)."""
    path = ICON_DIR / f"{name}.png"
    if not path.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def _img(name, height_rem):
    uri = icon_uri(name)
    return f'<img src="{uri}" alt="" style="height:{height_rem}rem;width:auto;flex-shrink:0;margin-right:0.6rem">' if uri else ""


def icon_heading(text, icon, level=2):
    """A heading with one of our icons in front of it."""
    size = {1: 2.4, 2: 2.0, 3: 1.6}[level]
    st.markdown(f'<h{level} style="display:flex;align-items:center;gap:0.6rem">{_img(icon, size)}'
                f'{html.escape(text)}</h{level}>', unsafe_allow_html=True)


NOTE_COLORS = {"info": ("rgba(28,131,225,0.10)", "#1C83E1"), "warning": ("rgba(255,189,69,0.18)", "#C48A00")}


def icon_note(text, icon, kind="info"):
    """A message box like st.info / st.warning, but with one of our icons."""
    bg, border = NOTE_COLORS[kind]
    st.markdown(f'<div style="display:flex;align-items:center;gap:0.8rem;background:{bg};'
                f'border-left:4px solid {border};border-radius:0.5rem;padding:0.75rem 1rem;margin:0 0 1rem 0">'
                f'{_img(icon, 2.0)}<span>{html.escape(text)}</span></div>', unsafe_allow_html=True)


def tab_icons(names):
    """Put an icon in front of each tab label (Streamlit tab labels cannot hold images directly)."""
    rules = []
    for i, name in enumerate(names, start=1):
        uri = icon_uri(name)
        if uri:
            rules.append(f'.stTabs [role="tab"]:nth-child({i} of [role="tab"]) p::before {{'
                         f'content:"";display:inline-block;width:1.4rem;height:1.4rem;margin-right:0.5rem;'
                         f'vertical-align:-0.35rem;background:url({uri}) center / contain no-repeat;}}')
    st.markdown("<style>" + "".join(rules) + "</style>", unsafe_allow_html=True)


@st.cache_data(ttl=20, show_spinner=False)
def get_insights():
    return api("GET", "/insights", timeout=60)


def show_figure(name, caption=None):
    path = FIG_DIR / name
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.caption(f"(figure {name} not found in reports/figures)")


def confusion_figure(m, title):
    z = [[m["TN"], m["FP"]], [m["FN"], m["TP"]]]
    fig = go.Figure(go.Heatmap(z=z, x=["Predicted NORMAL", "Predicted PNEUMONIA"],
                               y=["True NORMAL", "True PNEUMONIA"], colorscale="Blues", showscale=False,
                               text=z, texttemplate="%{text}", textfont={"size": 18}))
    fig.update_layout(title=title, height=330, margin=dict(l=10, r=10, t=50, b=10), yaxis_autorange="reversed")
    return fig


def test_images():
    """Labelled sample images shipped with the app (data/test/<CLASS>/)."""
    out = []
    for label in CLASSES:
        folder = TEST_DIR / label
        if folder.exists():
            out += [(p, label) for p in sorted(folder.iterdir()) if p.suffix.lower() in IMG_EXT]
    return out


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    icon_heading("Pneumonia X-ray", "pneumonia", level=2)
    st.caption("NORMAL vs PNEUMONIA classifier (MobileNetV2 transfer learning)")
    health, health_err, _ = api("GET", "/health", timeout=10)
    if health_err:
        st.error(health_err)
    else:
        (st.success if health["status"] == "ok" else st.warning)(f"API status: {health['status']}")
        st.write(f"**Model:** {health.get('model_name')} v{health.get('model_version')}")
        st.write(f"**Uptime:** {fmt_duration(health.get('uptime_seconds'))}")
    st.caption(f"API: `{API_URL}`")
    st.caption("Educational project. Not a medical device: predictions must not replace a clinician's diagnosis.")
    st.caption("Icons: [Flaticon](https://www.flaticon.com)")

tab_icons(["up-time", "analytics", "predict", "upload", "evaluation"])
tab_status, tab_insights, tab_predict, tab_retrain, tab_eval = st.tabs(
    ["Model up-time", "Data insights", "Predict", "Upload & retrain", "Evaluate in production"])


# ----------------------------------------------------------------------------- 1. up-time
with tab_status:
    icon_heading("Model up-time and status", "up-time")
    st.caption("Checks `/health` every 10 seconds while this page is open.")

    @st.fragment(run_every=10)
    def status_panel():
        h, err, ms = api("GET", "/health", timeout=10)
        log = st.session_state.setdefault("health_log", [])
        log.append({"time": datetime.datetime.now(), "up": bool(h) and h.get("status") == "ok",
                    "latency_ms": round(ms, 1), "container": (h or {}).get("container", "-")})
        del log[:-360]                                   # keep the last hour of checks

        if err:
            st.error(f"Model is DOWN: {err}")
        else:
            c = st.columns(5)
            c[0].metric("Status", h["status"].upper())
            c[1].metric("Uptime", fmt_duration(h["uptime_seconds"]))
            c[2].metric("Model version", f"v{h.get('model_version')}")
            c[3].metric("Predictions served", h.get("predictions_served") or 0)
            c[4].metric("Health check latency", f"{ms:.0f} ms")
            st.caption(f"Container: `{h.get('container')}` · started {h.get('started_at')} · "
                       f"database: {h.get('database')} · retraining now: {h.get('retraining')}")

        df = pd.DataFrame(log)
        st.metric("Availability during this session", f"{df['up'].mean():.0%}",
                  help=f"{int(df['up'].sum())} of {len(df)} health checks succeeded")
        fig = px.scatter(df, x="time", y="latency_ms", color=df["up"].map({True: "up", False: "down"}),
                         color_discrete_map={"up": "#2CA02C", "down": "#D62728"},
                         labels={"latency_ms": "health-check latency (ms)", "color": ""})
        fig.update_traces(mode="lines+markers")
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

    status_panel()

    st.subheader("Deployed model")
    ins, err, _ = get_insights()
    if ins:
        m = ins["model"]
        st.write(f"**{m.get('model_name')}**, version {m.get('version')}, trained {m.get('trained_at')} "
                 f"on {m.get('n_train_images')} images. Decision threshold: **{m.get('threshold'):.3f}**.")
        tm = m.get("test_metrics") or {}
        if tm:
            c = st.columns(6)
            for col, key in zip(c, ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]):
                col.metric(f"Test {key}", pct(tm.get(key)) if key != "roc_auc" else f"{tm.get(key, 0):.3f}")
            st.caption("Held-out test-set metrics recorded when the model was trained (notebook, Section 10).")


# ----------------------------------------------------------------------------- 2. insights
with tab_insights:
    icon_heading("What the data tells us", "analytics")
    st.write("Four features of the chest X-ray dataset were explored in the notebook. Each one shaped a "
             "modelling decision. Charts below are from the full dataset (5,824 images after cleaning).")

    st.subheader("1. Class balance: pneumonia outnumbers normal about 3 to 1")
    a, b = st.columns([3, 2])
    with a:
        show_figure("class_distribution.png")
    with b:
        st.markdown(
            "- 4,045 PNEUMONIA (74.6%) vs 1,379 NORMAL (25.4%) in train/val/test, a **2.93 : 1** ratio, kept "
            "identical in every split by stratification.\n"
            "- A model that always says PNEUMONIA would already score about **75% accuracy** while missing every "
            "healthy patient, so accuracy alone is misleading.\n"
            "- **Decision:** class weights (NORMAL 1.97, PNEUMONIA 0.67) and evaluation on recall, precision, "
            "F1, ROC-AUC and the confusion matrix.")

    st.subheader("2. Image size: the two classes come in very different sizes")
    a, b = st.columns([3, 2])
    with a:
        show_figure("image_dimensions.png")
    with b:
        st.markdown(
            "- Median NORMAL image: **1,654 × 1,323 px**; median PNEUMONIA image: **1,160 × 768 px**.\n"
            "- The middle 50% of the two classes do not overlap in width or height.\n"
            "- Size alone could separate many images, a **dataset bias** a model could exploit instead of "
            "reading the lungs.\n"
            "- **Decision:** every image is resized to 224 × 224 in exactly the same way in training and in "
            "this API, and high scores are read with this bias in mind.")

    st.subheader("3. Brightness: same average, different spread")
    a, b = st.columns([3, 2])
    with a:
        show_figure("brightness.png")
    with b:
        st.markdown(
            "- Mean pixel intensity is almost the same for both classes (about **123**), so overall "
            "brightness cannot tell them apart.\n"
            "- PNEUMONIA images vary more (std 20.0 vs 13.4), and the average PNEUMONIA image is hazier with "
            "greyer lung fields, while NORMAL has dark, well-defined lungs.\n"
            "- **Decision:** a CNN must learn *where* the opacities are, not overall brightness; brightness and "
            "contrast augmentation stops it relying on scanner exposure.")

    st.subheader("4. Appearance: what the model has to learn")
    show_figure("sample_images_1.png", "Random NORMAL training images")
    show_figure("sample_images_2.png", "Random PNEUMONIA training images")
    st.markdown(
        "NORMAL X-rays show dark, clear lung fields and sharp rib outlines; most PNEUMONIA X-rays are hazier with "
        "patchy white areas, though some look almost normal. Monitoring leads, timestamps and framing also differ "
        "between the classes: non-medical cues the model could latch onto, which is why augmentation is used "
        "and every prediction needs clinical review.")

    st.divider()
    st.subheader("Live data in this deployment")
    ins, err, _ = get_insights()
    if err:
        st.error(err)
    else:
        rows = [{"split": s, "class": c, "images": n} for s, counts in ins["class_counts"].items()
                for c, n in counts.items()]
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(pd.DataFrame(rows), x="split", y="images", color="class", barmode="group",
                         color_discrete_map=COLORS, title="Images available to the service, per folder")
            fig.update_layout(height=360)
            st.plotly_chart(fig, width="stretch")
            st.caption("`train` is replay data reused at every retrain, `val` decides whether a retrained model is "
                       "promoted, `test` is used by the evaluation tab, `uploads` is what users added.")
        with col2:
            bright = [{"class": c, "mean intensity": v} for c, d in (ins.get("brightness") or {}).items()
                      if d for v in d["values"]]
            if bright:
                fig = px.box(pd.DataFrame(bright), x="class", y="mean intensity", color="class", points="all",
                             color_discrete_map=COLORS, title="Mean pixel intensity per class (sample of the service's images)")
                fig.update_layout(height=360, showlegend=False)
                st.plotly_chart(fig, width="stretch")

        preds = ins["predictions"]
        col1, col2 = st.columns(2)
        with col1:
            if preds["by_label"]:
                df = pd.DataFrame([{"label": k, "count": v} for k, v in preds["by_label"].items()])
                fig = px.pie(df, names="label", values="count", color="label", color_discrete_map=COLORS,
                             title=f"Predictions served ({preds['total']} total)")
                fig.update_layout(height=340)
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No predictions served yet.")
        with col2:
            if preds["recent_probabilities"]:
                fig = px.histogram(x=preds["recent_probabilities"], nbins=20, range_x=[0, 1],
                                   labels={"x": "P(pneumonia)"}, title="Confidence of the last 500 predictions")
                thr = (ins["model"] or {}).get("threshold")
                if thr:
                    fig.add_vline(x=thr, line_dash="dash", annotation_text=f"threshold {thr:.2f}")
                fig.update_layout(height=340, yaxis_title="predictions")
                st.plotly_chart(fig, width="stretch")
                st.caption("Most predictions should sit near 0 or 1. Many values near the threshold would "
                           "signal data the model is unsure about, a reason to collect labels and retrain.")


# ----------------------------------------------------------------------------- 3. predict
with tab_predict:
    icon_heading("Predict one chest X-ray", "predict")
    source = st.radio("Image source", ["Upload an image", "Pick a labelled test image"], horizontal=True)
    image_bytes, filename, true_label = None, None, None

    if source == "Upload an image":
        up = st.file_uploader("Chest X-ray (JPG or PNG)", type=["jpg", "jpeg", "png"], key="predict_file")
        if up:
            image_bytes, filename = up.getvalue(), up.name
    else:
        samples = test_images()
        if not samples:
            st.info("No labelled test images in data/test.")
        else:
            labels = ["any"] + CLASSES
            want = st.selectbox("True class", labels)
            pool = [s for s in samples if want == "any" or s[1] == want]
            dice_col, button_col = st.columns([1, 20], vertical_alignment="center", gap="small")
            if (ICON_DIR / "dice.png").exists():
                dice_col.image(str(ICON_DIR / "dice.png"), width=40)
            if button_col.button("Random image") or "sample_idx" not in st.session_state:
                st.session_state.sample_idx = random.randrange(len(samples))
            path, true_label = pool[st.session_state.sample_idx % len(pool)]
            image_bytes, filename = path.read_bytes(), path.name

    if image_bytes:
        left, right = st.columns([1, 1])
        with left:
            st.image(image_bytes, caption=filename, width="stretch")
            if true_label:
                st.caption(f"True class: **{true_label}**")
        with right:
            if st.button("Predict", type="primary", width="stretch"):
                with st.spinner("Running the model..."):
                    res, err, ms = api("POST", "/predict", timeout=60,
                                       files={"file": (filename, image_bytes, "image/jpeg")})
                if err:
                    st.error(err)
                else:
                    show = st.error if res["label"] == "PNEUMONIA" else st.success
                    show(f"### {res['label']}")
                    c = st.columns(3)
                    c[0].metric("P(pneumonia)", f"{res['probability_pneumonia']:.3f}")
                    c[1].metric("Confidence", f"{res['confidence']:.1%}")
                    c[2].metric("Response time", f"{ms:.0f} ms")
                    st.progress(min(max(res["probability_pneumonia"], 0.0), 1.0),
                                text=f"Threshold {res['threshold']:.3f}: images at or above it are called PNEUMONIA")
                    if true_label:
                        if true_label == res["label"]:
                            st.success(f"Correct: the true class is {true_label}.")
                        else:
                            st.warning(f"Incorrect: the true class is {true_label}.")
                    st.caption(f"Model version {res.get('model_version')}")


# ----------------------------------------------------------------------------- 4. upload + retrain
with tab_retrain:
    icon_heading("Upload new data and retrain", "upload")
    st.write("Upload labelled X-rays in bulk. Each file is validated, de-duplicated, converted to RGB, resized "
             "to 224 × 224, saved, and recorded in MongoDB. Retraining fine-tunes the **deployed model** on the "
             "original replay data plus all uploads, and the new model is only promoted if it is not worse on "
             "the validation set.")

    st.subheader("1. Upload labelled images")
    col1, col2 = st.columns([1, 3])
    with col1:
        label = st.selectbox("True class of ALL files in this upload", CLASSES)
    with col2:
        files = st.file_uploader("Images (select many)", type=["jpg", "jpeg", "png"],
                                 accept_multiple_files=True, key=f"bulk_{st.session_state.get('upload_round', 0)}")
    if st.button(f"Upload {len(files or [])} image(s) as {label}", disabled=not files, type="primary"):
        with st.spinner("Uploading and preprocessing..."):
            res, err, _ = api("POST", "/upload", timeout=600, data={"label": label},
                              files=[("files", (f.name, f.getvalue(), f.type or "image/jpeg")) for f in files])
        if err:
            st.error(err)
        else:
            st.session_state.last_upload = res
            st.session_state.upload_round = st.session_state.get("upload_round", 0) + 1
            get_insights.clear()
            st.rerun()

    res = st.session_state.get("last_upload")
    if res:
        pre = res["preprocessing"]
        c = st.columns(3)
        c[0].metric("Received", pre["received"])
        c[1].metric("Saved to database", pre["saved_to_database"])
        c[2].metric("Rejected", pre["rejected"])
        st.write("Preprocessing steps: " + " → ".join(pre["steps"]))
        if res["rejected_files"]:
            st.dataframe(pd.DataFrame(res["rejected_files"]), hide_index=True, width="stretch")

    stored, err, _ = api("GET", "/uploads", params={"limit": 15})
    if stored:
        s = stored["summary"]
        st.write(f"**In the database:** {s['total']} uploaded images "
                 f"({s.get('new', 0)} new, {s.get('used', 0)} already used in a promoted model) · by class: "
                 + ", ".join(f"{k} {v}" for k, v in s["by_label"].items()) if s["total"] else
                 "**In the database:** no uploads yet.")
        if stored["recent"]:
            with st.expander("Most recent uploads (MongoDB `uploads` collection)"):
                st.dataframe(pd.DataFrame(stored["recent"]), hide_index=True, width="stretch")

    st.subheader("2. Retrain")

    @st.fragment(run_every=3)
    def retrain_panel():
        status, err, _ = api("GET", "/retrain/status", timeout=10)
        if err:
            st.error(err)
            return
        trig = status["trigger"]
        if trig["retrain"]:
            icon_note(f"Retraining trigger fired: {'; '.join(trig['reasons'])}", "trigger", "warning")
        else:
            st.info(f"Trigger: {trig['n_new_images']} of {trig['threshold']} new images needed for automatic "
                    "retraining. You can still retrain manually.")

        c1, c2 = st.columns([1, 3])
        epochs = c1.number_input("Epochs", min_value=1, max_value=20, value=3)
        if c2.button("Retrain model now", type="primary", disabled=status["running"]):
            r, e, _ = api("POST", "/retrain", data={"epochs": int(epochs)})
            if e:
                st.error(e)
            else:
                st.success(f"Retraining run {r['run_id']} started ({r['reason']}).")
                time.sleep(1)
                st.rerun()

        if status["running"]:
            icon_note(f"Retraining in progress: {status['message']}", "in-progress", "info")
        elif status["message"] and status["message"] != "idle":
            st.write(f"Last result: **{status['message']}**")

        hist, e, _ = api("GET", "/retrain/history", params={"limit": 10})
        if hist:
            df = pd.DataFrame(hist)
            cols = [c for c in ["id", "started_at", "status", "trigger_reason", "n_new_images", "old_version",
                                "new_version", "promoted", "old_val_f1", "new_val_f1", "old_val_recall",
                                "new_val_recall", "message"] if c in df]
            st.write("Retraining history (validation metrics before and after):")
            st.dataframe(df[cols], hide_index=True, width="stretch")

    retrain_panel()


# ----------------------------------------------------------------------------- 5. evaluate
with tab_eval:
    icon_heading("Evaluate the deployed model in production", "evaluation")
    st.write("Scores the model that is currently serving predictions on the labelled test images shipped with "
             "the service, using the same threshold as `/predict`.")
    n = st.slider("Number of test images", 50, 1000, 300, step=50)
    if st.button("Run evaluation", type="primary"):
        with st.spinner("Evaluating..."):
            res, err, ms = api("POST", "/evaluate", params={"max_images": n}, timeout=600)
        if err:
            st.error(err)
        else:
            m = res["metrics"]
            st.write(f"Model v{res['model_version']} · threshold {res['threshold']:.3f} · "
                     f"{m['n_images']} images · {ms / 1000:.1f} s")
            c = st.columns(6)
            for col, key in zip(c, ["accuracy", "precision", "recall", "specificity", "f1"]):
                col.metric(key.capitalize(), pct(m[key]))
            c[5].metric("ROC-AUC", "n/a" if m["roc_auc"] is None else f"{m['roc_auc']:.3f}")
            a, b = st.columns(2)
            a.plotly_chart(confusion_figure(m, "Confusion matrix (production)"), width="stretch")
            b.markdown(
                f"- **Missed pneumonia cases (false negatives):** {m['FN']}\n"
                f"- **False alarms (false positives):** {m['FP']}\n\n"
                "Recall matters most for screening: a missed case is the costly error. The test images here are "
                "a balanced sample, so accuracy and precision differ from the notebook's imbalanced test set.")

    with st.expander("Model evaluation from the notebook (training time)"):
        show_figure("mobilenet_curves.png", "MobileNetV2 training curves (feature extraction, then fine-tuning)")
        a, b = st.columns(2)
        with a:
            show_figure("roc_curves.png", "ROC curves on the test set")
        with b:
            show_figure("confusion_matrices.png", "Confusion matrices at threshold 0.5")
        show_figure("retrain_comparison.png", "Test metrics before vs after retraining in the notebook")