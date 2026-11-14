"""Pipeline console: DVC stages, MLflow runs, live service stats, links to the tools."""
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from api_client import ApiError, friendly, ops

st.set_page_config(page_title="Pipeline console", page_icon="🛠️", layout="wide")
st.title("🛠️ Pipeline console")
st.caption("Read from the API's /ops endpoints. Counters reset when the API restarts.")
st.button("Refresh")


def fetch(path):
    try:
        return ops(path)
    except ApiError as err:
        st.error(friendly(err))
        return None


st.header("Data and training pipeline (DVC)")
pipe = fetch("/ops/pipeline")
if pipe:
    dot = ["digraph { rankdir=LR; node [shape=box, style=filled, fontname=Helvetica];"]
    for s in pipe["stages"]:
        colour = "#b7e4c7" if s["status"] == "has_run" else "#dddddd"
        dot.append(f'"{s["name"]}" [fillcolor="{colour}"];')
    dot += [f'"{a}" -> "{b}";' for a, b in pipe["edges"]]
    st.graphviz_chart("\n".join(dot) + "}")
    st.caption("Green = has run, grey = never run.")
    st.dataframe(pd.DataFrame([{
        "Stage": s["name"], "Status": s["status"], "Last run": s["last_run"],
        "Seconds": s["summary"].get("seconds"),
        "Games/s": s["summary"].get("games_per_sec"),
        "Positions/s": s["summary"].get("positions_per_sec"),
        "Failed games": s["summary"].get("failed_games_last_run"),
        "Val MAE": s["summary"].get("val_mae"), "Test MAE": s["summary"].get("test_mae"),
    } for s in pipe["stages"]]), hide_index=True, use_container_width=True)

st.header("Experiment runs (MLflow)")
runs = fetch("/ops/runs")
if runs:
    counts = runs["counts"]
    cols = st.columns(max(len(counts), 1))
    for col, (name, n) in zip(cols, counts.items() or [("No runs", 0)]):
        col.metric(name.capitalize(), n)
    st.dataframe(pd.DataFrame([{
        "Run": r["run_id"][:8], "Experiment": r["experiment"], "Status": r["status"],
        "Started": pd.to_datetime(r["start_time"], unit="ms") if r["start_time"] else None,
        "Val MAE": r["metrics"].get("best_val_mae"), "Test MAE": r["metrics"].get("test_mae"),
        "Git commit": (r["git_commit"] or "")[:8],
    } for r in runs["runs"]]), hide_index=True, use_container_width=True)

st.header("Live service")
svc = fetch("/ops/service")
if svc:
    a, b, c, d = st.columns(4)
    a.metric("Requests", svc["requests_total"])
    b.metric("Predictions", svc["predictions"], f"avg {svc['predict_avg_seconds']} s"
             if svc["predict_avg_seconds"] else None, delta_color="off")
    c.metric("Errors", sum(svc["errors_by_type"].values()))
    d.metric("Engine failures", svc["stockfish"]["failures"])
    st.caption(f"Engine ready: {svc['engine_ready']} · Model ready: {svc['model_ready']} · "
               f"Idle engines: {svc['stockfish']['engines_available']} · "
               f"Up for {svc['uptime_seconds']} s")
    if svc["errors_by_type"]:
        st.bar_chart(pd.Series(svc["errors_by_type"], name="errors"))

st.header("Tool dashboards")
links = fetch("/ops/links")
if links:
    for name, url in links.items():
        st.markdown(f"- **{name.capitalize()}**: [{url}]({url})")
    pick = st.selectbox("Show one inside this page", ["(none)"] + list(links))
    if pick != "(none)":
        st.caption("If the frame stays blank, that tool blocks embedding. Use the link above.")
        components.iframe(links[pick], height=650, scrolling=True)