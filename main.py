"""
main.py — Machine Continuous Monitoring Analytics
Run with:  streamlit run main.py
"""

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from analyzer import Analyzer
from database import Database
from visualizer import Visualizer

load_dotenv()

# ------------------------------------------------------------------ #
# Page config
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="Machine Analytics",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="metric-container"] {
        background: rgba(128,128,128,0.05);
        border-radius: 8px;
        padding: 8px 12px;
    }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
# Helper — render a complete insights block
# (defined first so it can be called from anywhere below)
# ------------------------------------------------------------------ #

def render_insights(insights: dict, data: pd.DataFrame, viz: Visualizer):
    """Render KPIs, narrative, anomalies, key insights, and Plotly charts."""

    score = insights.get("health_score")
    if score is not None:
        colour = "green" if score >= 80 else "orange" if score >= 60 else "red"
        st.markdown(f"### Health score: :{colour}[{score} / 100]")
        st.progress(int(score) / 100)

    kpis = insights.get("kpis", [])
    if kpis:
        status_icon = {"normal": "✅", "warning": "⚠️", "critical": "🔴"}
        cols = st.columns(min(len(kpis), 5))
        for i, kpi in enumerate(kpis[:5]):
            icon = status_icon.get(kpi.get("status", "normal"), "")
            cols[i].metric(
                kpi.get("label", "—"),
                f"{icon} {kpi.get('value', '—')}",
            )

    if insights.get("narrative"):
        st.info(insights["narrative"])

    anomalies = insights.get("anomalies", [])
    if anomalies:
        st.subheader("⚠️ Anomalies detected")
        for a in anomalies:
            st.warning(f"**{a.get('parameter', '—')}** — {a.get('description', '')}")

    key_points = insights.get("insights", [])
    if key_points:
        st.subheader("Key insights")
        for point in key_points:
            st.markdown(f"- {point}")

    recs = insights.get("chart_recommendations", [])
    if recs and data is not None:
        st.subheader("Charts")
        for fig in viz.generate_charts(data, recs):
            st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------ #
# Session state
# ------------------------------------------------------------------ #

if "last_insights" not in st.session_state:
    st.session_state["last_insights"] = None
if "last_data" not in st.session_state:
    st.session_state["last_data"] = None


# ------------------------------------------------------------------ #
# Services (cached so they survive reruns)
# ------------------------------------------------------------------ #

@st.cache_resource
def get_services():
    return Database(), Visualizer()

db, viz = get_services()


# ================================================================== #
# SIDEBAR
# ================================================================== #

with st.sidebar:
    st.title("⚙️ Machine Analytics")
    st.caption("Continuous monitoring  ·  Powered by Claude")
    st.divider()

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            help="Paste your key here, or add ANTHROPIC_API_KEY to a .env file.",
        )
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key

    st.divider()

    with st.expander("➕ Register new machine", expanded=not db.get_machines()):
        machine_type = st.text_input(
            "Machine type", placeholder="e.g. Compressor, Pump, Furnace"
        )
        machine_id = st.text_input("Machine ID", placeholder="e.g. COMP-001")
        machine_desc = st.text_area(
            "Specs / notes",
            placeholder="Rated power: 75 kW\nFlow: 500 m³/h\nRPM: 1480",
            height=90,
        )
        if st.button("Register", type="primary", use_container_width=True):
            if machine_type and machine_id:
                db.register_machine(machine_id.strip(), machine_type.strip(), machine_desc)
                st.success(f"Machine **{machine_id}** registered.")
                st.rerun()
            else:
                st.error("Machine type and ID are required.")

    st.divider()

    machines = db.get_machines()
    if not machines:
        st.info("Register a machine above to get started.")
        st.stop()

    machine_labels = {
        m["machine_id"]: f"{m['machine_id']}  ({m['machine_type']})"
        for m in machines
    }
    selected_id = st.selectbox(
        "Active machine",
        options=list(machine_labels.keys()),
        format_func=lambda x: machine_labels[x],
    )

    st.divider()

    st.subheader("Upload data")
    uploaded_file = st.file_uploader(
        "CSV or Excel",
        type=["csv", "xlsx", "xls"],
        help="Any column named timestamp/date/time is auto-detected as the time axis.",
    )
    if uploaded_file:
        if st.button("Ingest", use_container_width=True):
            with st.spinner("Reading and storing data…"):
                result = db.ingest_file(uploaded_file, selected_id)
            if result["success"]:
                st.success(f"✓ {result['rows']:,} rows ingested")
                st.caption("Columns: " + ", ".join(result["columns"]))
                st.rerun()
            else:
                st.error(result["error"])

    file_info = db.get_file_info(selected_id)
    if file_info:
        st.caption(f"{len(file_info)} file(s) stored for this machine")


# ================================================================== #
# MAIN AREA
# ================================================================== #

machine_info = db.get_machine_info(selected_id)
st.title(f"{machine_info['machine_type']}  ·  {selected_id}")
if machine_info.get("description"):
    st.caption(machine_info["description"])

data = db.get_data(selected_id)

tab_data, tab_analysis, tab_history = st.tabs(["📊 Data", "🔍 Analysis", "📋 History"])


# ------------------------------------------------------------------ #
# TAB 1 — Data preview
# ------------------------------------------------------------------ #

with tab_data:
    if data is None or data.empty:
        st.info("No data uploaded yet. Use the sidebar to ingest a CSV or Excel file.")
    else:
        numeric_cols = data.select_dtypes(include="number").columns.tolist()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows",       f"{len(data):,}")
        c2.metric("Parameters", len(numeric_cols))
        c3.metric("From",       str(data.index.min().date()))
        c4.metric("To",         str(data.index.max().date()))

        st.subheader("Recent readings")
        st.dataframe(data.tail(200), use_container_width=True, height=280)

        st.subheader("Descriptive statistics")
        st.dataframe(data[numeric_cols].describe().round(3), use_container_width=True)

        if numeric_cols:
            st.subheader("Sensor overview")
            figs = viz.generate_charts(
                data,
                [{"type": "time_series",
                  "title": "All parameters",
                  "parameters": numeric_cols[:8]}],
            )
            st.plotly_chart(figs[0], use_container_width=True)


# ------------------------------------------------------------------ #
# TAB 2 — Analysis
# ------------------------------------------------------------------ #

with tab_analysis:
    if data is None or data.empty:
        st.info("Upload data first (sidebar), then run an analysis.")
    else:
        left, right = st.columns([1, 3])

        with left:
            analysis_type = st.selectbox(
                "Analysis type",
                [
                    "Overall Health Assessment",
                    "Trend & Drift Analysis",
                    "Anomaly Detection",
                    "Correlation Analysis",
                    "Parameter Distribution",
                    "Cross-Parameter Comparison",
                ],
            )

            min_d, max_d = data.index.min().date(), data.index.max().date()
            if min_d < max_d:
                date_range = st.date_input(
                    "Date range",
                    value=[min_d, max_d],
                    min_value=min_d,
                    max_value=max_d,
                )
                if not isinstance(date_range, (list, tuple)) or len(date_range) < 2:
                    date_range = (min_d, max_d)
            else:
                date_range = (min_d, max_d)

            extra_context = st.text_area(
                "Engineer notes (optional)",
                placeholder="e.g. Bearing replaced on 2024-03-01.\nUnusual vibration noticed last week.",
                height=100,
            )

            has_key = bool(os.getenv("ANTHROPIC_API_KEY"))
            analyze_clicked = st.button(
                "🔍  Analyze",
                type="primary",
                use_container_width=True,
                disabled=not has_key,
            )
            if not has_key:
                st.caption("⚠️ Add your API key in the sidebar first.")

        with right:
            if analyze_clicked:
                with st.spinner("Claude is analysing your data…"):
                    analyzer = Analyzer()
                    result = analyzer.analyze(
                        machine_info=machine_info,
                        data=data,
                        analysis_type=analysis_type,
                        date_range=date_range,
                        extra_context=extra_context,
                    )

                if result["success"]:
                    db.save_analysis(selected_id, analysis_type, result["insights"])
                    st.session_state["last_insights"] = result["insights"]
                    st.session_state["last_data"]     = data
                    st.rerun()
                else:
                    st.error(f"Analysis failed: {result['error']}")

            if st.session_state["last_insights"] is not None:
                render_insights(
                    st.session_state["last_insights"],
                    st.session_state["last_data"],
                    viz,
                )
            else:
                st.markdown(
                    "_Select an analysis type and press **Analyze** to generate insights._"
                )


# ------------------------------------------------------------------ #
# TAB 3 — History
# ------------------------------------------------------------------ #

with tab_history:
    history = db.get_analysis_history(selected_id)
    if not history:
        st.info("No analysis runs yet for this machine.")
    else:
        for record in history:
            label = f"{record['analysis_type']}  —  {record['timestamp']}"
            with st.expander(label):
                ins = record["insights"]
                score = ins.get("health_score")
                if score is not None:
                    colour = "green" if score >= 80 else "orange" if score >= 60 else "red"
                    st.markdown(f"**Health score:** :{colour}[{score} / 100]")
                if ins.get("narrative"):
                    st.info(ins["narrative"])
                for point in ins.get("insights", []):
                    st.markdown(f"- {point}")
