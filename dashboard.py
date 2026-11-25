"""
Smart Grid Load Intelligence — Interactive Dashboard
Run from the project root:
    streamlit run dashboard.py
"""

from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"

st.set_page_config(
    page_title="Smart Grid Load Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Keep the dashboard visually consistent and easy to read.
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
      div[data-testid="stMetric"] {
        background: rgba(120, 120, 120, 0.08);
        border: 1px solid rgba(120, 120, 120, 0.18);
        padding: 14px 16px;
        border-radius: 12px;
      }
      .subtitle {color: #7b8490; margin-top: -0.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚡ Smart Grid Load Intelligence")
st.markdown(
    '<p class="subtitle">Electricity consumption analysis · Spark SQL · Kafka streaming · PySpark ML forecasting</p>',
    unsafe_allow_html=True,
)


def load_csv(path: Path):
    """Read a CSV if it exists; show a useful message instead of crashing."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return df if not df.empty else None
    except Exception as exc:
        st.warning(f"Could not read {path.name}: {exc}")
        return None


def first_existing(paths):
    for path in paths:
        df = load_csv(path)
        if df is not None:
            return df, path
    return None, None


def find_col(df, candidates):
    """Find a column by case-insensitive exact match, then partial match."""
    if df is None:
        return None
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    for candidate in candidates:
        for col in df.columns:
            if candidate.lower() in str(col).lower():
                return col
    return None


def as_numeric(df, col):
    if df is None or col is None:
        return None
    return pd.to_numeric(df[col], errors="coerce")


def section_missing(label, expected):
    st.info(f"**{label} data not found yet.** Expected file: `{expected}`. Run the corresponding project script first.")


# Main source files produced by the existing project.
sample, sample_path = first_existing([
    DATA_DIR / "smart_grid_sample.csv",
    DATA_DIR / "smart_grid_features.csv",
])
sql_meter, sql_meter_path = first_existing([
    OUTPUT_DIR / "spark_meter_summary.csv",
])
sql_hour, sql_hour_path = first_existing([
    OUTPUT_DIR / "spark_hourly_summary.csv",
])
kafka_all, kafka_all_path = first_existing([
    OUTPUT_DIR / "kafka_meter_summary_all.csv",
])
kafka_totals, kafka_totals_path = first_existing([
    OUTPUT_DIR / "kafka_meter_totals.csv",
])
predictions, predictions_path = first_existing([
    OUTPUT_DIR / "spark_ml_predictions.csv",
    OUTPUT_DIR / "prediction_results.csv",
])

# Sidebar filters
st.sidebar.header("Dashboard controls")
st.sidebar.caption("Filters update the charts on this page.")

meter_col = find_col(sample, ["meter"])
available_meters = sorted(sample[meter_col].dropna().astype(str).unique()) if sample is not None and meter_col else []
selected_meters = st.sidebar.multiselect(
    "Meters",
    options=available_meters,
    default=available_meters,
    help="Choose one or more meters. This filter applies to charts that contain meter-level data.",
)
if not selected_meters:
    selected_meters = available_meters

# Parse timestamps once for the sample dataset.
time_col = find_col(sample, ["x_Timestamp", "timestamp", "datetime", "date_time", "date"])
if sample is not None and time_col:
    sample[time_col] = pd.to_datetime(sample[time_col], errors="coerce")

filtered_sample = sample.copy() if sample is not None else None
if filtered_sample is not None and meter_col:
    filtered_sample = filtered_sample[filtered_sample[meter_col].astype(str).isin(selected_meters)]

# KPI row
st.subheader("Project at a glance")
if filtered_sample is not None:
    consumption_col = find_col(filtered_sample, ["t_kWh", "consumption_kWh", "load_kWh", "consumption", "load"])
    voltage_col = find_col(filtered_sample, ["z_Avg Voltage (Volt)", "voltage_V", "voltage"])
    unique_meters = filtered_sample[meter_col].nunique() if meter_col else 0
    total_rows = len(filtered_sample)
    avg_consumption = as_numeric(filtered_sample, consumption_col).mean() if consumption_col else float("nan")
    max_consumption = as_numeric(filtered_sample, consumption_col).max() if consumption_col else float("nan")
    avg_voltage = as_numeric(filtered_sample, voltage_col).mean() if voltage_col else float("nan")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Readings", f"{total_rows:,}")
    c2.metric("Meters selected", f"{unique_meters:,}")
    c3.metric("Mean consumption", f"{avg_consumption:.5f} kWh" if pd.notna(avg_consumption) else "N/A")
    c4.metric("Peak reading", f"{max_consumption:.5f} kWh" if pd.notna(max_consumption) else "N/A")
    c5.metric("Mean voltage", f"{avg_voltage:.2f} V" if pd.notna(avg_voltage) else "N/A")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Readings", "Data not found")
    c2.metric("Meters", "—")
    c3.metric("Forecast model", "See ML tab")

st.divider()

overview_tab, consumption_tab, streaming_tab, forecast_tab, data_tab = st.tabs([
    "📊 Overview",
    "⚡ Consumption analysis",
    "📡 Kafka streaming",
    "🤖 ML forecasting",
    "🗂️ Data explorer",
])

with overview_tab:
    left, right = st.columns(2)
    with left:
        st.markdown("#### Consumption by meter")
        if filtered_sample is not None and meter_col:
            consumption_col = find_col(filtered_sample, ["t_kWh", "consumption_kWh", "load_kWh", "consumption", "load"])
            if consumption_col:
                temp = filtered_sample.copy()
                temp[consumption_col] = pd.to_numeric(temp[consumption_col], errors="coerce")
                chart_df = temp.groupby(meter_col, as_index=False)[consumption_col].mean()
                chart_df = chart_df.sort_values(consumption_col, ascending=False)
                fig = px.bar(
                    chart_df, x=meter_col, y=consumption_col,
                    title="Average consumption per meter",
                    labels={meter_col: "Meter", consumption_col: "Average consumption (kWh)"},
                    hover_data={consumption_col: ":.5f"},
                )
                fig.update_layout(xaxis_tickangle=-45, height=420)
                st.plotly_chart(fig, use_container_width=True)
            else:
                section_missing("Consumption column", "data/smart_grid_sample.csv")
        else:
            section_missing("Sample readings", "data/smart_grid_sample.csv")

    with right:
        st.markdown("#### Consumption by hour")
        if filtered_sample is not None:
            consumption_col = find_col(filtered_sample, ["t_kWh", "consumption_kWh", "load_kWh", "consumption", "load"])
            if time_col and consumption_col:
                temp = filtered_sample.copy()
                temp["Dashboard hour"] = temp[time_col].dt.hour
                temp[consumption_col] = pd.to_numeric(temp[consumption_col], errors="coerce")
                hourly = temp.groupby("Dashboard hour", as_index=False)[consumption_col].mean()
                fig = px.line(
                    hourly, x="Dashboard hour", y=consumption_col,
                    markers=True, title="Average reading by hour of day",
                    labels={"Dashboard hour": "Hour of day", consumption_col: "Average consumption (kWh)"},
                )
                fig.update_layout(height=420, xaxis=dict(dtick=1))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("A timestamp and consumption column are needed for this chart.")
        else:
            section_missing("Sample readings", "data/smart_grid_sample.csv")

    st.markdown("#### Project pipeline")
    pipeline = pd.DataFrame({
        "Stage": ["Dataset preparation", "Spark SQL", "Kafka producer", "Spark Structured Streaming", "PySpark ML"],
        "Purpose": [
            "Prepare a 10,000-reading sample",
            "Summarize meter and hourly consumption",
            "Publish readings as JSON messages",
            "Aggregate each micro-batch by meter",
            "Predict electricity consumption",
        ],
        "Output": [
            "data/smart_grid_sample.csv",
            "output/spark_meter_summary.csv",
            "Kafka topic: smart-meter-readings",
            "output/kafka_meter_summary_all.csv",
            "output/spark_ml_predictions.csv",
        ],
    })
    st.dataframe(pipeline, use_container_width=True, hide_index=True)

with consumption_tab:
    st.markdown("### Spark SQL results")
    if sql_meter is not None:
        mcol = find_col(sql_meter, ["meter"])
        avgcol = find_col(sql_meter, ["avg_consumption_kWh", "average_consumption_kWh", "avg_consumption", "mean_consumption"])
        totalcol = find_col(sql_meter, ["total_consumption_kWh", "total_consumption", "sum_consumption"])
        peakcol = find_col(sql_meter, ["peak_consumption_kWh", "max_consumption_kWh", "peak_consumption", "max_consumption"])
        show_df = sql_meter.copy()
        if mcol:
            show_df = show_df[show_df[mcol].astype(str).isin(selected_meters)]
        a, b = st.columns(2)
        with a:
            if mcol and avgcol:
                fig = px.bar(show_df.sort_values(avgcol, ascending=False), x=mcol, y=avgcol,
                             title="Average consumption by meter",
                             labels={mcol: "Meter", avgcol: "Average consumption (kWh)"})
                fig.update_layout(xaxis_tickangle=-45, height=420)
                st.plotly_chart(fig, use_container_width=True)
        with b:
            if mcol and totalcol:
                fig = px.bar(show_df.sort_values(totalcol, ascending=False), x=mcol, y=totalcol,
                             title="Total consumption by meter",
                             labels={mcol: "Meter", totalcol: "Total consumption (kWh)"})
                fig.update_layout(xaxis_tickangle=-45, height=420)
                st.plotly_chart(fig, use_container_width=True)
        st.dataframe(show_df, use_container_width=True, hide_index=True)
    else:
        section_missing("Spark SQL meter summary", "output/spark_meter_summary.csv")

    st.markdown("### Hourly pattern")
    if sql_hour is not None:
        hourcol = find_col(sql_hour, ["hour", "hour_of_day"])
        valcol = find_col(sql_hour, ["avg_consumption_kWh", "average_consumption_kWh", "avg_consumption", "total_consumption_kWh", "total_consumption"])
        if hourcol and valcol:
            fig = px.bar(sql_hour, x=hourcol, y=valcol, title="Spark SQL hourly summary",
                         labels={hourcol: "Hour", valcol: valcol.replace("_", " ")})
            fig.update_layout(height=400, xaxis=dict(dtick=1))
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(sql_hour, use_container_width=True, hide_index=True)
    else:
        section_missing("Spark SQL hourly summary", "output/spark_hourly_summary.csv")

with streaming_tab:
    st.markdown("### Kafka + Spark Structured Streaming")
    if kafka_all is not None:
        kmeter = find_col(kafka_all, ["meter"])
        kbatch = find_col(kafka_all, ["batch_id"])
        kcount = find_col(kafka_all, ["reading_count"])
        kavg = find_col(kafka_all, ["average_consumption_kWh", "avg_consumption_kWh"])
        ktotal = find_col(kafka_all, ["total_consumption_kWh"])
        kpeak = find_col(kafka_all, ["peak_consumption_kWh"])
        show_k = kafka_all.copy()
        if kmeter:
            show_k = show_k[show_k[kmeter].astype(str).isin(selected_meters)]

        total_readings = pd.to_numeric(show_k[kcount], errors="coerce").sum() if kcount else float("nan")
        distinct_meters = show_k[kmeter].nunique() if kmeter else 0
        distinct_batches = show_k[kbatch].nunique() if kbatch else 0
        a, b, c = st.columns(3)
        a.metric("Readings summarized", f"{int(total_readings):,}" if pd.notna(total_readings) else "N/A")
        b.metric("Meters summarized", f"{distinct_meters:,}")
        c.metric("Micro-batches", f"{distinct_batches:,}")

        lcol, rcol = st.columns(2)
        with lcol:
            if kmeter and ktotal:
                meter_totals = show_k.groupby(kmeter, as_index=False)[ktotal].sum().sort_values(ktotal, ascending=False)
                fig = px.bar(meter_totals, x=kmeter, y=ktotal, title="Summed consumption from streaming batches",
                             labels={kmeter: "Meter", ktotal: "Consumption (kWh)"})
                fig.update_layout(xaxis_tickangle=-45, height=420)
                st.plotly_chart(fig, use_container_width=True)
        with rcol:
            if kbatch and ktotal:
                batch_totals = show_k.groupby(kbatch, as_index=False)[ktotal].sum().sort_values(kbatch)
                fig = px.line(batch_totals, x=kbatch, y=ktotal, markers=True, title="Consumption by micro-batch",
                              labels={kbatch: "Batch ID", ktotal: "Consumption (kWh)"})
                fig.update_layout(height=420)
                st.plotly_chart(fig, use_container_width=True)

        if kmeter and kavg:
            fig = px.box(show_k, x=kmeter, y=kavg, points="all", title="Average consumption across recorded batches",
                         labels={kmeter: "Meter", kavg: "Average consumption (kWh)"})
            fig.update_layout(xaxis_tickangle=-45, height=420)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("#### Streaming batch records")
        st.dataframe(show_k, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download filtered streaming results",
            data=show_k.to_csv(index=False).encode("utf-8"),
            file_name="filtered_kafka_streaming_results.csv",
            mime="text/csv",
        )
    else:
        section_missing("Consolidated Kafka results", "output/kafka_meter_summary_all.csv")
        st.caption("Run the streaming job and then run src/consolidate_results.py.")

with forecast_tab:
    st.markdown("### PySpark ML forecasting")
    if predictions is not None:
        p_meter = find_col(predictions, ["meter"])
        p_time = find_col(predictions, ["timestamp", "x_Timestamp", "datetime", "date_time"])
        actual_col = find_col(predictions, ["actual_load", "actual_consumption", "actual_kWh", "actual", "label", "t_kWh", "consumption_kWh"])
        predicted_col = find_col(predictions, ["predicted_load", "predicted_consumption", "prediction_kWh", "prediction", "predicted"])
        p_df = predictions.copy()
        if p_meter:
            p_df = p_df[p_df[p_meter].astype(str).isin(selected_meters)]
        if actual_col:
            p_df[actual_col] = pd.to_numeric(p_df[actual_col], errors="coerce")
        if predicted_col:
            p_df[predicted_col] = pd.to_numeric(p_df[predicted_col], errors="coerce")
        if actual_col and predicted_col:
            valid = p_df[[actual_col, predicted_col]].dropna()
            if not valid.empty:
                mae = (valid[actual_col] - valid[predicted_col]).abs().mean()
                rmse = (((valid[actual_col] - valid[predicted_col]) ** 2).mean()) ** 0.5
                c1, c2, c3 = st.columns(3)
                c1.metric("MAE (filtered data)", f"{mae:.6f}")
                c2.metric("RMSE (filtered data)", f"{rmse:.6f}")
                c3.metric("Prediction rows", f"{len(valid):,}")

                left, right = st.columns(2)
                with left:
                    fig = px.scatter(valid, x=actual_col, y=predicted_col,
                                     title="Actual vs predicted consumption",
                                     labels={actual_col: "Actual (kWh)", predicted_col: "Predicted (kWh)"})
                    lo = min(valid[actual_col].min(), valid[predicted_col].min())
                    hi = max(valid[actual_col].max(), valid[predicted_col].max())
                    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction"))
                    fig.update_layout(height=430)
                    st.plotly_chart(fig, use_container_width=True)
                with right:
                    error_df = valid.copy()
                    error_df["Absolute error"] = (error_df[actual_col] - error_df[predicted_col]).abs()
                    fig = px.histogram(error_df, x="Absolute error", nbins=35,
                                       title="Distribution of absolute prediction errors")
                    fig.update_layout(height=430)
                    st.plotly_chart(fig, use_container_width=True)

                if p_time:
                    p_df[p_time] = pd.to_datetime(p_df[p_time], errors="coerce")
                    timeline = p_df.dropna(subset=[p_time, actual_col, predicted_col]).sort_values(p_time)
                    if not timeline.empty:
                        # Limit the line chart to a manageable number of points.
                        if len(timeline) > 1500:
                            timeline = timeline.iloc[::max(1, len(timeline) // 1500)]
                        long_df = timeline[[p_time, actual_col, predicted_col]].melt(
                            id_vars=p_time, var_name="Series", value_name="Consumption (kWh)"
                        )
                        fig = px.line(long_df, x=p_time, y="Consumption (kWh)", color="Series",
                                      title="Actual and predicted consumption over time")
                        fig.update_layout(height=430)
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Prediction file exists, but actual/predicted columns contain no usable numeric rows.")
        else:
            st.warning(f"Could not identify actual and predicted columns. Available columns: {', '.join(map(str, p_df.columns))}")
        st.dataframe(p_df.head(500), use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download filtered predictions",
            data=p_df.to_csv(index=False).encode("utf-8"),
            file_name="filtered_spark_ml_predictions.csv",
            mime="text/csv",
        )
    else:
        section_missing("PySpark ML predictions", "output/spark_ml_predictions.csv")

with data_tab:
    st.markdown("### Explore the source readings")
    if filtered_sample is not None:
        st.caption(f"Loaded from `{sample_path.relative_to(ROOT)}`")
        st.dataframe(filtered_sample, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download filtered source readings",
            data=filtered_sample.to_csv(index=False).encode("utf-8"),
            file_name="filtered_smart_grid_readings.csv",
            mime="text/csv",
        )
    else:
        section_missing("Sample dataset", "data/smart_grid_sample.csv")

    with st.expander("Files detected by the dashboard"):
        for label, path in [
            ("Sample data", sample_path),
            ("Spark SQL meter summary", sql_meter_path),
            ("Spark SQL hourly summary", sql_hour_path),
            ("Kafka consolidated summary", kafka_all_path),
            ("Kafka meter totals", kafka_totals_path),
            ("ML predictions", predictions_path),
        ]:
            st.write(f"**{label}:** `{path.relative_to(ROOT) if path else 'Not found'}`")

st.caption("Note: charts summarize the files currently available in the project folder. They do not query a live Kafka topic. Rerun the data pipeline to refresh the saved results.")
