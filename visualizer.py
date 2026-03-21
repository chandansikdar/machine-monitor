"""
visualizer.py — Plotly chart renderer

Takes Claude's chart_recommendations list and the actual DataFrame,
and returns a list of Plotly Figure objects ready for st.plotly_chart().

Supported chart types:
  - time_series  : one or more parameters over time
  - histogram    : distribution of one or more parameters
  - correlation  : heatmap of parameter correlations
  - scatter      : two parameters against each other
"""

from typing import List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Shared layout defaults — transparent background to match Streamlit theme
_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=20, t=50, b=40),
    hovermode="x unified",
    font=dict(size=12),
)


class Visualizer:
    def generate_charts(
        self, data: pd.DataFrame, recommendations: List[dict]
    ) -> List[go.Figure]:
        """
        Render each chart recommendation.  Falls back to a full time-series
        overview if Claude returned no recommendations or they all fail.
        """
        numeric_cols = data.select_dtypes(include="number").columns.tolist()
        charts = []

        for rec in recommendations:
            chart_type = rec.get("type", "time_series")
            # Only use columns that actually exist in the dataset
            params = [p for p in rec.get("parameters", []) if p in data.columns]
            if not params:
                params = numeric_cols[:4]
            title = rec.get("title", "Chart")
            note = rec.get("note", "")

            try:
                if chart_type == "time_series":
                    fig = self._time_series(data, params, title)
                elif chart_type == "histogram":
                    fig = self._histogram(data, params, title)
                elif chart_type == "correlation":
                    fig = self._correlation_heatmap(data, numeric_cols, title)
                elif chart_type == "scatter":
                    fig = self._scatter(data, params, title)
                else:
                    fig = self._time_series(data, params, title)

                if note:
                    fig.add_annotation(
                        text=f"Note: {note}",
                        xref="paper", yref="paper",
                        x=0, y=-0.12, showarrow=False,
                        font=dict(size=11, color="gray"),
                        align="left",
                    )
                charts.append(fig)
            except Exception:
                continue  # Skip charts that fail, don't crash the whole dashboard

        # Fallback — always show at least something
        if not charts and numeric_cols:
            charts.append(
                self._time_series(data, numeric_cols[:6], "Sensor Overview")
            )

        return charts

    # ------------------------------------------------------------------ #
    # Chart builders
    # ------------------------------------------------------------------ #

    def _time_series(self, data: pd.DataFrame, params: List[str], title: str) -> go.Figure:
        fig = go.Figure()
        for param in params:
            if param in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[param],
                    name=param,
                    mode="lines",
                    line=dict(width=1.5),
                ))
        fig.update_layout(title=title, xaxis_title="Time", **_LAYOUT)
        return fig

    def _histogram(self, data: pd.DataFrame, params: List[str], title: str) -> go.Figure:
        fig = go.Figure()
        for param in params:
            if param in data.columns:
                fig.add_trace(go.Histogram(
                    x=data[param],
                    name=param,
                    opacity=0.75,
                    nbinsx=40,
                ))
        fig.update_layout(
            title=title,
            barmode="overlay",
            xaxis_title="Value",
            yaxis_title="Count",
            **_LAYOUT,
        )
        return fig

    def _correlation_heatmap(
        self, data: pd.DataFrame, params: List[str], title: str
    ) -> go.Figure:
        corr = data[params].corr().round(2)
        fig = px.imshow(
            corr,
            title=title,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            aspect="auto",
            text_auto=True,
        )
        fig.update_layout(**_LAYOUT)
        return fig

    def _scatter(self, data: pd.DataFrame, params: List[str], title: str) -> go.Figure:
        if len(params) < 2:
            return self._time_series(data, params, title)
        x_col, y_col = params[0], params[1]
        fig = px.scatter(
            data,
            x=x_col,
            y=y_col,
            title=title,
            opacity=0.6,
            trendline="ols",
        )
        fig.update_layout(**_LAYOUT)
        return fig
