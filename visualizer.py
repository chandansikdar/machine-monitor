"""
visualizer.py — Plotly chart renderer

Supported chart types:
  - time_series  : one or more parameters over time
  - histogram    : distribution of one or more parameters
  - correlation  : heatmap of parameter correlations
  - scatter      : two parameters against each other

Plus dedicated builders:
  - control_chart : Shewhart chart with UCL/UWL/Mean/LWL/LCL lines
                    and highlighted anomaly zones
"""

from typing import List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=20, t=50, b=40),
    hovermode="x unified",
    font=dict(size=12),
)

# Control chart line colours
_CC = dict(
    ucl="#C0392B",   # red   — 3σ upper
    uwl="#E67E22",   # amber — 2σ upper
    mean="#2C3E50",  # dark  — mean
    lwl="#E67E22",   # amber — 2σ lower
    lcl="#C0392B",   # red   — 3σ lower
    data="#185FA5",  # blue  — data line
    zone1="rgba(192,57,43,0.07)",   # 3σ zone fill
    zone2="rgba(230,126,34,0.07)",  # 2σ zone fill
)


class Visualizer:

    # ------------------------------------------------------------------ #
    # Public — Claude recommendations
    # ------------------------------------------------------------------ #

    def generate_charts(self, data: pd.DataFrame, recommendations: List[dict]) -> List[go.Figure]:
        numeric_cols = data.select_dtypes(include="number").columns.tolist()
        charts = []
        for rec in recommendations:
            chart_type = rec.get("type", "time_series")
            params = [p for p in rec.get("parameters", []) if p in data.columns]
            if not params:
                params = numeric_cols[:4]
            title = rec.get("title", "Chart")
            note  = rec.get("note", "")
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
                        text=f"Note: {note}", xref="paper", yref="paper",
                        x=0, y=-0.12, showarrow=False,
                        font=dict(size=11, color="gray"), align="left",
                    )
                charts.append(fig)
            except Exception:
                continue
        if not charts and numeric_cols:
            charts.append(self._time_series(data, numeric_cols[:6], "Sensor Overview"))
        return charts

    # ------------------------------------------------------------------ #
    # Public — control charts (called directly from main.py)
    # ------------------------------------------------------------------ #

    def generate_control_charts(
        self,
        data: pd.DataFrame,
        thresholds: Optional[dict] = None,
        max_cols: int = 6,
    ) -> List[go.Figure]:
        """
        Generate one Shewhart control chart per numeric column (up to max_cols).
        Each chart shows:
          - Data line
          - UCL  (mean + 3σ)  red solid
          - UWL  (mean + 2σ)  amber dashed
          - Mean              dark solid
          - LWL  (mean − 2σ)  amber dashed
          - LCL  (mean − 3σ)  red solid
          - Shaded zones between limits
          - Engineering thresholds (if defined) as purple lines
          - Anomaly markers where |Z| > 3
        """
        numeric_cols = data.select_dtypes(include="number").columns.tolist()[:max_cols]
        figs = []
        for col in numeric_cols:
            try:
                fig = self._control_chart(data, col, thresholds or {})
                figs.append(fig)
            except Exception:
                continue
        return figs

    # ------------------------------------------------------------------ #
    # Control chart builder
    # ------------------------------------------------------------------ #

    def _control_chart(
        self,
        data: pd.DataFrame,
        col: str,
        thresholds: dict,
    ) -> go.Figure:
        s    = data[col].dropna()
        mean = float(s.mean())
        std  = float(s.std())
        if std == 0:
            std = 1e-9

        ucl = mean + 3 * std
        uwl = mean + 2 * std
        lwl = mean - 2 * std
        lcl = mean - 3 * std

        # Z-scores for anomaly detection
        z     = (s - mean) / std
        anom  = s[z.abs() > 3]

        fig = go.Figure()

        # ── Shaded zones ─────────────────────────────────────────────
        # Between UCL and UWL (upper 2σ–3σ zone)
        fig.add_hrect(y0=uwl, y1=ucl, fillcolor=_CC["zone1"],
                      line_width=0, layer="below")
        # Between LCL and LWL (lower 2σ–3σ zone)
        fig.add_hrect(y0=lcl, y1=lwl, fillcolor=_CC["zone1"],
                      line_width=0, layer="below")
        # Between UWL and LWL (inner ±2σ zone — subtle)
        fig.add_hrect(y0=lwl, y1=uwl, fillcolor=_CC["zone2"],
                      line_width=0, layer="below")

        # ── Limit lines ───────────────────────────────────────────────
        x0, x1 = data.index.min(), data.index.max()

        def hline(y, color, dash, name, width=1.2):
            fig.add_shape(type="line", x0=x0, x1=x1, y0=y, y1=y,
                          line=dict(color=color, width=width, dash=dash))
            # Invisible trace for legend entry
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="lines",
                line=dict(color=color, width=width, dash=dash),
                name=name, showlegend=True,
            ))

        hline(ucl,  _CC["ucl"],  "solid",  f"UCL  {ucl:.2f}  (mean+3σ)", 1.5)
        hline(uwl,  _CC["uwl"],  "dash",   f"UWL  {uwl:.2f}  (mean+2σ)", 1.0)
        hline(mean, _CC["mean"], "solid",  f"Mean {mean:.2f}", 1.5)
        hline(lwl,  _CC["lwl"],  "dash",   f"LWL  {lwl:.2f}  (mean-2σ)", 1.0)
        hline(lcl,  _CC["lcl"],  "solid",  f"LCL  {lcl:.2f}  (mean-3σ)", 1.5)

        # ── Engineering thresholds (if defined) ───────────────────────
        if col in thresholds:
            lims = thresholds[col]
            warn = lims.get("warning")
            crit = lims.get("critical")
            if warn is not None:
                fig.add_shape(type="line", x0=x0, x1=x1, y0=warn, y1=warn,
                              line=dict(color="#8E44AD", width=1.5, dash="dot"))
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="lines",
                    line=dict(color="#8E44AD", width=1.5, dash="dot"),
                    name=f"Warning threshold  {warn}", showlegend=True,
                ))
            if crit is not None:
                fig.add_shape(type="line", x0=x0, x1=x1, y0=crit, y1=crit,
                              line=dict(color="#641E16", width=1.5, dash="dot"))
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="lines",
                    line=dict(color="#641E16", width=1.5, dash="dot"),
                    name=f"Critical threshold  {crit}", showlegend=True,
                ))

        # ── Data line ─────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=data.index, y=data[col],
            mode="lines",
            line=dict(color=_CC["data"], width=1.2),
            name=col,
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>" + col + ": %{y:.3f}<extra></extra>",
        ))

        # ── Anomaly markers (|Z| > 3) ─────────────────────────────────
        if len(anom):
            fig.add_trace(go.Scatter(
                x=anom.index, y=anom.values,
                mode="markers",
                marker=dict(color=_CC["ucl"], size=7, symbol="circle-open",
                            line=dict(width=2, color=_CC["ucl"])),
                name=f"Anomalies (|Z|>3)  n={len(anom)}",
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Value: %{y:.3f}<br>Anomaly<extra></extra>",
            ))

        # ── Layout ────────────────────────────────────────────────────
        col_label = col.replace("_", " ").title()
        fig.update_layout(
            title=dict(text=f"Control chart — {col_label}", font=dict(size=14)),
            xaxis_title="Time",
            yaxis_title=col_label,
            legend=dict(
                orientation="h", yanchor="top", y=-0.22,
                xanchor="left", x=0,
                bgcolor="rgba(0,0,0,0)", borderwidth=0,
            ),
            **_LAYOUT,
            margin=dict(l=40, r=20, t=55, b=110),
        )
        return fig

    # ------------------------------------------------------------------ #
    # Standard chart builders
    # ------------------------------------------------------------------ #

    def _time_series(self, data, params, title):
        fig = go.Figure()
        for p in params:
            if p in data.columns:
                fig.add_trace(go.Scatter(x=data.index, y=data[p], name=p,
                                         mode="lines", line=dict(width=1.5)))
        fig.update_layout(title=title, xaxis_title="Time", **_LAYOUT)
        return fig

    def _histogram(self, data, params, title):
        fig = go.Figure()
        for p in params:
            if p in data.columns:
                fig.add_trace(go.Histogram(x=data[p], name=p, opacity=0.75, nbinsx=40))
        fig.update_layout(title=title, barmode="overlay",
                          xaxis_title="Value", yaxis_title="Count", **_LAYOUT)
        return fig

    def _correlation_heatmap(self, data, params, title):
        corr = data[params].corr().round(2)
        fig  = px.imshow(corr, title=title, color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1, aspect="auto", text_auto=True)
        fig.update_layout(**_LAYOUT)
        return fig

    def _scatter(self, data, params, title):
        if len(params) < 2:
            return self._time_series(data, params, title)
        fig = px.scatter(data, x=params[0], y=params[1], title=title,
                         opacity=0.6, trendline="ols")
        fig.update_layout(**_LAYOUT)
        return fig
