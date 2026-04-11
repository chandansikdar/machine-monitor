"""
test_electrical_diagnostics.py
===============================
Unit test suite for electrical_diagnostics.py.

One test per worked example in the methodology document v0.6:
    \u00a73.2   V-I pairing error \u2192 integrity gate catch
    \u00a75.2   VUF crossing critical
    \u00a75.3   VUF crossing watch only
    \u00a76.7.1  Cell 1: Healthy
    \u00a76.7.2  Cell 3: Panel-to-motor zone issue
    \u00a76.7.3  Cell 2: Motor zone issue
    \u00a76.7.4  Cell 4: Both zones
    \u00a77.6.1  Zone 4 positive / load unchanged \u2192 alarm
    \u00a77.6.2  Zone 4 positive / load increased \u2192 dismissed
    \u00a77.6.3  Zone 4 negative / load decreased \u2192 dismissed
    \u00a77.6.4  HVAC chiller \u2192 suppressed
    \u00a78.2    Full assessment cycle

Run with:
    python -m pytest test_electrical_diagnostics.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from electrical_diagnostics import (
    CLEANING_MIN_SAMPLES,
    IUF_CRITICAL,
    IUF_WATCH,
    PF_DRIFT_ACTION,
    PF_DRIFT_ALERT,
    PF_DRIFT_WATCH,
    VUF_CRITICAL,
    VUF_WATCH,
    assessment_summary,
    clean_samples,
    classify_motor_side,
    classify_vuf,
    compute_iuf,
    compute_pf_drift,
    compute_vuf,
    compute_zone4,
    ingest_baseline,
    integrity_gate,
    run_assessment,
    select_pf_bands,
    validate_baseline_state,
)

# ---------------------------------------------------------------------------
# Reference machines
# ---------------------------------------------------------------------------

META_45KW = {
    "v_nominal_phase": 230.0,
    "p_rated_shaft_kw": 45.0,
    "pf_rated": 0.87,
    "eta_rated": 0.93,
    "i_rated": 80.3,
    "application_type": "compressed_air",
    "four_wire": True,
    "measurement_at_panel": True,
}

META_112KW = {
    "v_nominal_phase": 230.0,
    "p_rated_shaft_kw": 112.0,
    "pf_rated": 0.90,
    "eta_rated": 0.90,
    "i_rated": 200.0,
    "application_type": "compressed_air",
    "four_wire": True,
    "measurement_at_panel": True,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_df(
    n: int,
    phase_1_voltage=230.9, phase_2_voltage=230.9, phase_3_voltage=230.9,
    phase_1_current=80.3,  phase_2_current=80.3,  phase_3_current=80.3,
    phase_1_active_power=16130.0, phase_2_active_power=16130.0, phase_3_active_power=16130.0,
    randomise=False,
    seed=42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1e-3, (n, 9)) if randomise else np.zeros((n, 9))
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1min"),
        "phase_1_voltage": phase_1_voltage + noise[:, 0],
        "phase_2_voltage": phase_2_voltage + noise[:, 1],
        "phase_3_voltage": phase_3_voltage + noise[:, 2],
        "phase_1_current": phase_1_current + noise[:, 3],
        "phase_2_current": phase_2_current + noise[:, 4],
        "phase_3_current": phase_3_current + noise[:, 5],
        "phase_1_active_power": phase_1_active_power + noise[:, 6],
        "phase_2_active_power": phase_2_active_power + noise[:, 7],
        "phase_3_active_power": phase_3_active_power + noise[:, 8],
    })


def make_multi_band_df(meta, n_per_band=30, n_bands=5, seed=42) -> pd.DataFrame:
    """DataFrame spanning multiple P_total load bands (powers stored in Watts)."""
    p_rated_elec_kw = meta["p_rated_shaft_kw"] / meta["eta_rated"]
    bin_width_w = 0.02 * p_rated_elec_kw * 1000.0
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_bands):
        p_centre_w = p_rated_elec_kw * 0.50 * 1000.0 + k * bin_width_w
        for _ in range(n_per_band):
            p1_w = p_centre_w / 3.0 + rng.normal(0, p_centre_w * 0.001)
            i1 = meta["i_rated"] * 0.70 + rng.normal(0, 0.01)
            v1 = 230.9 + rng.normal(0, 0.01)
            rows.append({
                "timestamp": pd.Timestamp("2024-01-01"),
                "phase_1_voltage": v1, "phase_2_voltage": v1, "phase_3_voltage": v1,
                "phase_1_current": i1, "phase_2_current": i1, "phase_3_current": i1,
                "phase_1_active_power": p1_w, "phase_2_active_power": p1_w, "phase_3_active_power": p1_w,
            })
    return pd.DataFrame(rows)


# ===========================================================================
# \u00a73.2  Integrity gate: V-I pairing error
# ===========================================================================

class TestIntegrityGate:
    def test_vi_pairing_error_fails_check3(self):
        """Swapped B/C CTs produce sign-cancelling per-phase powers: Check 3 fails."""
        row = pd.Series({
            "phase_1_voltage": 230.9, "phase_2_voltage": 230.9, "phase_3_voltage": 230.9,
            "phase_1_current": 80.3,  "phase_2_current": 80.3,  "phase_3_current": 80.3,
            "phase_1_active_power": 16130.0, "phase_2_active_power": -15990.0, "phase_3_active_power": -160.0,
        })
        result = integrity_gate(row, META_45KW)
        assert not result.passed
        assert result.failing_check == "check_3_power_sign_coherence"

    def test_healthy_sample_passes(self):
        row = pd.Series({
            "phase_1_voltage": 230.9, "phase_2_voltage": 230.9, "phase_3_voltage": 230.9,
            "phase_1_current": 80.3,  "phase_2_current": 80.3,  "phase_3_current": 80.3,
            "phase_1_active_power": 16130.0, "phase_2_active_power": 16130.0, "phase_3_active_power": 16130.0,
        })
        assert integrity_gate(row, META_45KW).passed

    def test_low_voltage_fails_check1(self):
        row = pd.Series({
            "phase_1_voltage": 40.0, "phase_2_voltage": 230.9, "phase_3_voltage": 230.9,
            "phase_1_current": 80.3, "phase_2_current": 80.3, "phase_3_current": 80.3,
            "phase_1_active_power": 16130.0, "phase_2_active_power": 16130.0, "phase_3_active_power": 16130.0,
        })
        result = integrity_gate(row, META_45KW)
        assert not result.passed
        assert result.failing_check == "check_1_voltage_plausibility"

    def test_low_pf_fails_check4(self):
        row = pd.Series({
            "phase_1_voltage": 230.9, "phase_2_voltage": 230.9, "phase_3_voltage": 230.9,
            "phase_1_current": 80.3,  "phase_2_current": 80.3,  "phase_3_current": 80.3,
            "phase_1_active_power": 100.0, "phase_2_active_power": 16130.0, "phase_3_active_power": 16130.0,  # PF_a << 0.30
        })
        result = integrity_gate(row, META_45KW)
        assert not result.passed
        assert result.failing_check == "check_4_pf_plausibility"


# ===========================================================================
# \u00a75.2 / \u00a75.3  Zone 1: supply channel
# ===========================================================================

class TestZone1:
    def test_vuf_critical(self):
        """VUF = 5/229 * 100 = 2.18% \u2192 critical (\u00a75.2)."""
        df = make_df(100, phase_1_voltage=234.0, phase_2_voltage=228.0, phase_3_voltage=225.0)
        vuf = compute_vuf(df)
        assert abs(vuf - 5.0 / 229.0 * 100.0) < 0.05
        assert classify_vuf(vuf).tier == "critical"

    def test_vuf_watch_only(self):
        """VUF ~1.01% \u2192 watch, not critical (\u00a75.3)."""
        df = make_df(100, phase_1_voltage=232.0, phase_2_voltage=229.0, phase_3_voltage=228.0)
        vuf = compute_vuf(df)
        assert VUF_WATCH < vuf <= VUF_CRITICAL
        assert classify_vuf(vuf).tier == "watch"

    def test_balanced_no_alarm(self):
        df = make_df(100)
        assert classify_vuf(compute_vuf(df)).tier is None


# ===========================================================================
# \u00a76.7  Four-cell decision matrix
# ===========================================================================

class TestDecisionMatrix:
    def test_cell1_healthy(self):
        """IUF 0.5%, PF drift +0.001 \u2192 Cell 1 (\u00a76.7.1)."""
        r = classify_motor_side(0.5, None, None, 0.001, False, None, [])
        assert r.cell == 1 and r.alarm_tier is None

    def test_cell3_panel_to_motor_watch(self):
        """IUF 5.5%, PF drift -0.005 \u2192 Cell 3 watch (\u00a76.7.2)."""
        r = classify_motor_side(5.5, "1", "low", -0.005, False, None, [])
        assert r.cell == 3
        assert r.alarm_tier == "watch"
        assert r.outlier_phase == "1"

    def test_cell2_motor_zone_watch(self):
        """IUF 0.6%, PF drift -0.018 \u2192 Cell 2 watch (\u00a76.7.3)."""
        r = classify_motor_side(0.6, None, None, -0.018, False, None, [])
        assert r.cell == 2
        assert r.pf_drift_tier == "watch"

    def test_cell4_both_zones_alert(self):
        """IUF 6.2%, PF drift -0.022 \u2192 Cell 4 alert (\u00a76.7.4)."""
        r = classify_motor_side(6.2, "A", "low", -0.022, False, None, [])
        assert r.cell == 4
        assert r.alarm_tier == "alert"   # IUF=watch, PF=alert \u2192 max=alert
        refs = " ".join(r.inspection_refs)
        assert "\u00a76.8.1" in refs and "\u00a76.8.2" in refs and "\u00a76.8.3" in refs

    def test_iuf_critical_tier(self):
        """IUF >= 10% \u2192 critical tier."""
        r = classify_motor_side(10.5, "B", "high", -0.005, False, None, [])
        assert r.iuf_tier == "critical"

    def test_pf_action_tier(self):
        """PF drift <= -0.03 \u2192 action tier."""
        r = classify_motor_side(0.4, None, None, -0.035, False, None, [])
        assert r.pf_drift_tier == "action"


# ===========================================================================
# \u00a77.6  Zone 4 worked examples
# ===========================================================================

class TestZone4:
    META = {**META_45KW, "application_type": "compressed_air"}
    META_HVAC = {**META_45KW, "application_type": "hvac_chiller"}
    P_BASE = 75.0

    def _recent(self, p_kw: float, n=200) -> pd.DataFrame:
        p = p_kw / 3.0
        return make_df(n, phase_1_active_power=p * 1000,
                       phase_2_active_power=p * 1000,
                       phase_3_active_power=p * 1000)

    def test_positive_finding(self):
        """78 kW vs 75 kW (+4%) \u2192 positive finding."""
        r = compute_zone4(self.P_BASE, self._recent(78.0), self.META)
        assert r.finding_state == "positive"
        assert r.delta_p_pct > 0

    def test_negative_finding(self):
        """67 kW vs 75 kW (-10.7%) \u2192 negative finding."""
        r = compute_zone4(self.P_BASE, self._recent(67.0), self.META)
        assert r.finding_state == "negative"
        assert r.delta_p_pct < 0

    def test_hvac_suppressed(self):
        """hvac_chiller \u2192 detection suppressed."""
        r = compute_zone4(self.P_BASE, self._recent(80.0), self.META_HVAC)
        assert r.suppressed
        assert r.finding_state == "suppressed"

    def test_below_threshold_no_finding(self):
        """1.3% change \u2192 no finding (within \u00b13% threshold)."""
        r = compute_zone4(self.P_BASE, self._recent(76.0), self.META)
        assert r.finding_state is None

    def test_result_has_message(self):
        """All results produce a non-empty message."""
        for p_kw in (67.0, 75.5, 78.0):
            r = compute_zone4(self.P_BASE, self._recent(p_kw), self.META)
            assert isinstance(r.message, str) and len(r.message) > 10

    def test_suppressed_has_message(self):
        r = compute_zone4(self.P_BASE, self._recent(80.0), self.META_HVAC)
        assert isinstance(r.message, str) and len(r.message) > 10


# ===========================================================================
# \u00a78.2  Full assessment cycle
# ===========================================================================

class TestFullAssessment:
    def _baseline(self):
        df = make_multi_band_df(META_112KW, n_per_band=300, n_bands=6, seed=0)
        return ingest_baseline(df, META_112KW), META_112KW

    def test_full_cycle_runs_and_produces_record(self):
        bm, meta = self._baseline()
        p_rated_elec = meta["p_rated_shaft_kw"] / meta["eta_rated"]
        # 112 kW motor at ~65% load: P_phase = 26.96 kW, I_phase = P/(V*PF) ~= 130 A
        p_phase_w = p_rated_elec / 3.0 * 0.65 * 1000
        i_phase = p_phase_w / (230.9 * meta["pf_rated"])   # ~130 A
        recent = make_df(300, randomise=True, seed=7,
                         phase_1_active_power=p_phase_w, phase_2_active_power=p_phase_w, phase_3_active_power=p_phase_w,
                         phase_1_current=i_phase,  phase_2_current=i_phase,  phase_3_current=i_phase)
        record = run_assessment(recent, bm, meta)
        assert not record.suppressed
        assert record.supply_alarm is not None
        assert record.motor_side is not None
        assert record.zone4 is not None

    def test_suppressed_when_tiny_recent(self):
        bm, meta = self._baseline()
        tiny = make_df(5, phase_1_active_power=29000.0, phase_2_active_power=29000.0, phase_3_active_power=29000.0)
        record = run_assessment(tiny, bm, meta)
        assert record.suppressed
        assert record.cleaning_report.n_cleaned < CLEANING_MIN_SAMPLES

    def test_summary_string_returned(self):
        bm, meta = self._baseline()
        p_rated_elec = meta["p_rated_shaft_kw"] / meta["eta_rated"]
        p_phase_w = p_rated_elec / 3.0 * 0.65 * 1000
        i_phase = p_phase_w / (230.9 * meta["pf_rated"])
        recent = make_df(300, randomise=True, seed=8,
                         phase_1_active_power=p_phase_w, phase_2_active_power=p_phase_w, phase_3_active_power=p_phase_w,
                         phase_1_current=i_phase,  phase_2_current=i_phase,  phase_3_current=i_phase)
        record = run_assessment(recent, bm, meta)
        s = assessment_summary(record)
        assert isinstance(s, str) and len(s) > 20


# ===========================================================================
# Multi-band PF drift
# ===========================================================================

class TestMultiBandPF:
    def test_qualifying_bands_produced(self):
        df = make_multi_band_df(META_112KW, n_per_band=30, n_bands=5)
        p_rated = META_112KW["p_rated_shaft_kw"] / META_112KW["eta_rated"]
        bands = select_pf_bands(df, p_rated)
        assert len(bands) >= 3

    def test_zero_drift_same_data(self):
        df = make_multi_band_df(META_112KW, n_per_band=50, n_bands=5, seed=10)
        p_rated = META_112KW["p_rated_shaft_kw"] / META_112KW["eta_rated"]
        bands = select_pf_bands(df, p_rated)
        agg, _, suppressed, _ = compute_pf_drift(df, bands)
        assert not suppressed
        assert abs(agg) < 0.005

    def test_suppressed_when_few_recent_samples(self):
        df = make_multi_band_df(META_112KW, n_per_band=50, n_bands=5, seed=20)
        p_rated = META_112KW["p_rated_shaft_kw"] / META_112KW["eta_rated"]
        bands = select_pf_bands(df, p_rated)
        agg, _, suppressed, _ = compute_pf_drift(df.iloc[:1], bands)
        assert suppressed and agg is None


# ===========================================================================
# Data cleaning
# ===========================================================================

class TestDataCleaning:
    def test_healthy_data_mostly_retained(self):
        df = make_df(500, randomise=True)
        _, report = clean_samples(df, META_45KW)
        assert report.fraction_retained > 0.80

    def test_low_load_removed(self):
        df = make_df(200, phase_1_active_power=500.0, phase_2_active_power=500.0, phase_3_active_power=500.0)
        _, report = clean_samples(df, META_45KW)
        assert report.n_after_load_precondition == 0

    def test_user_filter_reduces_count(self):
        df = make_df(200, randomise=True, seed=5)
        df["shift"] = ["day" if i % 2 == 0 else "night" for i in range(200)]
        _, r_all  = clean_samples(df, META_45KW)
        _, r_day  = clean_samples(df, META_45KW, user_filter="shift == 'day'")
        assert r_day.n_cleaned < r_all.n_cleaned


# ===========================================================================
# Threshold constants
# ===========================================================================

class TestThresholds:
    def test_vuf_ordering(self):     assert 0 < VUF_WATCH < VUF_CRITICAL
    def test_iuf_ordering(self):     assert 0 < IUF_WATCH < IUF_CRITICAL
    def test_pf_drift_ordering(self): assert PF_DRIFT_ACTION < PF_DRIFT_ALERT < PF_DRIFT_WATCH < 0
