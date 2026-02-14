#!/usr/bin/env python3
"""
================================================================================
UPS Battery String — Physics-Informed Anomaly Detection via Multivariate Calculus
================================================================================

Author:  Senior Principal Engineer, Datacenter Critical Environments
Purpose: Detect thermal runaway precursors in UPS battery strings using
         Jacobian (impedance) and Hessian (thermal acceleration) analysis.

KEY INSIGHT FOR OPERATIONS TEAMS
─────────────────────────────────
Traditional monitoring fires an alarm when temperature crosses a static
threshold (e.g., 40 °C).  By then you may have only seconds before thermal
runaway propagates to adjacent cells.

This system uses two calculus-derived "leading indicators":

    1. JACOBIAN  |dV/dI|  ≈  Internal Resistance (R_int)
       Rises when a cell develops high impedance (sulfation, dry joints).
       Detectable BEFORE voltage sag becomes visible.

    2. HESSIAN   d²T/dt²  =  Thermal Acceleration
       Captures whether the RATE of heating is itself increasing.
       In a healthy discharge d²T/dt² ≈ 0 (linear temp rise).
       During thermal runaway onset d²T/dt² >> 0 (exponential).
       This spikes BEFORE the temperature crosses the alarm limit.

    The combination of these two metrics, plus a Taylor-series voltage
    forecast, provides multi-minute lead time for controlled load transfer.

Physics Model (Thevenin + Arrhenius):
    V(t) = OCV  -  I(t) · R_int(t)
    T(t) = T_ambient + Joule(I²·R) + Arrhenius(ΔR)
================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import uniform_filter1d
import json, warnings
warnings.filterwarnings("ignore")


# ═════════════════════════════════════════════════════════════════════════════
# 1.  PARAMETERS
# ═════════════════════════════════════════════════════════════════════════════

N_POINTS        = 500
DT              = 1.0           # seconds per sample
ANOMALY_ONSET   = 300           # index where fault begins

# Battery string (48V nominal, 100Ah VRLA)
OCV             = 54.0          # open-circuit voltage (V)
R_INT_HEALTHY   = 0.012         # healthy internal resistance (Ω)
I_NOMINAL       = 42.0          # nominal load (A)
T_AMBIENT       = 25.0          # ambient (°C)

# Sensor noise
I_NOISE  = 0.8
V_NOISE  = 0.03
T_NOISE  = 0.08

# Signal processing
SMOOTH_W = 15                   # moving-average window for pre-smoothing
REGRESS_W = 21                  # rolling regression window for derivatives

# Detection thresholds (calibrated below in code)
JACOBIAN_THRESHOLD  = None      # auto-calibrated from healthy baseline
HESSIAN_THRESHOLD   = None      # auto-calibrated from healthy baseline
VOLTAGE_ALARM_LOW   = 46.0      # static voltage alarm (V)
TEMP_ALARM_HIGH     = 40.0      # static temperature alarm (°C)

# Taylor prediction horizon
PREDICT_HORIZON = 30.0          # seconds


# ═════════════════════════════════════════════════════════════════════════════
# 2.  TELEMETRY GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def generate_telemetry():
    """
    Generate 500 samples of synthetic UPS battery telemetry.

    Normal (0–299):  Stable load cycling, slow aging drift.
    Anomaly (300–499): Non-linear resistance growth → voltage sag → thermal runaway.
    """
    np.random.seed(42)
    t = np.arange(N_POINTS, dtype=float) * DT
    I = np.zeros(N_POINTS)
    V = np.zeros(N_POINTS)
    T = np.zeros(N_POINTS)
    R = np.zeros(N_POINTS)
    labels = np.zeros(N_POINTS, dtype=int)

    for i in range(N_POINTS):
        # Current: load cycling + sensor noise
        I[i] = I_NOMINAL + 5.0*np.sin(2*np.pi*t[i]/120) + np.random.normal(0, I_NOISE)

        # Internal resistance
        if i < ANOMALY_ONSET:
            R[i] = R_INT_HEALTHY * (1.0 + 1e-4 * i)   # tiny aging drift
        else:
            df = float(i - ANOMALY_ONSET)
            R[i] = R_INT_HEALTHY + 2e-4*df + 4e-6*df**2 + 3e-8*df**3
            labels[i] = 1

        # Voltage: Thevenin
        V[i] = OCV - I[i]*R[i] + np.random.normal(0, V_NOISE)

        # Temperature: base + Arrhenius
        base = T_AMBIENT + 0.008 * min(i, ANOMALY_ONSET)
        if i >= ANOMALY_ONSET:
            dR = R[i] - R_INT_HEALTHY
            gain = min(3.0*(np.exp(6.0*dR) - 1.0), 50.0)
            T[i] = base + gain + np.random.normal(0, T_NOISE)
        else:
            T[i] = base + np.random.normal(0, T_NOISE)

    return dict(time=t, current=I, voltage=V, temperature=T, resistance=R, labels=labels)


# ═════════════════════════════════════════════════════════════════════════════
# 3.  ROBUST DERIVATIVE COMPUTATION via Rolling OLS Regression
# ═════════════════════════════════════════════════════════════════════════════

def smooth(x, w=SMOOTH_W):
    """Moving-average pre-filter to suppress sensor noise."""
    return uniform_filter1d(x.astype(float), size=w, mode="nearest")


def rolling_derivatives(y, dt=DT, window=REGRESS_W):
    """
    Compute 1st and 2nd derivatives using rolling quadratic regression.

    WHY NOT FINITE DIFFERENCES?
        np.gradient amplifies noise by O(1/dt).  For second derivatives
        the amplification is O(1/dt²).  With 0.08 °C sensor noise and
        dt = 1s, the raw d²T/dt² is dominated by noise.

        Instead, we fit a local quadratic  y = a + b·t + c·t²  in a
        sliding window.  Then:
            dy/dt  = b   (1st derivative at window center)
            d²y/dt² = 2c  (2nd derivative at window center)

        This is equivalent to a Savitzky-Golay filter and provides
        numerically stable derivatives even on noisy sensor data.
    """
    n = len(y)
    dy  = np.zeros(n)
    d2y = np.zeros(n)
    half = window // 2

    for i in range(half, n - half):
        # Local time variable centered at zero
        t_local = np.arange(-half, half + 1, dtype=float) * dt
        y_local = y[i - half : i + half + 1]

        # Fit quadratic: y = a + b*t + c*t²
        # Using numpy polyfit (degree 2): coefficients [c, b, a]
        coeffs = np.polyfit(t_local, y_local, 2)
        d2y[i] = 2.0 * coeffs[0]   # d²y/dt² = 2c
        dy[i]  = coeffs[1]          # dy/dt = b (at center)

    # Pad boundaries
    dy[:half] = dy[half]
    dy[n-half:] = dy[n-half-1]
    d2y[:half] = d2y[half]
    d2y[n-half:] = d2y[n-half-1]

    return dy, d2y


# ═════════════════════════════════════════════════════════════════════════════
# 4.  JACOBIAN:  dV/dI  (Impedance)
# ═════════════════════════════════════════════════════════════════════════════

def compute_jacobian(voltage, current):
    """
    Compute dV/dI — the Jacobian of Voltage w.r.t. Current.

    PHYSICS:
        From Thevenin:  V = OCV - I·R_int
        Therefore:      dV/dI = -R_int
        So:            |dV/dI| is a non-invasive impedance measurement.

    METHOD:
        1. Pre-smooth V and I to suppress sensor noise.
        2. Compute dV/dt and dI/dt using rolling regression (stable).
        3. Apply chain rule:  dV/dI = (dV/dt) / (dI/dt).
        4. Smooth the result to suppress residual noise.
    """
    v_s = smooth(voltage)
    i_s = smooth(current)

    dV_dt, _ = rolling_derivatives(v_s)
    dI_dt, _ = rolling_derivatives(i_s)

    # Safe division: clamp |dI/dt| away from zero
    dI_safe = np.where(np.abs(dI_dt) < 0.05, np.sign(dI_dt + 1e-10) * 0.05, dI_dt)
    jacobian = dV_dt / dI_safe

    return smooth(np.abs(jacobian), w=11)


# ═════════════════════════════════════════════════════════════════════════════
# 5.  HESSIAN:  d²T/dt²  (Thermal Acceleration)
# ═════════════════════════════════════════════════════════════════════════════

def compute_hessian(temperature):
    """
    Compute d²T/dt² — the second time-derivative of Temperature.

    WHY d²T/dt² IS THE "LEADING INDICATOR":
    ────────────────────────────────────────
        • dT/dt (first derivative): "How fast is temperature rising?"
          During healthy discharge, dT/dt is small and roughly constant.

        • d²T/dt² (second derivative / Hessian): "Is the rate of
          temperature rise ITSELF accelerating?"

          Healthy:  d²T/dt² ≈ 0   (steady-state Joule heating)
          Fault:    d²T/dt² > 0   (heat generation outpaces dissipation)
          Runaway:  d²T/dt² >> 0  (exponential thermal gain)

        The Hessian spikes when the Arrhenius term "turns on" — typically
        when R_int has increased enough for I²·R losses to overwhelm the
        cell's thermal mass.  This happens BEFORE the temperature itself
        reaches the alarm threshold.

    NOTE ON TERMINOLOGY:
        Strictly, d²T/dt² is the acceleration, not the full Hessian matrix.
        In the multivariate case the Hessian would be the matrix of all
        second partial derivatives.  Here we use "Hessian" in the
        engineering sense: the curvature of the thermal response function,
        which is the diagonal element d²T/dt² that matters most for
        anomaly detection.
    """
    t_s = smooth(temperature)
    dT_dt, d2T_dt2 = rolling_derivatives(t_s)
    return smooth(d2T_dt2, w=9), smooth(dT_dt, w=9)


# ═════════════════════════════════════════════════════════════════════════════
# 6.  TAYLOR SERIES:  Voltage Forecast
# ═════════════════════════════════════════════════════════════════════════════

def taylor_predict(voltage, horizon=PREDICT_HORIZON):
    """
    V̂(t+Δt) = V(t) + V'(t)·Δt + ½·V''(t)·Δt²

    WHY:
        This is fundamentally how BMS systems compute "Remaining Run Time."
        The 2nd-order term captures whether the discharge slope is steepening
        (cell degradation) or flattening (load reduction).

        During a high-impedance event V'' < 0, causing the forecast to breach
        the low-voltage alarm before the actual voltage does.
    """
    v_s = smooth(voltage)
    dV, d2V = rolling_derivatives(v_s)

    v_o1 = voltage + dV * horizon
    v_o2 = voltage + dV * horizon + 0.5 * d2V * horizon**2

    return v_o2, v_o1, dV, d2V


# ═════════════════════════════════════════════════════════════════════════════
# 7.  THRESHOLD CALIBRATION
# ═════════════════════════════════════════════════════════════════════════════

def calibrate_thresholds(jacobian, hessian, healthy_end=ANOMALY_ONSET - 10):
    """
    Auto-calibrate detection thresholds from the healthy baseline.

    Uses the 95th percentile of each metric in the known-healthy region,
    multiplied by a safety factor, as the alarm threshold.

    WHY AUTO-CALIBRATE:
        Every battery string has different baseline impedance and thermal
        characteristics.  Hard-coded thresholds cause either nuisance alarms
        (too sensitive) or missed faults (too lax).  Calibrating from the
        first 5 minutes of "known good" data adapts to the specific string.
    """
    start = REGRESS_W  # skip regression boundary artifacts

    j_healthy = jacobian[start:healthy_end]
    h_healthy = np.abs(hessian[start:healthy_end])

    j_thresh = np.percentile(j_healthy, 95) * 3.0   # 3× safety factor
    h_thresh = np.percentile(h_healthy, 95) * 4.0   # 4× safety factor (more noise)

    # Enforce minimum thresholds
    j_thresh = max(j_thresh, 0.02)
    h_thresh = max(h_thresh, 0.0005)

    return j_thresh, h_thresh


# ═════════════════════════════════════════════════════════════════════════════
# 8.  ANOMALY DETECTION
# ═════════════════════════════════════════════════════════════════════════════

def detect_anomalies(jacobian, hessian, v_predicted, j_thresh, h_thresh):
    """
    Multi-criteria anomaly detection.

    ALERT LEVELS:
        1 — IMPEDANCE WARNING:   |dV/dI| > threshold
        2 — THERMAL INSTABILITY: d²T/dt² > threshold
        2 — VOLTAGE FORECAST:    V̂(t+30s) < low-voltage alarm
        3 — COMPOSITE (≥2 flags simultaneously) → EMERGENCY

    WHY MULTI-CRITERIA:
        Any single metric can produce false positives (load transients
        spike the Jacobian, HVAC cycling can cause Hessian blips).
        Requiring 2+ concurrent flags dramatically reduces nuisance alarms.
    """
    n = len(jacobian)
    alerts = dict(
        impedance_warning       = np.zeros(n, dtype=bool),
        thermal_instability     = np.zeros(n, dtype=bool),
        voltage_forecast_breach = np.zeros(n, dtype=bool),
        composite_critical      = np.zeros(n, dtype=bool),
        severity                = np.zeros(n, dtype=int),
    )

    SKIP = REGRESS_W + 5  # skip boundary artifacts

    for i in range(SKIP, n):
        if jacobian[i] > j_thresh:
            alerts["impedance_warning"][i] = True
            alerts["severity"][i] = max(alerts["severity"][i], 1)

        if np.abs(hessian[i]) > h_thresh:
            alerts["thermal_instability"][i] = True
            alerts["severity"][i] = max(alerts["severity"][i], 2)

        if v_predicted[i] < VOLTAGE_ALARM_LOW:
            alerts["voltage_forecast_breach"][i] = True
            alerts["severity"][i] = max(alerts["severity"][i], 2)

        flags = sum([alerts["impedance_warning"][i],
                     alerts["thermal_instability"][i],
                     alerts["voltage_forecast_breach"][i]])
        if flags >= 2:
            alerts["composite_critical"][i] = True
            alerts["severity"][i] = 3

    return alerts


# ═════════════════════════════════════════════════════════════════════════════
# 9.  DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

def create_dashboard(data, jacobian, hessian, dT_dt, v_pred, v_o1, alerts,
                     j_thresh, h_thresh):
    t = data["time"]; n = len(t)

    # Industrial SCADA palette
    BG="#0a0e17"; PNL="#0f1520"; GRD="#1a2235"; TXT="#c8d6e5"
    CYN="#00d2ff"; LIM="#39ff14"; AMB="#ffbe0b"; RED="#ff006e"; GRN="#06d6a0"

    fig = plt.figure(figsize=(22, 16), facecolor=BG)
    fig.suptitle("UPS BATTERY STRING — PHYSICS-INFORMED ANOMALY DETECTION",
                 fontsize=18, fontweight="bold", color=TXT, y=0.98, fontfamily="monospace")
    fig.text(0.5, 0.955,
             "Jacobian (Impedance)  ·  Hessian (Thermal Acceleration)  ·  Taylor Series (Voltage Forecast)",
             ha="center", fontsize=10, color="#5c7080", fontfamily="monospace")
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25,
                  left=0.06, right=0.97, top=0.93, bottom=0.05)

    def sty(ax, title, ylabel):
        ax.set_facecolor(PNL)
        ax.set_title(title, fontsize=11, fontweight="bold", color=TXT,
                     fontfamily="monospace", pad=10)
        ax.set_ylabel(ylabel, fontsize=9, color="#8899aa", fontfamily="monospace")
        ax.tick_params(colors="#5c7080", labelsize=8)
        ax.grid(True, color=GRD, lw=0.5, alpha=0.6)
        for s in ax.spines.values(): s.set_color("#1a2235")
        ax.axvspan(ANOMALY_ONSET*DT, n*DT, alpha=0.06, color=RED)
        ax.axvline(ANOMALY_ONSET*DT, color=RED, lw=1, ls="--", alpha=0.4)
        return ax

    # ── 1. Voltage & Current ────────────────────────────────────────────
    ax = sty(fig.add_subplot(gs[0,0]), "VOLTAGE & CURRENT", "Voltage (V)")
    ax.plot(t, data["voltage"], color=CYN, lw=0.8, alpha=0.9, label="Voltage")
    ax.axhline(VOLTAGE_ALARM_LOW, color=AMB, lw=1, ls=":", alpha=0.7,
               label=f"Low-V Alarm ({VOLTAGE_ALARM_LOW} V)")
    ax.legend(loc="lower left", fontsize=7, facecolor=PNL, edgecolor="#2a3a50", labelcolor=TXT)
    ax2 = ax.twinx()
    ax2.plot(t, data["current"], color=GRN, lw=0.5, alpha=0.4, label="Current")
    ax2.set_ylabel("Current (A)", fontsize=9, color="#8899aa", fontfamily="monospace")
    ax2.tick_params(colors="#5c7080", labelsize=8)
    ax2.legend(loc="upper right", fontsize=7, facecolor=PNL, edgecolor="#2a3a50", labelcolor=TXT)

    # ── 2. Temperature ──────────────────────────────────────────────────
    ax = sty(fig.add_subplot(gs[0,1]), "TEMPERATURE", "Temperature (°C)")
    ax.plot(t, data["temperature"], color="#ff6b6b", lw=1.0, alpha=0.9)
    ax.axhline(TEMP_ALARM_HIGH, color=RED, lw=1.5, ls=":", alpha=0.8,
               label=f"Temp Alarm ({TEMP_ALARM_HIGH} °C)")
    ax.fill_between(t, data["temperature"], TEMP_ALARM_HIGH,
                    where=data["temperature"]>TEMP_ALARM_HIGH, color=RED, alpha=0.2)
    ax.legend(loc="upper left", fontsize=7, facecolor=PNL, edgecolor="#2a3a50", labelcolor=TXT)

    # ── 3. Jacobian ─────────────────────────────────────────────────────
    ax = sty(fig.add_subplot(gs[1,0]),
             "JACOBIAN |dV/dI| — Impedance Map", "|dV/dI| (Ω)")
    ax.plot(t, jacobian, color=CYN, lw=0.9, alpha=0.9)
    ax.axhline(j_thresh, color=AMB, lw=1.5, ls="--", alpha=0.8,
               label=f"Threshold ({j_thresh:.3f} Ω)")
    ax.fill_between(t, jacobian, j_thresh,
                    where=jacobian > j_thresh, color=AMB, alpha=0.2)
    ax.set_ylim(0, min(np.percentile(jacobian, 99.8)*1.3, 2.0))
    ax.legend(loc="upper left", fontsize=7, facecolor=PNL, edgecolor="#2a3a50", labelcolor=TXT)
    # Annotate impedance climb
    ji = np.where((jacobian > j_thresh) & (np.arange(n) > ANOMALY_ONSET))[0]
    if len(ji) > 0:
        ax.annotate("← Impedance climb = cell degradation",
            xy=(t[ji[0]], j_thresh*1.3), fontsize=7, color=AMB, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1505", edgecolor=AMB, alpha=0.8))

    # ── 4. Hessian ──────────────────────────────────────────────────────
    ax = sty(fig.add_subplot(gs[1,1]),
             "HESSIAN d²T/dt² — Thermal Acceleration (LEADING INDICATOR)",
             "d²T/dt² (°C/s²)")
    ax.plot(t, hessian, color="#ff6b6b", lw=0.9, alpha=0.9)
    ax.axhline(h_thresh, color=RED, lw=1.5, ls="--", alpha=0.8,
               label=f"Threshold ({h_thresh:.4f})")
    ax.axhline(-h_thresh, color=RED, lw=1.0, ls="--", alpha=0.3)
    ax.axhline(0, color="#3a4a5a", lw=0.5, alpha=0.5)
    ax.fill_between(t, hessian, h_thresh,
                    where=hessian > h_thresh, color=RED, alpha=0.25)
    p1, p99 = np.percentile(hessian, [1, 99])
    margin = max(abs(p1), abs(p99)) * 1.5
    ax.set_ylim(-margin, margin)
    ax.legend(loc="upper left", fontsize=7, facecolor=PNL, edgecolor="#2a3a50", labelcolor=TXT)

    # Key callout
    hi = np.where((hessian > h_thresh) & (np.arange(n) > ANOMALY_ONSET))[0]
    ti_alarm = np.where(data["temperature"] > TEMP_ALARM_HIGH)[0]
    if len(hi) > 0 and len(ti_alarm) > 0:
        lead = (ti_alarm[0] - hi[0]) * DT
        ax.annotate(
            f"Hessian fires {lead:.0f}s\nBEFORE temp alarm\n→ This is your lead time",
            xy=(t[hi[0]], h_thresh*2),
            fontsize=7, color=RED, fontfamily="monospace", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a0a15", edgecolor=RED, alpha=0.9))

    # ── 5. Taylor Prediction ────────────────────────────────────────────
    ax = sty(fig.add_subplot(gs[2,0]),
             f"TAYLOR SERIES — {int(PREDICT_HORIZON)}s Voltage Forecast", "Voltage (V)")
    ax.plot(t, data["voltage"], color=CYN, lw=0.7, alpha=0.5, label="Actual V(t)")
    ax.plot(t, v_o1, color=LIM, lw=0.8, alpha=0.6, ls="--", label="1st Order V̂")
    ax.plot(t, v_pred, color=AMB, lw=1.0, alpha=0.9, label="2nd Order V̂")
    ax.axhline(VOLTAGE_ALARM_LOW, color=RED, lw=1.5, ls=":", alpha=0.7,
               label=f"Low-V ({VOLTAGE_ALARM_LOW} V)")
    ax.fill_between(t, v_pred, VOLTAGE_ALARM_LOW,
                    where=v_pred < VOLTAGE_ALARM_LOW, color=RED, alpha=0.15)
    vmin = min(np.min(data["voltage"]), np.min(v_pred))
    ax.set_ylim(max(vmin - 2, 30), OCV + 2)
    ax.legend(loc="lower left", fontsize=7, facecolor=PNL, edgecolor="#2a3a50", labelcolor=TXT)
    ax.set_xlabel("Time (s)", fontsize=9, color="#8899aa", fontfamily="monospace")

    # ── 6. Alert Timeline ───────────────────────────────────────────────
    ax = sty(fig.add_subplot(gs[2,1]), "COMPOSITE ALERT TIMELINE", "Severity")
    sc = {1: AMB, 2: "#ff8800", 3: RED}
    for i in range(n):
        if alerts["severity"][i] > 0:
            ax.bar(t[i], alerts["severity"][i], width=DT,
                   color=sc.get(alerts["severity"][i], RED), alpha=0.7)
    ax.set_yticks([0,1,2,3])
    ax.set_yticklabels(["OK","WARN","CRITICAL","EMERGENCY"], fontsize=8)
    ax.set_ylim(-0.3, 4.2)
    ax.set_xlabel("Time (s)", fontsize=9, color="#8899aa", fontfamily="monospace")

    # Annotate composite
    ci = np.where(alerts["composite_critical"])[0]
    if len(ci) > 0:
        fc = ci[0]
        ax.annotate(f"FIRST COMPOSITE\nt = {t[fc]:.0f}s",
            xy=(t[fc], 3), xytext=(max(t[fc]-100, 20), 3.7),
            fontsize=8, color=RED, fontweight="bold", fontfamily="monospace",
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a0a15", edgecolor=RED))
        if len(ti_alarm) > 0:
            lead = (ti_alarm[0] - fc) * DT
            ax.annotate(f"Temp alarm: t={t[ti_alarm[0]]:.0f}s\nLEAD TIME: {lead:.0f}s",
                xy=(t[ti_alarm[0]], 2.5), xytext=(min(t[ti_alarm[0]]+15, 480), 1.5),
                fontsize=7, color=AMB, fontfamily="monospace",
                arrowprops=dict(arrowstyle="->", color=AMB, lw=0.8),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1505", edgecolor=AMB))

    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 10.  REPORT
# ═════════════════════════════════════════════════════════════════════════════

def generate_report(data, jacobian, hessian, alerts, j_thresh, h_thresh):
    L = ["="*72, "  ANOMALY DETECTION REPORT — UPS Battery String", "="*72,
         f"  Samples: {N_POINTS}   Interval: {DT}s   Fault injected at: t={ANOMALY_ONSET}s",
         f"  Jacobian threshold (auto): {j_thresh:.4f} Ω",
         f"  Hessian threshold (auto):  {h_thresh:.6f} °C/s²",
         "-"*72, "  DETECTION TIMELINE:"]

    for name, key in [("Jacobian Warning", "impedance_warning"),
                      ("Hessian Alert", "thermal_instability"),
                      ("Voltage Forecast", "voltage_forecast_breach"),
                      ("Composite Critical", "composite_critical")]:
        idx = np.where(alerts[key])[0]
        if len(idx) > 0:
            L.append(f"    {name:24s} FIRST at t = {idx[0]*DT:.0f}s  (idx {idx[0]})")
        else:
            L.append(f"    {name:24s} Not triggered")

    ti = np.where(data["temperature"] > TEMP_ALARM_HIGH)[0]
    if len(ti) > 0:
        L.append(f"    {'Static Temp Alarm':24s} at t = {ti[0]*DT:.0f}s")

    L.append("-"*72)
    ci = np.where(alerts["composite_critical"])[0]
    if len(ci) > 0 and len(ti) > 0:
        lead = (ti[0] - ci[0]) * DT
        L += [f"  ★ LEAD TIME: {lead:.0f} seconds",
              f"    Composite alert fired {lead:.0f}s BEFORE temperature alarm.",
              f"    This is the window for controlled load transfer vs EPO."]
    elif len(ci) > 0:
        L.append("  ★ Composite alert fired; temperature alarm not reached in window.")
    else:
        L.append("  ★ No composite alerts triggered.")

    L += ["-"*72, "  PEAK VALUES:",
          f"    Max |dV/dI|:     {np.max(jacobian):.4f} Ω",
          f"    Max d²T/dt²:     {np.max(hessian):.6f} °C/s²",
          f"    Max Temp:        {np.max(data['temperature']):.2f} °C",
          f"    Min Voltage:     {np.min(data['voltage']):.2f} V",
          f"    Max R_int:       {np.max(data['resistance']):.4f} Ω",
          "="*72]
    return "\n".join(L)


# ═════════════════════════════════════════════════════════════════════════════
# 11.  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("Generating telemetry...")
    data = generate_telemetry()

    print("Computing Jacobian |dV/dI| (impedance)...")
    jacobian = compute_jacobian(data["voltage"], data["current"])

    print("Computing Hessian d²T/dt² (thermal acceleration)...")
    hessian, dT_dt = compute_hessian(data["temperature"])

    print("Taylor Series voltage prediction...")
    v_pred, v_o1, dV, d2V = taylor_predict(data["voltage"])

    print("Calibrating thresholds from healthy baseline...")
    j_thresh, h_thresh = calibrate_thresholds(jacobian, hessian)
    print(f"  Jacobian threshold: {j_thresh:.4f} Ω")
    print(f"  Hessian threshold:  {h_thresh:.6f} °C/s²")

    print("Running anomaly detection...")
    alerts = detect_anomalies(jacobian, hessian, v_pred, j_thresh, h_thresh)

    report = generate_report(data, jacobian, hessian, alerts, j_thresh, h_thresh)
    print("\n" + report)

    print("\nRendering dashboard...")
    fig = create_dashboard(data, jacobian, hessian, dT_dt, v_pred, v_o1, alerts,
                           j_thresh, h_thresh)
    out_png = "/home/claude/ups_anomaly_dashboard.png"
    fig.savefig(out_png, dpi=180, facecolor="#0a0e17")
    plt.close(fig)
    print(f"Dashboard saved: {out_png}")

    # Export JSON
    export = dict(
        time=data["time"].tolist(), voltage=data["voltage"].tolist(),
        current=data["current"].tolist(), temperature=data["temperature"].tolist(),
        resistance=data["resistance"].tolist(),
        jacobian=jacobian.tolist(), hessian=hessian.tolist(),
        dT_dt=dT_dt.tolist(),
        v_predicted=v_pred.tolist(), v_first_order=v_o1.tolist(),
        severity=alerts["severity"].tolist(), labels=data["labels"].tolist(),
        anomaly_onset=ANOMALY_ONSET,
        thresholds=dict(jacobian=j_thresh, hessian=h_thresh,
                        voltage_low=VOLTAGE_ALARM_LOW, temp_high=TEMP_ALARM_HIGH))
    with open("/home/claude/telemetry_data.json", "w") as f:
        json.dump(export, f)
    print("Data exported: telemetry_data.json")


if __name__ == "__main__":
    main()
