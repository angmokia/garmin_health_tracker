import datetime
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Setup ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Health Dashboard", layout="wide", page_icon="🏋️")

st.markdown("""
<style>
  .stApp { background-color: #0e1117; }
  .block-container { padding-top: 1rem; }
  .metric-card {
    background: #161b26; border: 1px solid #2a2f3e;
    border-radius: 8px; padding: 12px 16px; text-align: center;
  }
  .metric-label { font-size: 0.68rem; color: #8a94a6; letter-spacing: 0.08em;
                  text-transform: uppercase; margin-bottom: 3px; }
  .metric-value { font-size: 1.15rem; font-weight: 700; white-space: nowrap; color: #e0e0e0; }
  .metric-delta { font-size: 0.72rem; margin-top: 2px; }
  .metric-asof { font-size: 0.6rem; color: #5f6b7e; margin-top: 2px; }
  .positive { color: #26a69a; }
  .negative { color: #ef5350; }
  .neutral  { color: #8a94a6; }
  .section-header {
    font-size: 0.75rem; letter-spacing: 0.12em; text-transform: uppercase;
    color: #8a94a6; margin: 1.2rem 0 0.5rem;
    border-bottom: 1px solid #2a2f3e; padding-bottom: 4px;
  }
</style>
""", unsafe_allow_html=True)

TEMPLATE   = "plotly_dark"
PAPER_BG   = "#0e1117"
PLOT_BG    = "#161b26"
GRID_COLOR = "#2a2f3e"
MUTED      = "#8a94a6"

# Categorical slots (validated dark-mode order) — always assigned in this order, never cycled.
C = dict(blue="#3987e5", orange="#d95926", aqua="#199e70", yellow="#c98500",
         magenta="#d55181", green="#008300", violet="#9085e9", red="#e66767")
SERIES = list(C.values())
SEQ_BLUE = [[0, "#161b26"], [0.15, "#184f95"], [0.55, "#3987e5"], [1, "#9ec5f4"]]

TOKENSTORE = os.getenv("GARMINTOKENS_PATH", "~/.garminconnect")
# Personal health data lives outside the repo folder so it can never be pushed to GitHub.
CACHE_DIR  = Path(os.getenv("GARMIN_CACHE_DIR", "~/.garmin_health_cache")).expanduser()
TODAY      = datetime.date.today()

# ── Plain-language glossary (tooltips, per-tab explainers and the Guide tab) ───────
GLOSSARY = {
    "trend_delta": ("The green / red line under each number",
        "Compares your average over the **last 7 days** with the **28 days before that**. Green means it's moving "
        "the healthy way (e.g. HRV up, resting HR down), red the other way. One red week is normal — look for streaks."),
    "hrv": ("HRV — heart rate variability",
        "The tiny variation in time between heartbeats, measured while you sleep. More variation is **good**: it "
        "means your 'rest and recover' nervous system is in charge. HRV is very personal (someone at 40 ms isn't less "
        "fit than someone at 90 ms), so only compare yourself to **your own baseline** — the grey band is Garmin's "
        "normal range for you. Several nights below the band usually means poor sleep, stress, alcohol, illness or "
        "hard training is catching up: a good time for an easier day."),
    "baseline": ("Grey baseline band",
        "Your personal 'normal' HRV range, which Garmin recalculates from roughly the last 3 weeks. Inside the band = "
        "balanced; below it = your body is under more strain than usual."),
    "rhr": ("Resting heart rate",
        "Your lowest heart rate over the day and night. Lower generally means a fitter heart. If it sits **5+ bpm "
        "above your normal** for a few days, that often signals fatigue, oncoming illness, dehydration or alcohol."),
    "sleep_score": ("Sleep score",
        "Garmin's 0–100 rating of a night, combining how long you slept, time in each stage, restlessness and stress "
        "during sleep. 80+ good · 60–79 fair · under 60 poor."),
    "sleep_stages": ("Sleep stages",
        "**Deep** — physical repair and muscle recovery (key for lifting; mostly early in the night). **REM** — memory "
        "and learning (mostly late in the night, so it's what you lose when you cut sleep short). **Light** — most of "
        "the night, completely normal. **Awake** — short wake-ups are normal."),
    "body_battery": ("Body Battery",
        "Garmin's 0–100 energy estimate built from HRV, stress and activity. It charges while you rest and sleep and "
        "drains with exercise and stress. The **daily high** (usually on waking) shows how well you recharged; "
        "regularly waking below ~50 suggests you aren't recovering fully."),
    "readiness": ("Training readiness",
        "A 0–100 morning score combining last night's sleep, recovery time left from recent workouts, HRV, recent "
        "load and stress. 75+ go hard · 50–74 train normally · 25–49 keep it moderate · under 25 rest."),
    "stress": ("Stress",
        "Garmin's 0–100 score from your HRV while you're not exercising. 0–25 resting · 26–50 low · 51–75 medium · "
        "76–100 high. It's physical as well as mental: caffeine, a big meal, heat or getting sick all raise it."),
    "training_load": ("Training load (per workout)",
        "How hard a workout was on your body, from Garmin's EPOC estimate (the extra oxygen your body needs afterwards "
        "to recover, worked out from heart rate). Rough guide: easy walk ~20, lifting session 30–80, hard 10 km run "
        "150+. Lifting often scores lower than it feels because heart rate doesn't capture muscle strain."),
    "acute_chronic": ("Acute vs chronic load",
        "**Acute load** = your average daily training load over the **last 7 days** — how much you've done lately, "
        "i.e. your current fatigue. **Chronic load** = the average over the **last 28 days** — what your body is used "
        "to, i.e. your fitness base. Think of chronic as the bank balance and acute as this week's spending."),
    "acwr": ("Acute : chronic ratio",
        "Acute ÷ chronic. **1.0** = this week matches your usual. **0.8–1.3** = the 'sweet spot' for building up "
        "gradually. **Above 1.5** = you've ramped up much faster than your body is used to — sports science links "
        "this to higher injury risk, so ease in. **Below 0.8** = detraining, or a deliberate rest / deload week. It's a "
        "guide, not a rule: a normal week straight after a holiday will look like a spike."),
    "training_effect": ("Training effect (aerobic / anaerobic)",
        "0–5 scores for what a workout did for you. **Aerobic** = endurance (heart and lungs, steady efforts). "
        "**Anaerobic** = high-intensity capacity (sprints, intervals, heavy sets). 1–1.9 minor · 2–2.9 maintaining · "
        "3–3.9 improving · 4–4.9 highly improving · 5 overreaching."),
    "vo2max": ("VO₂ max",
        "The most oxygen your body can use per kg per minute — the best single measure of cardio fitness. Garmin "
        "estimates it from outdoor runs with heart rate. For men roughly: 35–45 average, 45–55 good, 55+ excellent "
        "(women ~5 lower). It moves slowly — judge it over months."),
    "efficiency": ("Aerobic efficiency (metres per heartbeat)",
        "Running speed divided by heart rate. If you cover more ground per heartbeat — same pace at a lower heart rate, "
        "or faster at the same heart rate — your aerobic fitness has improved. Compare similar easy runs; heat, hills "
        "and tiredness pull it down."),
    "pace": ("Pace", "Minutes per kilometre — **lower is faster**. 5:30 /km = 5 minutes 30 seconds per km."),
    "intensity_minutes": ("Intensity minutes",
        "Minutes with your heart rate in a moderate zone (brisk walking) or vigorous zone (running, hard training). "
        "Vigorous minutes count **double**. The WHO recommends 150 a week for general health."),
    "volume": ("Training volume",
        "Weight × reps added up across all sets (e.g. 3 sets of 10 at 60 kg = 1,800 kg). Gradually increasing weekly "
        "volume per muscle is one of the main drivers of muscle growth. Needs weights logged on your sets."),
    "sets_per_muscle": ("Sets per muscle per week",
        "How many working sets hit each muscle group in a week. Research suggests roughly **10–20 hard sets** per "
        "muscle per week for growth, and fewer to maintain. 'Other' = sets the watch couldn't identify."),
    "e1rm": ("Estimated 1-rep max (e1RM)",
        "The most you could probably lift for one rep, estimated from a normal set with the Epley formula "
        "(e.g. 100 kg × 5 reps ≈ 117 kg). It lets you compare sets of different reps without testing a true max."),
    "best_set": ("Best set (reps)",
        "The most reps you did in a single set of an exercise that session. Without logged weights, more reps in your "
        "best set is the clearest sign you're getting stronger."),
    "recovery_scatter": ("Load vs next-morning charts",
        "Each dot is one day: how hard you trained (left–right) against the next morning's HRV or resting HR. A "
        "downward-sloping HRV line (or upward RHR line) means hard days measurably cost you recovery. r is the "
        "correlation: near 0 = no clear link, ±0.3 = modest, ±0.5+ = strong."),
}

def explain(*keys, label="ℹ️ What do these mean?"):
    with st.expander(label):
        for k in keys:
            title, text = GLOSSARY[k]
            st.markdown(f"**{title}** — {text}")

# ── Chart helpers (same visual language as the macro dashboards) ───────────────

def base_layout(title="", height=380):
    return dict(
        template=TEMPLATE, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=14)),
        height=height,
        margin=dict(l=50, r=30, t=45, b=30),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center", font=dict(size=10)),
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
        hovermode="x unified",
    )

def csv_download(df: pd.DataFrame, label: str):
    buf = io.BytesIO()
    df.to_csv(buf)
    buf.seek(0)
    st.download_button("⬇ CSV", buf, file_name=f"{label}.csv",
                       mime="text/csv", key=f"dl_{label}")

def render_two_col(charts):
    """Render (title, fig [, df]) tuples in 2-column layout."""
    n, i = len(charts), 0
    while i < n:
        if i == n - 1 and n % 2 != 0:
            item = charts[i]
            st.plotly_chart(item[1], width="stretch", key=f"chart_{item[0]}")
            if len(item) > 2 and item[2] is not None:
                csv_download(item[2], item[0])
            i += 1
        else:
            c1, c2 = st.columns(2)
            for col, item in [(c1, charts[i]), (c2, charts[i+1])]:
                with col:
                    st.plotly_chart(item[1], width="stretch", key=f"chart_{item[0]}")
                    if len(item) > 2 and item[2] is not None:
                        csv_download(item[2], item[0])
            i += 2

def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def kpi_card(label, value, delta=None, good=None, asof=None, help=None):
    cls = "neutral" if good is None else ("positive" if good else "negative")
    delta_html = f'<div class="metric-delta {cls}">{delta}</div>' if delta else ""
    asof_html = f'<div class="metric-asof">{asof}</div>' if asof else ""
    tip = GLOSSARY.get(help, ("", ""))[1].replace('"', "&quot;").replace("**", "") if help else ""
    icon = ' <span style="cursor:help">ⓘ</span>' if tip else ""
    return (f'<div class="metric-card" title="{tip}"><div class="metric-label">{label}{icon}</div>'
            f'<div class="metric-value">{value}</div>{delta_html}{asof_html}</div>')

def kpi_row(cards):
    for col, html in zip(st.columns(len(cards)), cards):
        col.markdown(html, unsafe_allow_html=True)

def trend_fig(s: pd.Series, title, color, unit="", avg=7, height=340, band=None, kind="dots", per_obs=False):
    """Raw values faded + rolling average solid; optional (low, high) shaded band.
    per_obs: average over the last N observations (e.g. runs) rather than N calendar days."""
    s = s.dropna()
    fig = go.Figure()
    if band is not None:
        lo, hi = band[0].reindex(s.index), band[1].reindex(s.index)
        fig.add_trace(go.Scatter(x=s.index, y=hi, line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=s.index, y=lo, line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(138,148,166,0.15)", name="Baseline range",
                                 hoverinfo="skip"))
    if kind == "bars":
        fig.add_trace(go.Bar(x=s.index, y=s, name="Daily", marker_color=color, opacity=0.45,
                             hovertemplate=f"%{{y:,.0f}}{unit}<extra>Daily</extra>"))
    else:
        fig.add_trace(go.Scatter(x=s.index, y=s, mode="markers", name="Daily",
                                 marker=dict(color=color, size=6, opacity=0.4),
                                 hovertemplate=f"%{{y:,.1f}}{unit}<extra>Daily</extra>"))
    if avg and len(s) > avg:
        roll = s.rolling(avg, min_periods=2) if per_obs else s.rolling(f"{avg}D", min_periods=max(2, avg // 3))
        name = f"{avg}-session avg" if per_obs else f"{avg}d avg"
        fig.add_trace(go.Scatter(x=s.index, y=roll.mean(),
                                 mode="lines", name=name, line=dict(color=color, width=2),
                                 hovertemplate=f"%{{y:,.1f}}{unit}<extra>{name}</extra>"))
    fig.update_layout(**base_layout(title, height))
    return fig

def _last(s: pd.Series):
    s = s.dropna()
    return (s.iloc[-1], s.index[-1]) if len(s) else (None, None)

def _window_mean(s: pd.Series, start_days_ago, end_days_ago):
    s = s.dropna()
    if s.empty:
        return None
    end = pd.Timestamp(TODAY) - pd.Timedelta(days=end_days_ago)
    start = pd.Timestamp(TODAY) - pd.Timedelta(days=start_days_ago)
    v = s[(s.index > start) & (s.index <= end)].mean()
    return None if pd.isna(v) else float(v)

def delta_vs_baseline(s, higher_is_better=True, fmt="{:+.1f}"):
    """Last 7 days vs the 28 days before that — the usual 'is my recovery trending' read."""
    recent, base = _window_mean(s, 7, 0), _window_mean(s, 35, 7)
    if recent is None or base is None:
        return None, None
    d = recent - base
    if abs(d) < 1e-9:
        return "flat vs 28d baseline", None
    return f"{fmt.format(d)} 7d vs 28d", (d > 0) == higher_is_better

def _fmt_pace(min_per_km):
    if min_per_km is None or pd.isna(min_per_km) or not np.isfinite(min_per_km):
        return "–"
    m = int(min_per_km)
    return f"{m}:{int(round((min_per_km - m) * 60)):02d}"

# ── Garmin client ───────────────────────────────────────────────────────────────

# Owner's machine: a token file from garmin_login.py → auto-login + disk cache.
# Anywhere else (e.g. Streamlit Cloud): every visitor logs in with their own account, and their client
# and data live only in their own browser session's memory. GARMIN_HOSTED=1 forces hosted mode locally.
LOCAL_MODE = (Path(TOKENSTORE).expanduser() / "garmin_tokens.json").exists() and os.getenv("GARMIN_HOSTED") != "1"

def _finish_login(g):
    """Load the profile that Garmin's MFA-capable login path skips (display name is needed for daily endpoints)."""
    prof = g.client.connectapi("/userprofile-service/socialProfile") or {}
    g.display_name, g.full_name = prof.get("displayName"), prof.get("fullName", "")
    try:
        g.unit_system = g.client.connectapi(g.garmin_connect_user_settings_url)["userData"].get("measurementSystem")
    except Exception:
        g.unit_system = None
    return g

def session_client():
    """This browser session's Garmin client (st.session_state is per visitor — never st.cache_resource here)."""
    if "garmin" not in st.session_state and LOCAL_MODE:
        from garminconnect import Garmin
        g = Garmin()
        g.login(TOKENSTORE)
        st.session_state.garmin = g
    return st.session_state.get("garmin")

def _login_error(e):
    msg = str(e)
    if "401" in msg or "Invalid" in msg:
        return ("Garmin rejected that email / password. If you sign in to Garmin with Google or Apple, set a "
                "Garmin password first via **Forgot password?** on connect.garmin.com.")
    if "429" in msg or "Too many" in msg:
        return "Garmin is rate-limiting logins — wait a few minutes before trying again."
    return f"Login failed: {msg}"

def login_screen():
    from garminconnect import Garmin
    st.markdown("## Health & Training Dashboard")
    st.markdown("Sleep, recovery, strength, running and training load from **your own Garmin Connect account**, "
                "with every metric explained in plain language.")
    left, right = st.columns([3, 2], gap="large")
    with left:
        pending = st.session_state.get("_mfa")
        if pending:
            st.markdown("#### Enter your verification code")
            with st.form("mfa"):
                code = st.text_input("Code Garmin just sent you (email or authenticator app)")
                if st.form_submit_button("Verify", type="primary"):
                    g, state = pending
                    try:
                        g.resume_login(state, code.strip())
                        st.session_state.garmin = _finish_login(g)
                        del st.session_state["_mfa"]
                        st.rerun()
                    except Exception as e:
                        st.error(_login_error(e))
            if st.button("Start over"):
                del st.session_state["_mfa"]
                st.rerun()
            return
        st.markdown("#### Connect your Garmin account")
        pw_tab, tok_tab = st.tabs(["Email & password", "Use a token instead"])
        with pw_tab:
            with st.form("login"):
                email = st.text_input("Garmin Connect email")
                pw = st.text_input("Garmin Connect password", type="password")
                if st.form_submit_button("Connect", type="primary"):
                    with st.spinner("Signing in to Garmin…"):
                        try:
                            g = Garmin(email=email.strip(), password=pw, return_on_mfa=True)
                            needs_mfa, state = g.login()
                            if needs_mfa:
                                st.session_state._mfa = (g, state)
                            else:
                                st.session_state.garmin = _finish_login(g)
                            st.rerun()
                        except Exception as e:
                            st.error(_login_error(e))
        with tok_tab:
            st.caption("Prefer not to type your password into a website? Run `garmin_login.py` from this app's "
                       "repo on your own computer, answer **y** to print the token, and paste it here.")
            with st.form("token"):
                tok = st.text_area("Token string", height=100)
                if st.form_submit_button("Connect with token", type="primary"):
                    try:
                        g = Garmin()
                        g.login(tok.strip())
                        st.session_state.garmin = g
                        st.rerun()
                    except Exception as e:
                        st.error(_login_error(e))
    with right:
        st.markdown("#### Your data & privacy")
        st.markdown(
            "- Your password goes **straight to Garmin** to sign in. It's never saved.\n"
            "- The app keeps only Garmin's session token and your data **in this browser tab's memory**. "
            "Close the tab or press *Log out* and it's gone. Nothing is written to a database.\n"
            "- This is an unofficial personal project, not affiliated with Garmin, using the open-source "
            "[python-garminconnect](https://github.com/cyberjunky/python-garminconnect) library.\n"
            "- First load downloads your history (about 1–2 minutes for 90 days).")
        st.markdown("Just looking? Pick **Demo data** in the sidebar.")

def _safe(fn, *args):
    for attempt in range(3):
        try:
            return fn(*args)
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            return None

# ── Storage: owner → JSON on disk (history fetched once ever); visitors → session memory only ──

class Store:
    def __init__(self, disk_dir=None):
        self.dir = disk_dir
        self.mem = None if disk_dir else st.session_state.setdefault("_store", {})

    def get(self, kind, key):
        """(value, age in seconds) or (None, None)."""
        if self.dir:
            f = self.dir / kind / f"{key}.json"
            return (json.loads(f.read_text()), time.time() - f.stat().st_mtime) if f.exists() else (None, None)
        hit = self.mem.get((kind, str(key)))
        return (hit[1], time.time() - hit[0]) if hit else (None, None)

    def put(self, kind, key, value):
        if self.dir:
            (self.dir / kind).mkdir(parents=True, exist_ok=True)
            (self.dir / kind / f"{key}.json").write_text(json.dumps(value, default=str))
        else:
            self.mem[(kind, str(key))] = (time.time(), value)


def _parse_day(day: str, summary, sleep, hrv, readiness) -> dict:
    r = {"date": day}
    s = summary or {}
    r.update(
        steps=s.get("totalSteps"), rhr=s.get("restingHeartRate"),
        stress_avg=s.get("averageStressLevel") if (s.get("averageStressLevel") or -1) >= 0 else None,
        bb_high=s.get("bodyBatteryHighestValue"), bb_low=s.get("bodyBatteryLowestValue"),
        bb_charged=s.get("bodyBatteryChargedValue"), bb_drained=s.get("bodyBatteryDrainedValue"),
        kcal_total=s.get("totalKilocalories"), kcal_active=s.get("activeKilocalories"),
        im_moderate=s.get("moderateIntensityMinutes"), im_vigorous=s.get("vigorousIntensityMinutes"),
        floors=s.get("floorsAscended"),
    )
    d = (sleep or {}).get("dailySleepDTO") or {}
    if d.get("sleepTimeSeconds"):
        r.update(
            sleep_s=d.get("sleepTimeSeconds"), deep_s=d.get("deepSleepSeconds"),
            light_s=d.get("lightSleepSeconds"), rem_s=d.get("remSleepSeconds"),
            awake_s=d.get("awakeSleepSeconds"),
            sleep_score=((d.get("sleepScores") or {}).get("overall") or {}).get("value"),
        )
    h = (hrv or {}).get("hrvSummary") or {}
    if h:
        b = h.get("baseline") or {}
        r.update(hrv_night=h.get("lastNightAvg"), hrv_weekly=h.get("weeklyAvg"),
                 hrv_low=b.get("balancedLow"), hrv_high=b.get("balancedUpper"),
                 hrv_status=h.get("status"))
    tr = readiness[0] if isinstance(readiness, list) and readiness else readiness
    if isinstance(tr, dict) and tr.get("score") is not None:
        r.update(readiness=tr.get("score"), readiness_level=tr.get("level"))
    return r

def _fetch_day(client, day: str) -> dict:
    return _parse_day(day,
                      _safe(client.get_user_summary, day),
                      _safe(client.get_sleep_data, day),
                      _safe(client.get_hrv_data, day),
                      _safe(client.get_training_readiness, day))

def sync_daily(client, store: Store, days: int):
    """Fetch any day not stored yet (plus re-fetch the last 2 days, which are still filling in)."""
    todo = []
    for i in range(days):
        d = str(TODAY - datetime.timedelta(days=i))
        row, age = store.get("daily", d)
        if row is None or (i <= 1 and age > 1800):
            todo.append(d)
    if not todo:
        return
    bar = st.progress(0.0, text=f"Syncing {len(todo)} days from Garmin Connect…")
    with ThreadPoolExecutor(max_workers=4) as ex:
        for done, row in enumerate(ex.map(lambda d: _fetch_day(client, d), todo), 1):
            store.put("daily", row["date"], row)
            bar.progress(done / len(todo), text=f"Syncing {done}/{len(todo)} days from Garmin Connect…")
    bar.empty()

def read_daily(store: Store, days: int) -> pd.DataFrame:
    rows = [r for i in range(days) if (r := store.get("daily", str(TODAY - datetime.timedelta(days=i)))[0])]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()

def cached_call(store: Store, kind, key, ttl, fn, *args):
    """Activities / weigh-ins: re-pulled at most every `ttl` seconds instead of on every widget click."""
    val, age = store.get(kind, key)
    if val is None or age > ttl:
        val = _safe(fn, *args)
        if val is not None:
            store.put(kind, key, val)
    return val

def fetch_weight(client, store: Store, days: int) -> pd.DataFrame:
    start = TODAY - datetime.timedelta(days=days)
    data = cached_call(store, "weighins", f"{start}_{TODAY}", 1800, client.get_weigh_ins, str(start), str(TODAY)) or {}
    rows = []
    for w in data.get("dailyWeightSummaries") or []:
        lw = w.get("latestWeight") or {}
        if lw.get("weight"):
            rows.append(dict(date=w.get("summaryDate"), weight_kg=lw["weight"] / 1000,
                             bodyfat=lw.get("bodyFat"),
                             muscle_kg=(lw.get("muscleMass") or 0) / 1000 or None))
    if not rows:
        return pd.DataFrame(columns=["weight_kg", "bodyfat", "muscle_kg"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()

# ── Activities & strength sets ──────────────────────────────────────────────────

def _parse_activities(acts) -> pd.DataFrame:
    rows = []
    for a in acts or []:
        rows.append(dict(
            id=a.get("activityId"), name=a.get("activityName"),
            start=pd.to_datetime(a.get("startTimeLocal")),
            type=(a.get("activityType") or {}).get("typeKey", "other"),
            distance_km=(a.get("distance") or 0) / 1000,
            duration_min=(a.get("duration") or 0) / 60,
            avg_hr=a.get("averageHR"), max_hr=a.get("maxHR"), kcal=a.get("calories"),
            load=a.get("activityTrainingLoad"),
            te_aer=a.get("aerobicTrainingEffect"), te_ana=a.get("anaerobicTrainingEffect"),
            vo2max=a.get("vO2MaxValue"), elev_gain=a.get("elevationGain"),
            cadence=a.get("averageRunningCadenceInStepsPerMinute"),
            sets=a.get("totalSets"), reps=a.get("totalReps"),
        ))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = df["start"].dt.normalize()
    return df.sort_values("start")

def _parse_sets(activity_id, start, raw) -> list:
    out = []
    for s in (raw or {}).get("exerciseSets") or []:
        if s.get("setType") != "ACTIVE":
            continue
        ex = (s.get("exercises") or [{}])
        ex = max(ex, key=lambda e: e.get("probability") or 0) if ex else {}
        w = s.get("weight")
        out.append(dict(activity_id=activity_id, start=str(start),
                        category=ex.get("category") or "UNKNOWN",
                        exercise=ex.get("name") or ex.get("category") or "UNKNOWN",
                        reps=s.get("repetitionCount"),
                        weight_kg=(w / 1000) if w and w > 0 else 0.0))
    return out

def fetch_strength_sets(client, store: Store, strength: pd.DataFrame) -> pd.DataFrame:
    recent = pd.Timestamp(TODAY) - pd.Timedelta(days=14)
    def stale(r):
        val, age = store.get("sets", r.id)
        if val is None:
            return True
        # Sessions from the last 2 weeks get re-pulled every 6h — that's when you'd edit weights/names in Connect
        return r.start >= recent and age > 6 * 3600
    todo = [r for r in strength.itertuples() if stale(r)]
    if todo:
        bar = st.progress(0.0, text=f"Fetching sets for {len(todo)} strength sessions…")
        job = lambda r: (r.id, _parse_sets(r.id, r.start, _safe(client.get_activity_exercise_sets, r.id)))
        with ThreadPoolExecutor(max_workers=4) as ex:
            for i, (aid, rows) in enumerate(ex.map(job, todo), 1):
                store.put("sets", aid, rows)
                bar.progress(i / len(todo), text=f"Fetching sets {i}/{len(todo)}…")
        bar.empty()
    rows = []
    for r in strength.itertuples():
        rows.extend(store.get("sets", r.id)[0] or [])
    return pd.DataFrame(rows)

# ── Demo data (lets the dashboard run before Garmin is connected) ───────────────

DEMO_PROGRAM = {
    "Push": [("BENCH_PRESS", "BARBELL_BENCH_PRESS", 80), ("SHOULDER_PRESS", "OVERHEAD_BARBELL_PRESS", 50),
             ("BENCH_PRESS", "INCLINE_DUMBBELL_BENCH_PRESS", 30), ("TRICEPS_EXTENSION", "TRICEPS_PRESSDOWN", 30),
             ("LATERAL_RAISE", "DUMBBELL_LATERAL_RAISE", 10)],
    "Pull": [("DEADLIFT", "BARBELL_DEADLIFT", 140), ("PULL_UP", "WEIGHTED_PULL_UP", 15),
             ("ROW", "BARBELL_ROW", 70), ("CURL", "DUMBBELL_BICEPS_CURL", 15), ("SHRUG", "DUMBBELL_SHRUG", 35)],
    "Legs": [("SQUAT", "BARBELL_BACK_SQUAT", 115), ("DEADLIFT", "ROMANIAN_DEADLIFT", 90),
             ("LUNGE", "DUMBBELL_LUNGE", 24), ("LEG_CURL", "LEG_CURL", 45), ("CALF_RAISE", "STANDING_CALF_RAISE", 60),
             ("PLANK", "PLANK", 0)],
}

@st.cache_data(show_spinner=False)
def demo_data(days: int):
    rng = np.random.default_rng(7)
    dates = pd.date_range(end=pd.Timestamp(TODAY), periods=days, freq="D")
    acts, sets = [], []
    split = ["Push", "Pull", "Legs"]
    k, aid = 0, 1_000_000
    for i, d in enumerate(dates):
        prog = i / days
        wd = d.dayofweek
        if wd in (0, 1, 3, 4) and rng.random() > 0.08:
            day = split[k % 3]; k += 1; aid += 1
            start = d + pd.Timedelta(hours=7, minutes=int(rng.integers(0, 40)))
            n_sets, n_reps = 0, 0
            for cat, name, base in DEMO_PROGRAM[day]:
                top = base * (1 + 0.12 * prog) if base else 0
                for _ in range(int(rng.integers(3, 5))):
                    reps = int(rng.integers(5, 11)) if base else 1
                    w = round(top * (1.08 - reps * 0.025) / 2.5) * 2.5 if base else 0
                    sets.append(dict(activity_id=aid, start=str(start), category=cat, exercise=name,
                                     reps=reps, weight_kg=max(w, 0)))
                    n_sets += 1; n_reps += reps
            acts.append(dict(id=aid, name=f"{day} Day", start=start, type="strength_training",
                             distance_km=0, duration_min=float(rng.normal(62, 8)), avg_hr=float(rng.normal(112, 6)),
                             max_hr=float(rng.normal(158, 6)), kcal=float(rng.normal(380, 40)),
                             load=float(rng.normal(55, 12)), te_aer=1.8, te_ana=float(rng.normal(2.1, .4)),
                             vo2max=None, elev_gain=None, cadence=None, sets=n_sets, reps=n_reps))
        if (wd == 5 or (wd == 2 and rng.random() > 0.5)) and rng.random() > 0.15:
            aid += 1
            km = float(rng.uniform(5, 10) if wd == 5 else rng.uniform(3, 6))
            pace = float(rng.normal(5.9 - 0.4 * prog, 0.15))
            acts.append(dict(id=aid, name="Easy Run" if km < 7 else "Long Run",
                             start=d + pd.Timedelta(hours=8), type="running", distance_km=km,
                             duration_min=km * pace, avg_hr=float(rng.normal(152 - 5 * prog, 4)),
                             max_hr=float(rng.normal(172, 4)), kcal=km * 70, load=km * 14 + float(rng.normal(0, 8)),
                             te_aer=float(rng.normal(3.0, .4)), te_ana=0.6, vo2max=round(47 + 2.5 * prog + rng.normal(0, .3)),
                             elev_gain=km * 8, cadence=float(rng.normal(166, 3)), sets=None, reps=None))
        if wd == 5 and rng.random() > 0.7:
            aid += 1
            acts.append(dict(id=aid, name="Shakeout Run", start=d + pd.Timedelta(hours=18), type="running",
                             distance_km=3.0, duration_min=3.0 * 6.2, avg_hr=140.0, max_hr=155.0, kcal=210.0,
                             load=30.0, te_aer=2.0, te_ana=0.2, vo2max=None, elev_gain=20.0, cadence=164.0,
                             sets=None, reps=None))
    act = pd.DataFrame(acts)
    act["date"] = act["start"].dt.normalize()
    load_by_day = act.groupby("date")["load"].sum().reindex(dates, fill_value=0)

    n = len(dates)
    prev_load = load_by_day.shift(1, fill_value=0).values
    hrv = 58 + 6 * np.linspace(0, 1, n) + rng.normal(0, 5, n) - 0.04 * prev_load
    rhr = 54 - 3 * np.linspace(0, 1, n) + rng.normal(0, 1.5, n) + 0.02 * prev_load
    sleep_h = np.clip(rng.normal(7.2, 0.6, n) + np.where(dates.dayofweek >= 5, 0.5, 0), 4.5, 9.5)
    deep, rem = sleep_h * rng.uniform(.15, .22, n), sleep_h * rng.uniform(.18, .26, n)
    awake = rng.uniform(.1, .5, n)
    score = np.clip(45 + 5 * sleep_h + rng.normal(0, 5, n), 30, 98)
    daily = pd.DataFrame(dict(
        steps=np.clip(rng.normal(9000, 2500, n), 2000, None).round(), rhr=rhr.round(),
        stress_avg=np.clip(rng.normal(30, 7, n), 10, 70).round(),
        bb_high=np.clip(55 + 4 * (sleep_h - 6) + rng.normal(25, 8, n), 30, 100).round(),
        bb_low=np.clip(rng.normal(18, 7, n), 5, 50).round(),
        kcal_total=rng.normal(2600, 200, n).round(), kcal_active=rng.normal(650, 180, n).round(),
        im_moderate=np.clip(rng.normal(20, 12, n), 0, None).round(),
        im_vigorous=np.clip(load_by_day.values / 3 + rng.normal(0, 4, n), 0, None).round(),
        sleep_s=sleep_h * 3600, deep_s=deep * 3600, rem_s=rem * 3600, awake_s=awake * 3600,
        light_s=(sleep_h - deep - rem) * 3600, sleep_score=score.round(),
        hrv_night=hrv.round(), hrv_low=pd.Series(hrv).rolling(21, min_periods=1).mean().values - 7,
        hrv_high=pd.Series(hrv).rolling(21, min_periods=1).mean().values + 7,
        readiness=np.clip(30 + 0.6 * score - 0.25 * prev_load + rng.normal(0, 8, n), 5, 100).round(),
    ), index=dates)
    daily["hrv_weekly"] = daily["hrv_night"].rolling(7, min_periods=1).mean().round()
    wdates = dates[::3]
    weight = pd.DataFrame(dict(
        weight_kg=(81.5 + 1.8 * np.linspace(0, 1, len(wdates)) + rng.normal(0, .4, len(wdates))).round(1),
        bodyfat=(17.5 - 1.5 * np.linspace(0, 1, len(wdates)) + rng.normal(0, .3, len(wdates))).round(1),
    ), index=wdates)
    return daily, weight, act.sort_values("start"), pd.DataFrame(sets)

# ── Loading ──────────────────────────────────────────────────────────────────────

def load_live(client, days):
    store = Store(CACHE_DIR if LOCAL_MODE else None)
    sync_daily(client, store, days)
    daily = read_daily(store, days)
    weight = fetch_weight(client, store, days)
    start = str(TODAY - datetime.timedelta(days=days))
    acts = _parse_activities(cached_call(store, "activities", f"{start}_{TODAY}", 1800,
                                         client.get_activities_by_date, start, str(TODAY)))
    strength = acts[acts["type"].str.contains("strength")] if not acts.empty else acts
    sets = fetch_strength_sets(client, store, strength) if not strength.empty else pd.DataFrame()
    return daily, weight, acts, sets

client, client_err = None, None
try:
    client = session_client()
except Exception as e:  # saved token expired / revoked
    client_err = e

with st.sidebar:
    st.markdown("### 🏋️ Health Dashboard")
    source = st.radio("Data source", ["Garmin Connect", "Demo data"])
    hist_opts = [30, 60, 90, 180, 365, 730] if LOCAL_MODE else [30, 60, 90, 180, 365]
    days = st.select_slider("History", options=hist_opts, value=180 if LOCAL_MODE else 90,
                            format_func=lambda d: f"{d} days")
    imperial = getattr(client, "unit_system", None) == "statute_us"
    unit = st.radio("Weight unit", ["kg", "lb"], index=1 if imperial else 0, horizontal=True)
    if st.button("🔄 Refresh", width="stretch"):
        if LOCAL_MODE:
            st.session_state.pop("garmin", None)
        st.rerun()
    if client is not None and source == "Garmin Connect":
        name = getattr(client, "full_name", "") or getattr(client, "display_name", "")
        st.caption(f"Signed in{' as ' + name if name else ''}")
        if not LOCAL_MODE and st.button("Log out", width="stretch"):
            for k in ("garmin", "_store", "_mfa"):
                st.session_state.pop(k, None)
            st.rerun()
    if LOCAL_MODE:
        st.caption(f"Cache: `{CACHE_DIR}`")

W = 2.20462 if unit == "lb" else 1.0

if source == "Demo data":
    st.info("Showing synthetic demo data. Choose **Garmin Connect** in the sidebar to see your own.")
    daily, weight, acts, sets = demo_data(days)
elif client is None:
    if LOCAL_MODE and client_err is not None:
        st.error(f"Your saved Garmin login no longer works ({client_err}). Re-run the one-time login:")
        login_script = Path(__file__).with_name("garmin_login.py")
        st.code(f'"{sys.executable}" "{login_script}"', language="bash")
        st.stop()
    login_screen()
    st.stop()
else:
    daily, weight, acts, sets = load_live(client, days)

if daily.empty:
    st.warning("No daily wellness data came back for this period.")
    st.stop()

# ── Derived frames ───────────────────────────────────────────────────────────────

for col in ["steps", "rhr", "stress_avg", "bb_high", "bb_low", "kcal_total", "kcal_active", "im_moderate",
            "im_vigorous", "sleep_s", "deep_s", "light_s", "rem_s", "awake_s", "sleep_score",
            "hrv_night", "hrv_weekly", "hrv_low", "hrv_high", "readiness"]:
    if col not in daily:
        daily[col] = np.nan
    daily[col] = pd.to_numeric(daily[col], errors="coerce")
daily["sleep_h"] = daily["sleep_s"] / 3600
daily["im_weighted"] = daily["im_moderate"].fillna(0) + 2 * daily["im_vigorous"].fillna(0)
all_days = pd.date_range(end=pd.Timestamp(TODAY), periods=days, freq="D")

if acts.empty:
    acts = pd.DataFrame(columns=["id", "name", "start", "type", "distance_km", "duration_min", "avg_hr",
                                 "max_hr", "kcal", "load", "te_aer", "te_ana", "vo2max", "elev_gain",
                                 "cadence", "sets", "reps", "date"])
acts = acts.copy()
acts["load_eff"] = pd.to_numeric(acts["load"], errors="coerce").fillna(acts["duration_min"].astype(float))
acts["kind"] = np.select([acts["type"].str.contains("strength"), acts["type"].str.contains("running")],
                         ["Strength", "Running"], "Other")
runs = acts[acts["kind"] == "Running"].copy()
# Accidental starts (e.g. 11 m in 2 min) wreck pace stats: ignore anything under 0.5 km or 3 min
junk_runs = runs[(runs["distance_km"] < 0.5) | (runs["duration_min"] < 3)]
runs = runs.drop(junk_runs.index)
lifts = acts[acts["kind"] == "Strength"].copy()
if not runs.empty:
    runs["pace"] = runs["duration_min"] / runs["distance_km"].replace(0, np.nan)
    # Walk breaks / GPS glitches outside 2:30–15:00 /km aren't meaningful running paces
    runs.loc[~runs["pace"].between(2.5, 15), "pace"] = np.nan
    runs["efficiency"] = (runs["distance_km"] * 1000 / runs["duration_min"]) / runs["avg_hr"]  # m per heartbeat
    runs.loc[runs["pace"].isna(), "efficiency"] = np.nan

MUSCLE = {
    "Chest": ["BENCH_PRESS", "CHEST_FLYE", "FLYE", "PUSH_UP"],
    "Back": ["ROW", "PULL_UP", "DEADLIFT", "SHRUG", "HYPEREXTENSION", "PULLDOWN"],
    "Shoulders": ["SHOULDER_PRESS", "LATERAL_RAISE", "SHOULDER_STABILITY"],
    "Arms": ["CURL", "TRICEPS_EXTENSION"],
    "Legs": ["SQUAT", "LUNGE", "LEG_CURL", "CALF_RAISE", "HIP_RAISE", "HIP_STABILITY", "HIP_SWING", "OLYMPIC_LIFT"],
    "Core": ["PLANK", "CRUNCH", "CORE", "LEG_RAISE", "SIT_UP"],
}
MUSCLE_ORDER = list(MUSCLE) + ["Other"]
CAT2MUSCLE = {c: m for m, cats in MUSCLE.items() for c in cats}

if not sets.empty:
    sets = sets.copy()
    sets["start"] = pd.to_datetime(sets["start"])
    sets["date"] = sets["start"].dt.normalize()
    sets["reps"] = pd.to_numeric(sets["reps"], errors="coerce").fillna(0)
    sets["weight"] = pd.to_numeric(sets["weight_kg"], errors="coerce").fillna(0) * W
    sets["volume"] = sets["reps"] * sets["weight"]
    sets["muscle"] = sets["category"].map(CAT2MUSCLE).fillna("Other")
    sets["lift"] = sets["exercise"].replace("UNKNOWN", "Unidentified").str.replace("_", " ").str.title()
    # Epley e1RM, only trusted up to 12 reps
    sets["e1rm"] = np.where((sets["reps"] > 0) & (sets["reps"] <= 12) & (sets["weight"] > 0),
                            sets["weight"] * (1 + sets["reps"] / 30), np.nan)

def weekly(s: pd.Series, how="sum"):
    return getattr(s.resample("W-MON", label="left", closed="left"), how)()

# ── Header ───────────────────────────────────────────────────────────────────────

last_sync = daily.index.max().strftime("%a %d %b %Y")
st.markdown(f"## Health & Training Dashboard <span style='font-size:0.8rem;color:{MUTED}'>· through {last_sync}</span>",
            unsafe_allow_html=True)

tabs = st.tabs(["Overview", "Recovery & Sleep", "Strength", "Running", "Body & Activity", "Training Load", "Data", "Guide"])

# ── 1. Overview ──────────────────────────────────────────────────────────────────
with tabs[0]:
    section("Recovery — latest reading, trend = last 7d vs prior 28d")
    cards = []
    for label, col, hib, fmt, unit_s, hk in [
        ("Resting HR", "rhr", False, "{:.0f}", " bpm", "rhr"), ("HRV (overnight)", "hrv_night", True, "{:.0f}", " ms", "hrv"),
        ("Sleep score", "sleep_score", True, "{:.0f}", "", "sleep_score"), ("Sleep", "sleep_h", True, "{:.1f}", " h", "sleep_stages"),
        ("Body Battery peak", "bb_high", True, "{:.0f}", "", "body_battery"),
        ("Training readiness", "readiness", True, "{:.0f}", "", "readiness"),
        ("Avg stress", "stress_avg", False, "{:.0f}", "", "stress"),
    ]:
        v, d = _last(daily[col])
        dl, good = delta_vs_baseline(daily[col], hib)
        cards.append(kpi_card(label, "–" if v is None else fmt.format(v) + unit_s, dl, good,
                              d.strftime("%d %b") if d is not None else "no data", help=hk))
    kpi_row(cards)
    explain("trend_delta", "rhr", "hrv", "sleep_score", "body_battery", "readiness", "stress",
            label="ℹ️ What do these numbers mean? (or hover any card)")

    section("Training — this week (Mon–today) vs 4-week weekly average")
    wk_start = pd.Timestamp(TODAY) - pd.Timedelta(days=TODAY.weekday())
    four_wk = wk_start - pd.Timedelta(weeks=4)

    def wk_vs_avg(df, col, fn="sum"):
        this = getattr(df[df["date"] >= wk_start][col], fn)() if not df.empty else 0
        prev = getattr(df[(df["date"] >= four_wk) & (df["date"] < wk_start)][col], fn)() / 4 if not df.empty else 0
        return this, prev

    n_l, a_l = wk_vs_avg(lifts, "id", "count")
    v_this = sets[sets["date"] >= wk_start]["volume"].sum() if not sets.empty else 0
    v_prev = sets[(sets["date"] >= four_wk) & (sets["date"] < wk_start)]["volume"].sum() / 4 if not sets.empty else 0
    km_this, km_prev = wk_vs_avg(runs, "distance_km")
    im_this = daily.loc[daily.index >= wk_start, "im_weighted"].sum()
    w7 = _window_mean(weight["weight_kg"] * W, 7, 0) if not weight.empty else None
    w_d, w_good = delta_vs_baseline(weight["weight_kg"] * W, True) if not weight.empty else (None, None)
    kpi_row([
        kpi_card("Lifting sessions", f"{n_l}", f"avg {a_l:.1f}/wk"),
        kpi_card("Lifting volume", f"{v_this/1000:,.1f} k{unit}", f"avg {v_prev/1000:,.1f} k{unit}/wk", help="volume"),
        kpi_card("Running", f"{km_this:.1f} km", f"avg {km_prev:.1f} km/wk"),
        kpi_card("Intensity minutes", f"{im_this:.0f} / 150", "vigorous counts 2×",
                 good=im_this >= 150 if TODAY.weekday() >= 5 else None, help="intensity_minutes"),
        kpi_card("Weight (7d avg)", "–" if w7 is None else f"{w7:.1f} {unit}", w_d, None),
        kpi_card("Steps (7d avg)", f"{_window_mean(daily['steps'], 7, 0) or 0:,.0f}",
                 *delta_vs_baseline(daily["steps"], True, "{:+,.0f}")),
    ])

    section("Readiness signals")
    hrv_fig = trend_fig(daily["hrv_night"], "Overnight HRV vs baseline (ms)", C["blue"], " ms",
                        band=(daily["hrv_low"], daily["hrv_high"]))
    rhr_fig = trend_fig(daily["rhr"], "Resting heart rate (bpm)", C["orange"], " bpm")
    render_two_col([("ov_hrv", hrv_fig), ("ov_rhr", rhr_fig)])
    explain("hrv", "baseline", "rhr", label="ℹ️ How to read these two charts")

    section("Training calendar")
    weeks_shown = min(26, days // 7)
    cal_days = pd.date_range(end=pd.Timestamp(TODAY), periods=weeks_shown * 7 + TODAY.weekday() + 1 - 7, freq="D")
    load_day = acts.groupby("date")["load_eff"].sum().reindex(cal_days, fill_value=0)
    names = acts.groupby("date")["name"].apply(lambda x: " + ".join(x.astype(str))).reindex(cal_days, fill_value="Rest")
    cal = pd.DataFrame({"load": load_day, "name": names})
    cal["week"] = cal.index - pd.to_timedelta(cal.index.dayofweek, unit="D")
    cal["dow"] = cal.index.dayofweek
    z = cal.pivot(index="dow", columns="week", values="load")
    txt = cal.pivot(index="dow", columns="week", values="name")
    dts = cal.reset_index().pivot(index="dow", columns="week", values="index")
    fig = go.Figure(go.Heatmap(
        z=z.values, x=z.columns, y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        customdata=np.dstack([txt.values, dts.apply(lambda c: c.dt.strftime("%a %d %b")).values]),
        colorscale=SEQ_BLUE, xgap=3, ygap=3, colorbar=dict(title="Load", thickness=10),
        hovertemplate="%{customdata[1]}<br>%{customdata[0]}<br>Load %{z:.0f}<extra></extra>"))
    lay = base_layout("Daily training load (darker = harder)", 260)
    lay.update(yaxis=dict(autorange="reversed", showgrid=False), xaxis=dict(showgrid=False), hovermode="closest")
    fig.update_layout(**lay)
    st.plotly_chart(fig, width="stretch", key="ov_cal")
    explain("training_load", label="ℹ️ What is 'load'?")

# ── 2. Recovery & Sleep ──────────────────────────────────────────────────────────
with tabs[1]:
    section("Sleep")
    stg = daily[["deep_s", "light_s", "rem_s", "awake_s"]].div(3600).dropna(how="all")
    fig = go.Figure()
    for (col, name), color in zip([("deep_s", "Deep"), ("light_s", "Light"), ("rem_s", "REM"), ("awake_s", "Awake")],
                                  [C["blue"], C["aqua"], C["violet"], C["yellow"]]):
        fig.add_trace(go.Bar(x=stg.index, y=stg[col], name=name, marker_color=color,
                             marker_line=dict(color=PAPER_BG, width=1),
                             hovertemplate=f"{name} %{{y:.1f}} h<extra></extra>"))
    fig.update_layout(**base_layout("Sleep stages (hours)"), barmode="stack", bargap=0.15)
    render_two_col([
        ("sleep_stages", fig, stg.round(2)),
        ("sleep_score", trend_fig(daily["sleep_score"], "Sleep score", C["blue"]), daily[["sleep_score"]]),
    ])
    explain("sleep_stages", "sleep_score")

    section("Recovery")
    bb = daily[["bb_low", "bb_high"]].dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bb.index, y=bb["bb_low"], name="Daily low", mode="lines",
                             line=dict(color=C["orange"], width=1.5), hovertemplate="Low %{y:.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=bb.index, y=bb["bb_high"], name="Daily high", mode="lines", fill="tonexty",
                             fillcolor="rgba(57,135,229,0.15)", line=dict(color=C["blue"], width=1.5),
                             hovertemplate="High %{y:.0f}<extra></extra>"))
    fig.update_layout(**base_layout("Body Battery — daily high & low"))
    fig.update_yaxes(range=[0, 100])
    readiness_fig = trend_fig(daily["readiness"], "Training readiness", C["blue"])
    readiness_fig.add_hrect(y0=0, y1=25, fillcolor="rgba(230,103,103,0.08)", line_width=0)
    readiness_fig.add_hrect(y0=75, y1=100, fillcolor="rgba(38,166,154,0.08)", line_width=0)
    render_two_col([
        ("bb_range", fig, bb),
        ("readiness", readiness_fig, daily[["readiness"]]),
        ("stress", trend_fig(daily["stress_avg"], "Average stress", C["orange"]), daily[["stress_avg"]]),
        ("hrv_weekly", trend_fig(daily["hrv_weekly"], "HRV 7-day average (ms)", C["blue"], " ms", avg=0,
                                 band=(daily["hrv_low"], daily["hrv_high"])), daily[["hrv_weekly", "hrv_low", "hrv_high"]]),
    ])
    explain("body_battery", "readiness", "stress", "hrv", "baseline")

    section("Does training cost me recovery? — yesterday's load vs this morning")
    prev_load = acts.groupby("date")["load_eff"].sum().reindex(daily.index, fill_value=0).shift(1)
    rel = pd.DataFrame({"load": prev_load, "hrv": daily["hrv_night"], "sleep": daily["sleep_score"],
                        "rhr": daily["rhr"]}).dropna(subset=["load"])
    charts = []
    for col, label, color in [("hrv", "Overnight HRV (ms)", C["blue"]), ("rhr", "Resting HR (bpm)", C["orange"])]:
        d = rel[["load", col]].dropna()
        fig = go.Figure(go.Scatter(x=d["load"], y=d[col], mode="markers",
                                   marker=dict(color=color, size=8, opacity=0.6, line=dict(color=PAPER_BG, width=1)),
                                   customdata=d.index.strftime("%d %b"),
                                   hovertemplate="%{customdata}<br>Load %{x:.0f} → %{y:.0f}<extra></extra>",
                                   showlegend=False))
        if len(d) > 10 and d["load"].std() > 0:
            b, a = np.polyfit(d["load"], d[col], 1)
            xs = np.array([d["load"].min(), d["load"].max()])
            r = d["load"].corr(d[col])
            fig.add_trace(go.Scatter(x=xs, y=a + b * xs, mode="lines", line=dict(color=MUTED, width=2, dash="dash"),
                                     name=f"fit (r = {r:.2f})", hoverinfo="skip"))
        lay = base_layout(f"{label} vs previous day's training load")
        lay.update(hovermode="closest", xaxis=dict(title="Previous-day load", gridcolor=GRID_COLOR))
        fig.update_layout(**lay)
        charts.append((f"rel_{col}", fig))
    render_two_col(charts)
    explain("recovery_scatter", "training_load", label="ℹ️ How to read these charts")

# ── 3. Strength ──────────────────────────────────────────────────────────────────
with tabs[2]:
    if lifts.empty or sets.empty:
        st.info("No strength-training activities with logged sets in this period.")
    else:
        # Garmin only records load if you enter it on the watch / in Connect. Without it, fall back to reps.
        has_w = (sets["weight"] > 0).mean() >= 0.3
        last28 = pd.Timestamp(TODAY) - pd.Timedelta(days=28)
        s28 = sets[sets["date"] > last28]
        work_col, work_lbl = ("volume", f"volume ({unit} × reps)") if has_w else ("reps", "reps")
        if not has_w:
            st.caption("Tracking sets & reps (no weights logged). If you ever enter weights on the watch or in "
                       "Connect, volume and estimated-1RM charts switch on automatically.")
        known = (sets["category"] != "UNKNOWN").mean()
        kpi_row([
            kpi_card("Sessions (28d)", f"{(lifts['date'] > last28).sum()}"),
            kpi_card("Sets (28d)", f"{len(s28):,}"),
            kpi_card("Volume (28d)", f"{s28['volume'].sum()/1000:,.1f} k{unit}") if has_w
            else kpi_card("Reps (28d)", f"{s28['reps'].sum():,.0f}"),
            kpi_card("Avg session", f"{lifts.loc[lifts['date'] > last28, 'duration_min'].mean():.0f} min"),
            kpi_card("Sets identified", f"{known:.0%}", "auto-detected by watch", good=known >= 0.8),
        ])

        section("Weekly work by muscle group")
        piv_v = sets.pivot_table(index="date", columns="muscle", values=work_col, aggfunc="sum")
        piv_s = sets.pivot_table(index="date", columns="muscle", values="reps", aggfunc="count")
        charts = []
        for key, piv, title, fmt in [("vol", piv_v, f"Weekly {work_lbl}", ",.0f"),
                                     ("sets", piv_s, "Weekly working sets", ".0f")]:
            wk = piv.reindex(all_days).fillna(0).resample("W-MON", label="left", closed="left").sum()
            fig = go.Figure()
            for m, color in zip(MUSCLE_ORDER, SERIES):
                if m in wk and wk[m].sum() > 0:
                    fig.add_trace(go.Bar(x=wk.index, y=wk[m], name=m, marker_color=color,
                                         marker_line=dict(color=PAPER_BG, width=1),
                                         hovertemplate=f"{m} %{{y:{fmt}}}<extra></extra>"))
            fig.update_layout(**base_layout(title), barmode="stack", bargap=0.2)
            charts.append((f"str_{key}", fig, wk))
        render_two_col(charts)
        explain("sets_per_muscle", "volume")

        st.caption("Hypertrophy research commonly cites ~10–20 hard sets per muscle per week. "
                   "'Other' includes sets the watch couldn't identify — renaming them in Connect fixes the split.")

        named = sets[sets["category"] != "UNKNOWN"]
        freq = named.groupby("lift")["activity_id"].nunique().sort_values(ascending=False)
        if has_w:
            section("Strength progression — estimated 1RM (Epley, sets ≤12 reps)")
            metric, m_lbl, m_unit = "e1rm", f"Best estimated 1RM per session ({unit})", f" {unit}"
            ranked = named[named["lift"].isin(freq[freq >= 3].index)].groupby("lift")["e1rm"].max().sort_values(ascending=False)
        else:
            section("Progression — most reps in a single set, per session")
            metric, m_lbl, m_unit = "reps", "Best set (reps) per session", " reps"
            ranked = freq
        pick = st.multiselect("Exercises (max 3 so lines stay distinguishable)", list(freq.index),
                              default=list(ranked.index[:3]), max_selections=3)
        fig = go.Figure()
        for lift, color in zip(pick, SERIES):
            best = named[named["lift"] == lift].groupby("date")[metric].max().dropna()
            fig.add_trace(go.Scatter(x=best.index, y=best, mode="markers+lines", name=lift,
                                     line=dict(color=color, width=2), marker=dict(size=8, line=dict(color=PAPER_BG, width=2)),
                                     hovertemplate=f"{lift}<br>%{{y:.1f}}{m_unit}<extra></extra>"))
        fig.update_layout(**base_layout(m_lbl, 420))
        st.plotly_chart(fig, width="stretch", key="progression")
        explain("e1rm" if has_w else "best_set", "e1rm" if not has_w else "volume", label="ℹ️ What is this measuring?")

        section("Exercise summary")
        grp = sets.groupby("lift")
        recent_cut = pd.Timestamp(TODAY) - pd.Timedelta(days=28)
        base = {
            "Muscle": grp["muscle"].first(),
            "Sessions": grp["activity_id"].nunique(),
            "Sets": grp.size(),
            "Total reps": grp["reps"].sum(),
            "Avg reps/set": grp["reps"].mean(),
        }
        if has_w:
            top = sets.loc[sets.dropna(subset=["e1rm"]).groupby("lift")["e1rm"].idxmax()].set_index("lift")
            base.update({
                f"Heaviest ({unit})": sets[sets["weight"] > 0].groupby("lift")["weight"].max(),
                f"Best e1RM ({unit})": top["e1rm"],
                "Best set": top.apply(lambda r: f"{r['weight']:.1f} × {r['reps']:.0f}", axis=1),
                "PR date": top["date"].dt.strftime("%d %b %y"),
                f"e1RM last 28d ({unit})": sets[sets["date"] > recent_cut].groupby("lift")["e1rm"].max(),
            })
        else:
            top = sets.loc[grp["reps"].idxmax()].set_index("lift")
            base.update({
                "Best set (reps)": top["reps"],
                "PR date": top["date"].dt.strftime("%d %b %y"),
                "Best set last 28d": sets[sets["date"] > recent_cut].groupby("lift")["reps"].max(),
            })
        base["Last done"] = grp["date"].max().dt.strftime("%d %b")
        summary = pd.DataFrame(base).sort_values("Sessions", ascending=False)
        st.dataframe(summary.round(1), width="stretch")
        csv_download(summary, "exercise_summary")

        section("Recent sessions")
        last_ids = lifts.sort_values("start", ascending=False)["id"].head(10)
        for aid in last_ids:
            a = lifts[lifts["id"] == aid].iloc[0]
            ss = sets[sets["activity_id"] == aid]
            work = f"{ss['volume'].sum():,.0f} {unit} volume" if has_w else f"{ss['reps'].sum():.0f} reps"
            with st.expander(f"{a['start']:%a %d %b} · {a['name']} · {a['duration_min']:.0f} min · "
                             f"{len(ss)} sets · {work}"):
                tbl = (ss.groupby("lift", sort=False)[["weight", "reps"]]
                         .apply(lambda x: " · ".join(f"{w:g}×{r:.0f}" if w else f"{r:.0f}"
                                                     for w, r in zip(x["weight"].round(1), x["reps"])))
                         .rename("Sets (weight × reps)" if has_w else "Reps per set"))
                st.dataframe(tbl, width="stretch")

# ── 4. Running ───────────────────────────────────────────────────────────────────
with tabs[3]:
    if not junk_runs.empty:
        st.caption(f"Ignoring {len(junk_runs)} accidental run recording(s) under 0.5 km or 3 min.")
    if runs.empty:
        st.info("No runs in this period.")
    else:
        if len(runs) < 6:
            st.caption(f"Only {len(runs)} run(s) in the selected period — trends (median pace, efficiency) "
                       "need a few more runs to mean much.")
        last28 = pd.Timestamp(TODAY) - pd.Timedelta(days=28)
        r28 = runs[runs["date"] > last28]
        vo2, vo2_d = _last(runs.set_index("date")["vo2max"].astype(float))
        pace28 = r28["duration_min"].sum() / r28["distance_km"].sum() if r28["distance_km"].sum() else None
        kpi_row([
            kpi_card("Runs (28d)", f"{len(r28)}"),
            kpi_card("Distance (28d)", f"{r28['distance_km'].sum():.1f} km"),
            kpi_card("Avg pace (28d)", f"{_fmt_pace(pace28)} /km", help="pace"),
            kpi_card("Avg HR (28d)", f"{r28['avg_hr'].mean():.0f} bpm" if len(r28) else "–"),
            kpi_card("Longest run", f"{runs['distance_km'].max():.1f} km"),
            kpi_card("VO₂ max", "–" if vo2 is None else f"{vo2:.0f}", asof=vo2_d.strftime("%d %b") if vo2_d is not None else None,
                     help="vo2max"),
        ])

        section("Volume & fitness")
        # groupby first: two runs on one day would give duplicate dates, which reindex can't handle
        wk_km = runs.groupby("date")["distance_km"].sum().reindex(all_days, fill_value=0).resample(
            "W-MON", label="left", closed="left").sum()
        fig = go.Figure(go.Bar(x=wk_km.index, y=wk_km, name="Weekly km", marker_color=C["blue"],
                               hovertemplate="%{y:.1f} km<extra>Week</extra>"))
        fig.add_trace(go.Scatter(x=wk_km.index, y=wk_km.rolling(4, min_periods=1).mean(), name="4-wk avg",
                                 line=dict(color=C["orange"], width=2), hovertemplate="%{y:.1f} km<extra>4-wk avg</extra>"))
        fig.update_layout(**base_layout("Weekly running distance (km)"))

        rp = runs.set_index("start").sort_index().dropna(subset=["pace"])  # start time is unique per run; date isn't
        pace_fig = go.Figure(go.Scatter(x=rp.index, y=rp["pace"], mode="markers", name="Run",
                                        marker=dict(color=C["blue"], size=np.clip(rp["distance_km"] * 2, 6, 20), opacity=0.6,
                                                    line=dict(color=PAPER_BG, width=1)),
                                        customdata=np.c_[rp["pace"].map(_fmt_pace), rp["distance_km"]],
                                        hovertemplate="%{customdata[0]} /km · %{customdata[1]:.1f} km<extra></extra>"))
        if len(rp) >= 6:
            pace_fig.add_trace(go.Scatter(x=rp.index, y=rp["pace"].rolling(5, min_periods=2).median(), name="5-run median",
                                          line=dict(color=C["blue"], width=2), hoverinfo="skip"))
        lay = base_layout("Pace (min/km, lower = faster; dot size = distance)")
        # Pick a tick step that gives ~4–8 labels whatever the spread (15 s, 30 s, 1 min, 2 min…)
        lo_p, hi_p = rp["pace"].min(), rp["pace"].max()
        step = next((st_ for st_ in (0.25, 0.5, 1, 2, 5) if (hi_p - lo_p) / st_ <= 6), 5)
        ticks = np.arange(np.floor(lo_p / step) * step, hi_p + step, step)
        lay.update(yaxis=dict(autorange="reversed", gridcolor=GRID_COLOR,
                              tickvals=ticks, ticktext=[_fmt_pace(t) for t in ticks]))
        pace_fig.update_layout(**lay)

        eff_fig = trend_fig(rp["efficiency"], "Aerobic efficiency — metres per heartbeat (↑ = fitter)",
                            C["aqua"], " m", avg=5, per_obs=True)
        vo2_fig = trend_fig(rp["vo2max"].astype(float), "VO₂ max (Garmin estimate)", C["violet"], avg=0)
        render_two_col([("run_km", fig, wk_km.to_frame("km")), ("run_pace", pace_fig),
                        ("run_eff", eff_fig), ("run_vo2", vo2_fig)])
        explain("pace", "efficiency", "vo2max", "training_effect")

        section("Run log")
        log = runs.sort_values("start", ascending=False).assign(
            Pace=lambda d: d["pace"].map(_fmt_pace))[
            ["start", "name", "distance_km", "duration_min", "Pace", "avg_hr", "max_hr", "cadence", "elev_gain", "te_aer", "load"]]
        log.columns = ["Start", "Name", "Km", "Min", "Pace /km", "Avg HR", "Max HR", "Cadence", "Elev +m", "Aerobic TE", "Load"]
        st.dataframe(log.round(1), width="stretch", hide_index=True)
        csv_download(log, "runs")

# ── 5. Body & Activity ───────────────────────────────────────────────────────────
with tabs[4]:
    section("Body composition")
    if weight.empty:
        st.info("No weigh-ins logged in this period (Garmin Index scale or manual entries in Connect).")
    else:
        wt = weight["weight_kg"] * W
        charts = [("weight", trend_fig(wt.asfreq("D"), f"Body weight ({unit})", C["blue"], f" {unit}"), wt.to_frame())]
        if weight["bodyfat"].notna().sum() > 2:
            charts.append(("bodyfat", trend_fig(weight["bodyfat"].asfreq("D"), "Body fat (%)", C["orange"], "%"),
                           weight[["bodyfat"]]))
        if "muscle_kg" in weight and weight["muscle_kg"].notna().sum() > 2:
            charts.append(("muscle", trend_fig((weight["muscle_kg"] * W).asfreq("D"), f"Muscle mass ({unit})",
                                               C["aqua"], f" {unit}"), weight[["muscle_kg"]]))
        render_two_col(charts)
        if len(wt.dropna()) > 7:
            span = (wt.index.max() - wt.index.min()).days / 7
            def _slope(x):
                x = x.dropna()
                return np.polyfit((x.index - x.index.min()).days, x.values, 1)[0] * 7 if len(x) >= 3 else None
            slope = _slope(wt)
            s4 = _slope(wt[wt.index > pd.Timestamp(TODAY) - pd.Timedelta(days=28)])
            recent_txt = f" · last 4 weeks: **{s4:+.2f} {unit}/week**" if s4 is not None else ""
            st.caption(f"Trend over {span:.0f} weeks: **{slope:+.2f} {unit}/week**{recent_txt} "
                       f"(range {wt.min():.1f}–{wt.max():.1f} {unit}; lean-bulk guidance is often "
                       "~0.25–0.5% of bodyweight per week).")

    section("Daily movement")
    im = daily[["im_moderate", "im_vigorous"]].fillna(0)
    im_wk = pd.DataFrame({"Moderate": weekly(im["im_moderate"]), "Vigorous (×2)": weekly(im["im_vigorous"]) * 2})
    fig = go.Figure()
    for col, color in zip(im_wk.columns, [C["blue"], C["orange"]]):
        fig.add_trace(go.Bar(x=im_wk.index, y=im_wk[col], name=col, marker_color=color,
                             marker_line=dict(color=PAPER_BG, width=1), hovertemplate=f"{col} %{{y:.0f}} min<extra></extra>"))
    fig.add_hline(y=150, line=dict(color=MUTED, dash="dash", width=1), annotation_text="WHO 150 min",
                  annotation_font_color=MUTED)
    fig.update_layout(**base_layout("Weekly intensity minutes"), barmode="stack", bargap=0.2)
    render_two_col([
        ("steps", trend_fig(daily["steps"], "Daily steps", C["blue"], kind="bars"), daily[["steps"]]),
        ("im", fig, im_wk),
        ("kcal_active", trend_fig(daily["kcal_active"], "Active calories (kcal)", C["orange"], " kcal", kind="bars"),
         daily[["kcal_active", "kcal_total"]]),
        ("kcal_total", trend_fig(daily["kcal_total"], "Total calories burned (kcal)", C["aqua"], " kcal"), None),
    ])
    explain("intensity_minutes")

# ── 6. Training Load ─────────────────────────────────────────────────────────────
with tabs[5]:
    load = acts.groupby("date")["load_eff"].sum().reindex(all_days, fill_value=0)
    acute, chronic = load.rolling(7, min_periods=1).mean(), load.rolling(28, min_periods=7).mean()
    acwr = (acute / chronic).where(chronic > 0)
    v, _ = _last(acwr)
    zone = ("–" if v is None else "Undertraining" if v < 0.8 else "Sweet spot" if v <= 1.3
            else "Elevated" if v <= 1.5 else "Spike risk")
    kpi_row([
        kpi_card("Acute load (7d avg)", f"{acute.iloc[-1]:.0f}", "recent fatigue", help="acute_chronic"),
        kpi_card("Chronic load (28d avg)", f"{chronic.iloc[-1]:.0f}" if pd.notna(chronic.iloc[-1]) else "–",
                 "fitness base", help="acute_chronic"),
        kpi_card("Acute : chronic", "–" if v is None else f"{v:.2f}", zone,
                 None if v is None else 0.8 <= v <= 1.3, help="acwr"),
        kpi_card("Rest days (28d)", f"{(load.iloc[-28:] == 0).sum()}"),
    ])
    if v is not None:
        st.markdown(f"**In plain terms:** over the last 7 days you've trained at **{v:.1f}×** your usual 4-week level — "
                    + ("well below normal (a rest / deload phase, or detraining if unplanned)." if v < 0.8 else
                       "a steady, sustainable build." if v <= 1.3 else
                       "a noticeable step up; fine short-term, but don't stack several weeks like this." if v <= 1.5 else
                       "a sharp jump compared with what your body is used to — the range linked to higher injury risk. "
                       "Consider an easier few days or build up more gradually."))
    explain("acute_chronic", "acwr", "training_load", "training_effect")
    st.caption("Load = Garmin's per-activity training load (EPOC); activities without one fall back to minutes.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=all_days, y=acute, name="Acute (7d)", line=dict(color=C["blue"], width=2)))
    fig.add_trace(go.Scatter(x=all_days, y=chronic, name="Chronic (28d)", line=dict(color=C["orange"], width=2)))
    fig.update_layout(**base_layout("Acute vs chronic load"))
    r_fig = go.Figure(go.Scatter(x=all_days, y=acwr, name="ACWR", line=dict(color=C["blue"], width=2),
                                 hovertemplate="%{y:.2f}<extra>ACWR</extra>"))
    r_fig.add_hrect(y0=0.8, y1=1.3, fillcolor="rgba(38,166,154,0.12)", line_width=0,
                    annotation_text="sweet spot 0.8–1.3", annotation_font_color=MUTED, annotation_position="top left")
    r_fig.add_hline(y=1.5, line=dict(color=C["red"], dash="dash", width=1))
    r_fig.update_layout(**base_layout("Acute : chronic workload ratio"))

    by_kind = acts.pivot_table(index="date", columns="kind", values="duration_min", aggfunc="sum")
    by_kind = by_kind.reindex(all_days).fillna(0).resample("W-MON", label="left", closed="left").sum() / 60
    k_fig = go.Figure()
    for kind, color in zip(["Strength", "Running", "Other"], SERIES):
        if kind in by_kind:
            k_fig.add_trace(go.Bar(x=by_kind.index, y=by_kind[kind], name=kind, marker_color=color,
                                   marker_line=dict(color=PAPER_BG, width=1),
                                   hovertemplate=f"{kind} %{{y:.1f}} h<extra></extra>"))
    k_fig.update_layout(**base_layout("Weekly training hours by type"), barmode="stack", bargap=0.2)

    te = acts.dropna(subset=["te_aer", "te_ana"])
    te_fig = go.Figure()
    for kind, color in zip(["Strength", "Running", "Other"], SERIES):
        d = te[te["kind"] == kind]
        if not d.empty:
            te_fig.add_trace(go.Scatter(x=d["te_aer"], y=d["te_ana"], mode="markers", name=kind,
                                        marker=dict(color=color, size=9, opacity=0.7, line=dict(color=PAPER_BG, width=2)),
                                        customdata=np.c_[d["start"].dt.strftime("%d %b"), d["name"]],
                                        hovertemplate="%{customdata[0]} %{customdata[1]}<br>Aerobic %{x:.1f} · Anaerobic %{y:.1f}<extra></extra>"))
    lay = base_layout("Training effect per session")
    lay.update(hovermode="closest", xaxis=dict(title="Aerobic TE", range=[0, 5.2], gridcolor=GRID_COLOR),
               yaxis=dict(title="Anaerobic TE", range=[0, 5.2], gridcolor=GRID_COLOR))
    te_fig.update_layout(**lay)
    render_two_col([("load_ac", fig, pd.DataFrame({"acute": acute, "chronic": chronic, "acwr": acwr})),
                    ("load_acwr", r_fig), ("load_kind", k_fig, by_kind), ("load_te", te_fig)])

# ── 7. Data ──────────────────────────────────────────────────────────────────────
with tabs[6]:
    for title, df in [("Daily wellness", daily), ("Activities", acts), ("Strength sets", sets), ("Weigh-ins", weight)]:
        section(title)
        st.dataframe(df, width="stretch", height=260)
        csv_download(df, title.lower().replace(" ", "_"))

# ── 8. Guide ─────────────────────────────────────────────────────────────────────
with tabs[7]:
    st.markdown("### What everything means")
    st.caption("Every metric on this dashboard in plain language. Garmin's numbers are estimates from your watch's "
               "sensors — great for spotting trends in yourself, not medical measurements.")
    groups = {
        "Reading the dashboard": ["trend_delta", "baseline"],
        "Recovery & sleep": ["hrv", "rhr", "sleep_score", "sleep_stages", "body_battery", "readiness", "stress",
                             "recovery_scatter"],
        "Training load": ["training_load", "acute_chronic", "acwr", "training_effect"],
        "Strength": ["sets_per_muscle", "volume", "e1rm", "best_set"],
        "Running & fitness": ["pace", "efficiency", "vo2max", "intensity_minutes"],
    }
    for g_title, keys in groups.items():
        section(g_title)
        for k in keys:
            title, text = GLOSSARY[k]
            st.markdown(f"**{title}** — {text}")
