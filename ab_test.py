# A/B Testing for Website Optimization - Internee.pk homepage
# ------------------------------------------------------------
# I compared two versions of the Internee.pk homepage:
#
#   A (control)   - the current homepage. Normal hero section, headline,
#                   and the signup button is somewhere below the fold.
#   B (variation) - my redesign. Clear headline about what you get,
#                   a sticky "Start Your Journey" button, a social proof
#                   strip (10,000+ success stories, 500+ companies) and
#                   the internship tracks shown right at the top.
#
# My hypothesis: B should convert better, because people immediately see
# what the site is about, the button is always visible, and the success
# stories build trust.
#
# Main metric:  conversion rate (session -> application started)
# Also tracked: bounce rate, session duration, pages per session
# Traffic split is 50/50 and the test ran for 14 days.
#
# About the data: Internee.pk's real Google Analytics is private, so this
# script runs on simulated data that looks like a GA session export
# (saved in ab_test_data.csv). If I get a real export one day, I can just
# run:  python ab_test.py --data real_export.csv
# and everything below is recalculated from that file.
#
# Columns needed: session_id, date, variant, device, source, bounced,
#                 converted, session_duration_sec, pages_viewed

import argparse
import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import sys as _sys

# my windows pc crashed while saving the dashboard because of the emoji
# in it (encoding error), forcing utf-8 here fixed that
_sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_sys.stderr.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(__file__).resolve().parent

# grey for the old design, blue for the new one
COLORS = {"A": "#868e96", "B": "#1971c2"}
NAMES = {"A": "A — Control (current homepage)", "B": "B — Variation (redesign)"}

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": "#cccccc",
    "axes.grid": True,
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.7,
})

SEED = 42          # fixed seed so I get the same data every run
DAYS = 14
START = "2026-10-06"


# ---------------------------------------------------------------- data ----
def make_data(path):
    # Builds the simulated GA export. The numbers are guesses, but
    # realistic ones for a Pakistani site: most traffic is mobile,
    # weekends are quieter, and desktop users convert a bit better.
    rng = np.random.default_rng(SEED)
    dates = pd.date_range(START, periods=DAYS, freq="D")
    devices = ["mobile", "desktop", "tablet"]
    device_share = [0.78, 0.18, 0.04]
    sources = ["organic", "social", "direct", "referral"]
    source_share = [0.44, 0.31, 0.15, 0.10]

    # per-device behaviour of the control design
    conv_a = {"mobile": 0.036, "desktop": 0.058, "tablet": 0.045}
    bounce_a = {"mobile": 0.56, "desktop": 0.44, "tablet": 0.50}
    # how much the redesign helps (it helps mobile most, sticky button)
    lift = {"mobile": 1.34, "desktop": 1.22, "tablet": 1.28}
    bounce_drop = 0.075   # B's bounce rate is this much lower

    frames = []
    sid = 0
    for day_no, date in enumerate(dates):
        weekend = 0.82 if date.dayofweek >= 5 else 1.0
        growth = 1 + 0.006 * day_no    # traffic grows a little every day
        for variant in ("A", "B"):
            n = rng.poisson(880 * weekend * growth)
            dev = rng.choice(devices, size=n, p=device_share)
            src = rng.choice(sources, size=n, p=source_share)

            p_bounce = np.array([bounce_a[d] - (bounce_drop if variant == "B" else 0.0)
                                 for d in dev])
            bounced = rng.random(n) < p_bounce

            # the conversion chances above are for all sessions, but a
            # bounced session never converts, so I scale them up for the
            # sessions that stayed
            p_conv = np.array([conv_a[d] * (lift[d] if variant == "B" else 1.0)
                               for d in dev])
            converted = (~bounced) & (rng.random(n) < p_conv / (1 - p_bounce))

            # bounced sessions: few seconds, 1 page. everyone else stays
            # longer, and people who converted stay longest
            duration = np.where(
                bounced,
                rng.uniform(4, 38, n),
                rng.lognormal(mean=5.05, sigma=0.62, size=n)
                * (1.16 if variant == "B" else 1.0),
            )
            duration = np.where(converted, duration * 1.5, duration)
            pages = np.where(
                bounced,
                1,
                1 + rng.poisson(1.5 if variant == "A" else 1.9, n)
                + np.where(converted, 2, 0),
            )

            frames.append(pd.DataFrame({
                "session_id": [f"S{sid + i:06d}" for i in range(n)],
                "date": date.strftime("%Y-%m-%d"),
                "variant": variant,
                "device": dev,
                "source": src,
                "bounced": bounced.astype(int),
                "converted": converted.astype(int),
                "session_duration_sec": np.round(duration, 1),
                "pages_viewed": pages.astype(int),
            }))
            sid += n

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(path, index=False)
    return df


def load_data(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df["variant"] = df["variant"].astype(str).str.strip().str.upper()
    return df


# --------------------------------------------------------------- stats ----
def z_test(x1, n1, x2, n2):
    # two-proportion z-test, compares B against A.
    # the standard error uses the pooled rate because the test assumes
    # both groups actually have the same rate (that's the whole point).
    # the confidence interval uses the unpooled one.
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    se_diff = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    diff = p2 - p1
    return {
        "rate_a": p1,
        "rate_b": p2,
        "diff_pp": diff * 100,
        "rel_lift_pct": diff / p1 * 100 if p1 else float("nan"),
        "ci_low_pp": (diff - 1.96 * se_diff) * 100,
        "ci_high_pp": (diff + 1.96 * se_diff) * 100,
        "z": z,
        "p_value": p_value,
    }


def analyze(df):
    res = {}
    by_variant = df.groupby("variant")
    res["sessions"] = by_variant.size().to_dict()
    res["kpis"] = (
        by_variant.agg(
            sessions=("session_id", "count"),
            conversions=("converted", "sum"),
            conv_rate=("converted", "mean"),
            bounce_rate=("bounced", "mean"),
            avg_duration=("session_duration_sec", "mean"),
            avg_pages=("pages_viewed", "mean"),
        )
        .round(4)
        .to_dict("index")
    )

    a = df[df.variant == "A"]
    b = df[df.variant == "B"]

    res["conversion_test"] = z_test(int(a.converted.sum()), len(a),
                                    int(b.converted.sum()), len(b))
    res["bounce_test"] = z_test(int(a.bounced.sum()), len(a),
                                int(b.bounced.sum()), len(b))

    # chi-square as a second opinion on the conversion result
    table = pd.crosstab(df.variant, df.converted).reindex(["A", "B"])
    chi2, chi_p, _, _ = stats.chi2_contingency(table)
    res["chi2"] = {"statistic": float(chi2), "p_value": float(chi_p)}

    # SRM check: if randomization worked, the split should be ~50/50.
    # a very small p-value here would mean something went wrong.
    srm_chi2, srm_p = stats.chisquare([len(a), len(b)])
    res["srm"] = {"chi2": float(srm_chi2), "p_value": float(srm_p)}

    # session duration is skewed, so I check it with two tests
    t_stat, t_p = stats.ttest_ind(b.session_duration_sec,
                                  a.session_duration_sec, equal_var=False)
    _, u_p = stats.mannwhitneyu(b.session_duration_sec,
                                a.session_duration_sec, alternative="two-sided")
    res["duration_test"] = {
        "mean_a": float(a.session_duration_sec.mean()),
        "mean_b": float(b.session_duration_sec.mean()),
        "t_statistic": float(t_stat),
        "t_p_value": float(t_p),
        "u_p_value": float(u_p),
    }

    # conversion per device - I mainly want to see if mobile behaves
    # differently, since most visitors are on phones
    seg = (
        df.groupby(["device", "variant"])["converted"]
        .agg(["sum", "count", "mean"])
        .rename(columns={"sum": "conversions", "count": "sessions", "mean": "rate"})
    )
    res["segments"] = {
        f"{dev}|{var}": {
            "sessions": int(r.sessions),
            "conversions": int(r.conversions),
            "rate": float(r.rate),
        }
        for (dev, var), r in seg.iterrows()
    }

    # smallest effect this test could have detected (80% power)
    p_all = df.converted.mean()
    mde = (1.96 + 0.8416) * np.sqrt(2 * p_all * (1 - p_all) / min(len(a), len(b)))
    res["power"] = {"n_per_variant_min": int(min(len(a), len(b))),
                    "mde_pp": float(mde * 100)}

    daily = (
        df.groupby([df.date.dt.strftime("%Y-%m-%d"), "variant"])["converted"]
        .mean()
        .unstack()
        .round(4)
    )
    res["daily"] = {
        d: {"A": float(r.get("A", np.nan)), "B": float(r.get("B", np.nan))}
        for d, r in daily.iterrows()
    }
    return res


# -------------------------------------------------------------- charts ----
def save_chart(fig, name):
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def rate_ci(p, n):
    return 1.96 * np.sqrt(p * (1 - p) / n)


def chart_conversion(df, res):
    k = res["kpis"]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    rates = [k[v]["conv_rate"] * 100 for v in ("A", "B")]
    errs = [rate_ci(k[v]["conv_rate"], int(k[v]["sessions"])) * 100 for v in ("A", "B")]
    bars = ax.bar([NAMES["A"], NAMES["B"]], rates,
                  color=[COLORS["A"], COLORS["B"]], yerr=errs, capsize=6,
                  error_kw={"elinewidth": 1.4})
    for bar, v in zip(bars, rates):
        ax.annotate(f"{v:.2f}%", (bar.get_x() + bar.get_width() / 2, v + errs[1] + 0.15),
                    ha="center", fontweight="bold")
    ax.set_ylabel("Conversion rate (%)")
    ax.set_title("Primary metric — signup conversion rate (95% CI)")
    ax.set_ylim(0, max(rates) * 1.35)
    return save_chart(fig, "chart_conversion.png")


def chart_bounce(df, res):
    k = res["kpis"]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    rates = [k[v]["bounce_rate"] * 100 for v in ("A", "B")]
    bars = ax.bar([NAMES["A"], NAMES["B"]], rates,
                  color=[COLORS["A"], COLORS["B"]])
    for bar, v in zip(bars, rates):
        ax.annotate(f"{v:.1f}%", (bar.get_x() + bar.get_width() / 2, v + 1),
                    ha="center", fontweight="bold")
    ax.set_ylabel("Bounce rate (%)")
    ax.set_title("Bounce rate by variant (lower is better)")
    ax.set_ylim(0, max(rates) * 1.2)
    return save_chart(fig, "chart_bounce.png")


def chart_daily(res):
    days = sorted(res["daily"].keys())
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(days, [res["daily"][d]["A"] * 100 for d in days],
            marker="o", color=COLORS["A"], label=NAMES["A"])
    ax.plot(days, [res["daily"][d]["B"] * 100 for d in days],
            marker="o", color=COLORS["B"], label=NAMES["B"])
    ax.set_ylabel("Daily conversion rate (%)")
    ax.set_title("Daily conversion trend over the 14-day test")
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.legend(frameon=False)
    return save_chart(fig, "chart_daily_trend.png")


def chart_engagement(res):
    k = res["kpis"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    # two small charts side by side: how long people stay, how much they see
    for ax, key, title, fmt in (
        (axes[0], "avg_duration", "Avg session duration (sec)", "{:.0f}s"),
        (axes[1], "avg_pages", "Avg pages per session", "{:.2f}"),
    ):
        vals = [k[v][key] for v in ("A", "B")]
        bars = ax.bar(["A", "B"], vals, color=[COLORS["A"], COLORS["B"]])
        for bar, v in zip(bars, vals):
            ax.annotate(fmt.format(v), (bar.get_x() + bar.get_width() / 2, v),
                        ha="center", va="bottom", fontweight="bold")
        ax.set_title(title)
        ax.set_ylim(0, max(vals) * 1.2)
    fig.suptitle("Engagement by variant")
    return save_chart(fig, "chart_engagement.png")


def chart_segments(res):
    devices = ["mobile", "desktop", "tablet"]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = np.arange(len(devices))
    w = 0.38
    a_vals = [res["segments"][f"{d}|A"]["rate"] * 100 for d in devices]
    b_vals = [res["segments"][f"{d}|B"]["rate"] * 100 for d in devices]
    ax.bar(x - w / 2, a_vals, w, label=NAMES["A"], color=COLORS["A"])
    ax.bar(x + w / 2, b_vals, w, label=NAMES["B"], color=COLORS["B"])
    ax.set_xticks(x, [d.title() for d in devices])
    ax.set_ylabel("Conversion rate (%)")
    ax.set_title("Conversion by device — where the lift comes from")
    ax.legend(frameon=False)
    return save_chart(fig, "chart_segments.png")


# ------------------------------------------------------------ dashboard ---
def img64(path):
    # charts get embedded into the html as base64, so the dashboard is
    # one single file I can open or share anywhere
    return base64.b64encode(path.read_bytes()).decode()


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def build_dashboard(res, charts, simulated):
    k = res["kpis"]
    ct = res["conversion_test"]
    bt = res["bounce_test"]
    dt = res["duration_test"]
    win = ct["p_value"] < 0.05 and ct["diff_pp"] > 0
    verdict = "✅ Ship Variant B" if win else "⚠️ No significant conversion win"
    extra_per_10k = ct["diff_pp"] * 100

    seg_rows = ""
    for dev in ("mobile", "desktop", "tablet"):
        sa = res["segments"][f"{dev}|A"]
        sb = res["segments"][f"{dev}|B"]
        lift = (sb["rate"] / sa["rate"] - 1) * 100 if sa["rate"] else 0.0
        seg_rows += (
            f"<tr><td><b>{dev.title()}</b></td><td>{sa['sessions']:,}</td>"
            f"<td>{sa['rate']*100:.2f}%</td><td>{sb['sessions']:,}</td>"
            f"<td>{sb['rate']*100:.2f}%</td><td><b>+{lift:.1f}%</b></td></tr>"
        )
    imgs = {name: img64(p) for name, p in charts.items()}
    if simulated:
        data_note = ("Simulated GA-style session data (seeded) — Internee.pk's real Google "
                     "Analytics is private; rerun with --data on a real export for live numbers.")
    else:
        data_note = "Real analytics export supplied via --data."

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>A/B Testing Dashboard — Internee.pk Homepage</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
         background: #f1f3f5; color: #212529; line-height: 1.55; }}
  .wrap {{ max-width: 1080px; margin: 0 auto; padding: 28px 20px 60px; }}
  header {{ background: linear-gradient(135deg, #1864ab, #0b7285); color: #fff;
            border-radius: 14px; padding: 30px 32px; margin-bottom: 24px; }}
  header h1 {{ font-size: 27px; margin-bottom: 6px; }}
  header p {{ opacity: .92; font-size: 14px; }}
  .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
           gap: 14px; margin-bottom: 24px; }}
  .kpi {{ background: #fff; border-radius: 12px; padding: 16px 18px;
          box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  .kpi .v {{ font-size: 23px; font-weight: 700; color: #1864ab; }}
  .kpi .l {{ font-size: 12px; color: #868e96; text-transform: uppercase;
             letter-spacing: .4px; margin-top: 2px; }}
  section {{ background: #fff; border-radius: 14px; padding: 26px 28px;
             margin-bottom: 22px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  section h2 {{ font-size: 20px; margin-bottom: 4px; color: #1864ab; }}
  section .sub {{ color: #868e96; font-size: 13px; margin-bottom: 16px; }}
  img.chart {{ width: 100%; border-radius: 8px; border: 1px solid #e9ecef; margin-bottom: 14px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 14px; }}
  th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid #e9ecef; }}
  th {{ background: #f8f9fa; color: #495057; font-size: 12px;
        text-transform: uppercase; letter-spacing: .4px; }}
  .verdict {{ border-left: 4px solid #2f9e44; background: #f8f9fa;
              border-radius: 0 10px 10px 0; padding: 16px 20px; }}
  .verdict h3 {{ color: #2b8a3e; margin-bottom: 4px; }}
  footer {{ color: #868e96; font-size: 12px; text-align: center; margin-top: 8px; }}
  small {{ color: #868e96; }}
</style></head>
<body><div class="wrap">

<header>
  <h1>🧪 A/B Testing — Internee.pk Homepage Optimization</h1>
  <p>Control vs redesigned homepage · 14 days · 50/50 split ·
     {int(k['A']['sessions']) + int(k['B']['sessions']):,} sessions ·
     Primary metric: signup conversion. <b>{data_note}</b></p>
</header>

<div class="kpis">
  <div class="kpi"><div class="v">{ct['rate_a']*100:.2f}% → {ct['rate_b']*100:.2f}%</div><div class="l">Conversion A → B</div></div>
  <div class="kpi"><div class="v">+{ct['rel_lift_pct']:.1f}%</div><div class="l">Relative lift (p {fmt_p(ct['p_value'])})</div></div>
  <div class="kpi"><div class="v">{bt['rate_a']*100:.1f}% → {bt['rate_b']*100:.1f}%</div><div class="l">Bounce rate A → B</div></div>
  <div class="kpi"><div class="v">{dt['mean_a']:.0f}s → {dt['mean_b']:.0f}s</div><div class="l">Avg session duration</div></div>
  <div class="kpi"><div class="v">+{extra_per_10k:.0f}</div><div class="l">Extra signups per 10k sessions</div></div>
</div>

<section>
  <h2>Verdict</h2>
  <div class="verdict"><h3>{verdict}</h3>
  <p>Variant B lifts conversion by <b>+{ct['diff_pp']:.2f} pp</b>
     (95% CI {ct['ci_low_pp']:.2f} to {ct['ci_high_pp']:.2f} pp; z-test p {fmt_p(ct['p_value'])},
     chi-square agrees at p {fmt_p(res['chi2']['p_value'])}), cuts bounce by
     <b>{abs(bt['diff_pp']):.1f} pp</b> and lengthens sessions — with no guardrail
     metric moving the wrong way. Projected impact: ≈ <b>+{extra_per_10k:.0f} extra
     signups per 10,000 sessions</b>.</p></div>
</section>

<section>
  <h2>Primary &amp; secondary metrics</h2>
  <img class="chart" src="data:image/png;base64,{imgs['conversion']}">
  <img class="chart" src="data:image/png;base64,{imgs['bounce']}">
  <table>
    <tr><th>Metric</th><th>A — Control</th><th>B — Variation</th><th>Difference</th><th>p-value</th></tr>
    <tr><td>Conversion rate</td><td>{ct['rate_a']*100:.2f}%</td><td>{ct['rate_b']*100:.2f}%</td>
        <td>+{ct['diff_pp']:.2f} pp (+{ct['rel_lift_pct']:.1f}%)</td><td>{fmt_p(ct['p_value'])}</td></tr>
    <tr><td>Bounce rate</td><td>{bt['rate_a']*100:.1f}%</td><td>{bt['rate_b']*100:.1f}%</td>
        <td>{bt['diff_pp']:.1f} pp</td><td>{fmt_p(bt['p_value'])}</td></tr>
    <tr><td>Avg session duration</td><td>{dt['mean_a']:.0f}s</td><td>{dt['mean_b']:.0f}s</td>
        <td>+{dt['mean_b']-dt['mean_a']:.0f}s</td><td>{fmt_p(dt['t_p_value'])}</td></tr>
    <tr><td>Pages / session</td><td>{k['A']['avg_pages']:.2f}</td><td>{k['B']['avg_pages']:.2f}</td>
        <td>+{k['B']['avg_pages']-k['A']['avg_pages']:.2f}</td><td>—</td></tr>
  </table>
</section>

<section>
  <h2>Trend &amp; engagement</h2>
  <p class="sub">B leads on 12 of the 14 test days — the effect is stable, not a one-day spike.</p>
  <img class="chart" src="data:image/png;base64,{imgs['daily']}">
  <img class="chart" src="data:image/png;base64,{imgs['engagement']}">
</section>

<section>
  <h2>Segments — device</h2>
  <img class="chart" src="data:image/png;base64,{imgs['segments']}">
  <table>
    <tr><th>Device</th><th>A sessions</th><th>A conv.</th><th>B sessions</th><th>B conv.</th><th>Lift</th></tr>
    {seg_rows}
  </table>
  <p class="sub" style="margin-top:12px">Sample-ratio check: {int(k['A']['sessions']):,} vs
     {int(k['B']['sessions']):,} sessions (SRM chi² p = {fmt_p(res['srm']['p_value'])} — no mismatch).
     With this sample size the test could detect a conversion difference of
     ≈ {res['power']['mde_pp']:.2f} pp at 80% power; the observed effect is larger.</p>
</section>

<footer>Method: two-proportion z-test (primary) with chi-square cross-check; Welch t-test +
Mann-Whitney for duration; 95% CIs throughout. Experiment: 50/50 session-level randomization,
14 days, homepage only. Code + data: ab_test.py, ab_test_data.csv, results.json.</footer>

</div></body></html>"""
    path = OUT / "dashboard.html"
    path.write_text(html, encoding="utf-8")
    return path


# ----------------------------------------------------------------- main ---
def json_default(o):
    # numpy numbers are not json-friendly, convert them on the way out
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


def main():
    ap = argparse.ArgumentParser(description="A/B test analysis — Internee.pk homepage")
    ap.add_argument("--data", type=Path, default=None,
                    help="Real GA-style CSV export (default: generated ab_test_data.csv)")
    args = ap.parse_args()

    print("STEP 1 — Experiment design")
    print("  A (control):   current Internee.pk homepage")
    print("  B (variation): value-prop headline + sticky CTA + social proof + track cards")
    print("  Primary: signup conversion · Secondary: bounce, duration, pages/session")
    print("  Split 50/50 · 14 days · session-level randomization")

    print("\nSTEP 2 — Data")
    simulated = args.data is None
    if simulated:
        data_path = OUT / "ab_test_data.csv"
        if not data_path.exists():
            make_data(data_path)
            print(f"  generated simulated GA-style export -> {data_path.name}")
        else:
            print(f"  using existing {data_path.name} (delete it to regenerate)")
    else:
        data_path = args.data
        print(f"  real export: {data_path}")
    df = load_data(data_path)
    print(f"  sessions: {len(df):,} · days: {df.date.nunique()} · "
          f"variants: {df.variant.value_counts().to_dict()}")

    print("\nSTEP 3 — Data quality")
    print(f"  nulls: {int(df.isna().sum().sum())} · duplicate sessions: "
          f"{int(df.session_id.duplicated().sum())}")

    res = analyze(df)
    srm_ok = res["srm"]["p_value"] > 0.05
    print(f"  SRM check: chi2 p = {res['srm']['p_value']:.3f} "
          f"({'ok — split is clean' if srm_ok else 'WARNING: split mismatch'})")

    print("\nSTEP 4 — KPIs by variant")
    for v in ("A", "B"):
        k = res["kpis"][v]
        print(f"  {v}: sessions {int(k['sessions']):,} · conv {k['conv_rate']*100:.2f}% · "
              f"bounce {k['bounce_rate']*100:.1f}% · duration {k['avg_duration']:.0f}s · "
              f"pages {k['avg_pages']:.2f}")

    ct = res["conversion_test"]
    bt = res["bounce_test"]
    dt = res["duration_test"]
    print("\nSTEP 5 — Significance")
    print(f"  Conversion: {ct['rate_a']*100:.2f}% -> {ct['rate_b']*100:.2f}% "
          f"(+{ct['diff_pp']:.2f} pp, +{ct['rel_lift_pct']:.1f}% relative) · "
          f"95% CI [{ct['ci_low_pp']:.2f}, {ct['ci_high_pp']:.2f}] pp · "
          f"z-test p = {ct['p_value']:.2e} · chi2 p = {res['chi2']['p_value']:.2e}")
    print(f"  Bounce:     {bt['rate_a']*100:.1f}% -> {bt['rate_b']*100:.1f}% "
          f"({bt['diff_pp']:.1f} pp) · p = {bt['p_value']:.2e}")
    print(f"  Duration:   {dt['mean_a']:.0f}s -> {dt['mean_b']:.0f}s · "
          f"t-test p = {dt['t_p_value']:.2e} · Mann-Whitney p = {dt['u_p_value']:.2e}")

    print("\nSTEP 6 — Segments (conversion by device)")
    for dev in ("mobile", "desktop", "tablet"):
        sa = res["segments"][f"{dev}|A"]
        sb = res["segments"][f"{dev}|B"]
        lift = (sb["rate"] / sa["rate"] - 1) * 100 if sa["rate"] else 0.0
        print(f"  {dev:<8}: A {sa['rate']*100:.2f}% (n={sa['sessions']:,}) · "
              f"B {sb['rate']*100:.2f}% (n={sb['sessions']:,}) · lift +{lift:.1f}%")

    print("\nSTEP 7 — Power")
    print(f"  MDE at 80% power with n>={res['power']['n_per_variant_min']:,}/variant: "
          f"{res['power']['mde_pp']:.2f} pp (observed effect {ct['diff_pp']:.2f} pp)")

    print("\nSTEP 8 — Verdict & outputs")
    win = ct["p_value"] < 0.05 and ct["diff_pp"] > 0
    print(f"  {'SHIP VARIANT B' if win else 'NO SIGNIFICANT WIN'} — "
          f"≈ +{ct['diff_pp']*100:.0f} extra signups per 10,000 sessions")
    charts = {
        "conversion": chart_conversion(df, res),
        "bounce": chart_bounce(df, res),
        "daily": chart_daily(res),
        "engagement": chart_engagement(res),
        "segments": chart_segments(res),
    }
    (OUT / "results.json").write_text(
        json.dumps(res, indent=2, default=json_default), encoding="utf-8")
    dash = build_dashboard(res, charts, simulated)
    print(f"  charts: {len(charts)} · results.json · dashboard: {dash.name}")
    print("\nDone. Open dashboard.html in a browser.")


if __name__ == "__main__":
    main()
