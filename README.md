# A/B Testing for Website Optimization — Internee.pk Homepage

For this task I designed and analyzed an A/B test for the **Internee.pk** homepage — Pakistan's virtual internship platform. The question was simple: does a redesigned homepage actually get more students to sign up, or does it just *look* better? So I tested it properly instead of guessing.

Everything in this repo comes from one script, `ab_test.py`, which I wrote to walk through the whole analysis step by step — the same 8 steps it prints when you run it. The charts, the dashboard and every number below are its output (they're all in this folder: `ab_test_data.csv`, the five `chart_*.png` files, `dashboard.html` and `results.json`).

## The two designs

**A — Control:** the current homepage. Standard hero section, a general headline, and the signup button sits below the fold, so you have to scroll before you can do anything.

**B — My redesign:** a headline that says clearly what you get, a sticky "Start Your Journey" button that's always on screen, a strip with social proof (10,000+ success stories, 500+ partner companies), and the internship tracks shown right at the top.

My thinking was that B would win because visitors understand the site in the first few seconds, the button never disappears, and the success stories make people trust it enough to apply.

## How I ran the test

- 50/50 traffic split, assigned per session, for 14 days
- **Main metric:** conversion rate (session → internship application started)
- **Also tracked:** bounce rate, session duration, pages per session
- 24,567 sessions in total (12,201 on A, 12,366 on B) — all of them in `ab_test_data.csv`

One honest note about the data: Internee.pk's real Google Analytics is private, so I worked with a simulated dataset that looks like a GA session export — same columns, realistic values (about 78% mobile traffic, quieter weekends, desktop converting a bit better). The script doesn't care where the data came from though: `python ab_test.py --data real_export.csv` reruns the whole analysis on a real export.

## Results

These are the numbers from my run, saved in `results.json` and shown visually in `dashboard.html`:

| Metric | A — Control | B — Redesign | What happened |
|---|---|---|---|
| Conversion rate | 3.98% | **5.10%** | **+28.1% relative lift**, p < 0.001 |
| Bounce rate | 53.9% | 46.3% | 7.7 points lower |
| Avg. session duration | 103s | 132s | people stayed ~30s longer |
| Pages per session | 1.77 | 2.12 | people explored more |

The conversion difference is +1.12 percentage points, with a 95% confidence interval of +0.60 to +1.64 — so even the pessimistic end of the interval is a solid win. I cross-checked the z-test with a chi-square test and they agree.

Each chart tells one part of the story: `chart_conversion.png` is the main result (with the confidence intervals as error bars), `chart_bounce.png` the bounce drop, `chart_daily_trend.png` shows B in front on 12 of the 14 days — so it wasn't a one-day spike or a novelty effect — `chart_engagement.png` covers duration and pages, and `chart_segments.png` breaks conversion down by device. On devices: mobile (most of the traffic) improved +28.6% and desktop +31.1%. Tablet barely moved, but there were too few tablet sessions to conclude anything from that.

## My verdict

**Ship design B.** At this traffic level it works out to roughly **+112 extra signups for every 10,000 sessions** — about +590 extra applications a month. The redesign doesn't just look better, it measurably works better, and no metric moved in the wrong direction.

## Requirements

The project needs four Python libraries, and they're all listed in **`requirements.txt`** — that's the file pip reads to install everything a project needs in one go, so anyone who downloads this repo can recreate the same setup instead of installing packages one by one and guessing versions.

Install them like this (from the project folder):

```bash
python -m pip install -r requirements.txt
```

What each library actually does in this project:

- **pandas** — loads the session data and does all the grouping (per variant, per device, per day)
- **numpy** — the number work underneath: generating the simulated data and the formulas in between
- **matplotlib** — draws the five charts
- **scipy** — the statistical tests (z-test, chi-square, t-test, Mann-Whitney)

Nothing else is needed — the rest is Python's built-in library. (The script also sets its file output to UTF-8 itself, after my Windows PC crashed saving the dashboard over an emoji in it.)

## Run it yourself

```bash
python ab_test.py
```

It prints the analysis step by step — design → data → quality checks → KPIs → significance tests → device segments → power → verdict — and writes `results.json`, the five charts and `dashboard.html`. It also checks its own homework along the way: there's a sample-ratio check confirming the 50/50 split really was 50/50, and a power calculation showing the test could detect effects of about 0.75 points — smaller than the 1.12 points actually observed, so the sample was big enough to trust the result.

## Files in this project

- `ab_test.py` — the whole pipeline, one file, comments explain my reasoning as it goes
- `ab_test_data.csv` — the session-level dataset (simulated GA-style export)
- `results.json` — every statistic the script computed
- `dashboard.html` — a visual summary of the results, open it in any browser
- `chart_conversion.png`, `chart_bounce.png`, `chart_daily_trend.png`, `chart_engagement.png`, `chart_segments.png` — the five charts
- `requirements.txt` — the libraries needed, installable in one command (see above)
- `REPORT.md` — the full written analysis with next-step recommendations
- `README.md` — this file

## What I'd test next

Now that B is the new baseline: the CTA button text ("Start Your Journey" vs "Find an Internship"), and the application form length — that's where the next drop-offs probably are.

The biggest thing I took from this task: a design opinion becomes a decision the moment you attach a confidence interval to it.
