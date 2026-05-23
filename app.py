"""
SolarSquare 2-Install Vendor Solver — v34 (FAST)

Speed fixes from v33:
- NO slab decision variables in CP-SAT (slabs assigned post-hoc)
- CP-SAT only decides activity per vendor per day
- stop_after_first_solution=True (no proving optimality)
- Per-iteration time limit (~5s each)
- Profitability checked post-CP-SAT

Install:  pip install flask ortools
Run:      python app.py
"""

from flask import Flask, request, jsonify, send_from_directory, Response
import os
import math
import traceback as tb_mod
import io
import datetime

# Cluster data — all 28 clusters with P50/P75/P90 daily distributions and slab mix
from clusters import CLUSTER_DATA

try:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image, KeepTogether
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    REPORTLAB_OK = True
    # Register a font that supports ₹ symbol. DejaVu Sans is bundled with most systems.
    DEJAVU_PATHS = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',  # Linux
        '/Library/Fonts/DejaVuSans.ttf',  # Mac (if installed via Homebrew)
        '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',  # Mac built-in (has ₹)
        '/Library/Fonts/Arial Unicode.ttf',  # Older Mac
    ]
    DEJAVU_BOLD_PATHS = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/Library/Fonts/DejaVuSans-Bold.ttf',
    ]
    FONT_REGULAR = 'Helvetica'
    FONT_BOLD = 'Helvetica-Bold'
    for path in DEJAVU_PATHS:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont('PDFFont', path))
                FONT_REGULAR = 'PDFFont'
                print(f'[pdf] Using font: {path}')
                # Try bold too
                for bpath in DEJAVU_BOLD_PATHS:
                    if os.path.exists(bpath):
                        pdfmetrics.registerFont(TTFont('PDFFont-Bold', bpath))
                        FONT_BOLD = 'PDFFont-Bold'
                        break
                else:
                    FONT_BOLD = 'PDFFont'  # fall back to regular if no bold
                break
            except Exception as e:
                print(f'[pdf] Could not register {path}: {e}')
    else:
        print('[pdf] No Unicode font found. ₹ symbol may not render. Run: brew install --cask font-dejavu')
except ImportError:
    REPORTLAB_OK = False
    FONT_REGULAR = 'Helvetica'
    FONT_BOLD = 'Helvetica-Bold'
    print('[warning] reportlab not installed. PDF reports will not work.')
    print('[warning] Install with: pip3 install reportlab --break-system-packages')
import random

app = Flask(__name__)

try:
    from ortools.sat.python import cp_model
    ORTOOLS_OK = True
    ORTOOLS_ERR = None
except Exception as e:
    ORTOOLS_OK = False
    ORTOOLS_ERR = f'{type(e).__name__}: {e}'


# Pune historical install patterns - P90 distribution derived from 11 real months
# (Jun 2025 - Apr 2026, 3,031 total installs).
#
# Method: for each day-of-month (1-30), compute what % of that month's total
# installs landed on that day. Across 11 months, we have 11 percentages per day.
# Take the 90th percentile -- the upper-bound % we should plan for any given day.
# This captures the day-by-day worst case across history (e.g., D14 had occasional
# 6.5% spikes, D30 routinely hits 6%+) instead of smoothing it away.
#
# Each entry stores percentile values per day for 30-day grid. Months with 28-31
# days were resampled to 30-day grid before percentile calc.

# Per-day percentile arrays (each list = % of monthly total on that day-of-month)
# Indices 0-29 represent days 1-30. Use linear interpolation between percentiles.
PUNE_DAY_DISTRIBUTION = {
    # Each value is "% of monthly total installs on this day-of-month, at percentile X"
    # Range across 11 months: P0, P25, P50, P75, P90, P100
    'P0':  [0.00, 1.02, 0.32, 0.00, 0.72, 2.08, 1.30, 0.88, 1.02, 0.72,
            2.25, 1.45, 2.59, 2.71, 2.10, 1.45, 2.54, 0.74, 0.98, 1.33,
            0.00, 0.95, 2.14, 1.24, 1.73, 2.28, 2.03, 2.59, 2.77, 3.55],
    'P25': [1.31, 1.46, 1.55, 1.38, 1.96, 2.47, 2.38, 2.42, 1.76, 2.29,
            3.00, 2.91, 2.93, 3.11, 3.16, 2.51, 2.92, 2.77, 2.05, 2.40,
            3.05, 2.76, 3.25, 2.62, 2.63, 2.87, 3.01, 3.86, 3.62, 4.10],
    'P50': [1.49, 2.07, 1.97, 1.88, 2.77, 3.12, 2.64, 3.21, 3.02, 2.87,
            3.65, 3.35, 3.24, 3.81, 3.70, 3.51, 3.39, 3.36, 3.32, 3.55,
            3.47, 3.11, 3.53, 3.05, 4.32, 3.49, 3.48, 4.36, 4.84, 5.65],
    'P75': [2.50, 2.23, 2.69, 2.97, 3.30, 3.80, 3.36, 3.78, 3.70, 3.95,
            4.28, 3.83, 4.03, 4.28, 4.19, 3.95, 3.64, 4.08, 3.62, 4.27,
            4.03, 4.14, 4.12, 4.60, 4.86, 4.18, 3.93, 4.75, 5.41, 5.85],
    'P90': [3.40, 2.53, 2.90, 4.41, 3.55, 4.35, 4.25, 4.12, 3.93, 4.34,
            4.55, 4.06, 4.25, 6.52, 4.24, 4.50, 4.53, 4.35, 3.82, 4.85,
            4.58, 4.35, 4.81, 4.97, 5.07, 4.59, 4.46, 5.58, 5.66, 6.63],
    'P100': [4.15, 4.50, 3.33, 5.08, 4.44, 4.61, 5.07, 5.47, 5.07, 4.55,
             5.08, 4.14, 4.57, 6.60, 4.71, 5.30, 5.48, 4.88, 5.19, 5.12,
             5.80, 4.71, 5.58, 5.08, 5.41, 5.07, 4.68, 6.52, 6.52, 7.27],
}


def _resample_to_days(arr, target_days):
    """Resample a daily array to target_days, preserving the shape."""
    src_days = len(arr)
    if src_days == target_days:
        return arr[:]
    out = []
    for i in range(target_days):
        src = i / target_days * src_days
        lo = int(src)
        hi = min(lo + 1, src_days - 1)
        frac = src - lo
        v = arr[lo] * (1 - frac) + arr[hi] * frac
        out.append(v)
    return out


def end_skewed_demand(total, days, peak_ratio, skew_pct=100, cluster='Pune'):
    """Demand curve using P-percentile of historical day-of-month install
    percentages, blended with a flat distribution by skew_pct.

    peak_ratio acts as a stress-level selector (which historical percentile
    to use). skew_pct controls how much of that historical shape to apply
    versus a flat distribution.

    peak_ratio:
      <= 1.45: uses P50 (typical month)
      1.45-1.55: uses P75 (slightly stressed)
      >= 1.55: uses P90 (planning-grade stress)

    skew_pct:
      0   = perfectly flat (every day gets total/days sites)
      50  = halfway between flat and historical
      100 = full historical shape

    cluster: which cluster's historical distribution to use. Falls back to Pune
             if the cluster name is unknown.
    """
    if peak_ratio <= 1.001 or skew_pct <= 0.5:
        # Flat distribution (either explicitly requested or skew is ~0)
        base = total // days
        rem = total - base * days
        arr = [base] * days
        for i in range(rem):
            arr[i % days] += 1
        return arr

    # Pick percentile level based on peak_ratio slider value (1.0–6.0)
    # The slider value directly indicates which stop: 1=P0, 2=P25, 3=P50, 4=P75, 5=P90, 6=P100
    # Map peak_ratio (from slider 0-5 * 0.15 + 1.0) to percentile label
    # Slider 0 → 1.00 → P0
    # Slider 1 → 1.15 → P25
    # Slider 2 → 1.30 → P50
    # Slider 3 → 1.45 → P75
    # Slider 4 → 1.60 → P90
    # Slider 5 → 1.75 → P100
    if peak_ratio <= 1.05:
        pct_label = 'P0'
    elif peak_ratio <= 1.20:
        pct_label = 'P25'
    elif peak_ratio <= 1.35:
        pct_label = 'P50'
    elif peak_ratio <= 1.50:
        pct_label = 'P75'
    elif peak_ratio <= 1.65:
        pct_label = 'P90'
    else:
        pct_label = 'P100'

    # Get distribution for the selected cluster, falling back to Pune
    cluster_info = CLUSTER_DATA.get(cluster) or CLUSTER_DATA.get('Pune')
    if cluster_info is None:
        # Last-resort fallback: hardcoded Pune
        base_dist = PUNE_DAY_DISTRIBUTION[pct_label]
    else:
        base_dist = cluster_info[pct_label]  # 30 values, % shares

    # Resample to target days if not 30
    resampled = _resample_to_days(base_dist, days)

    # Normalize the historical shape so percentages sum to 100
    total_pct = sum(resampled)
    if total_pct <= 0:
        return [round(total / days)] * days
    hist_normalized = [v * 100 / total_pct for v in resampled]

    # Flat distribution percentage (each day = 100/days %)
    flat_pct = 100.0 / days

    # Blend: skew=0 → all flat, skew=100 → all historical
    weight = skew_pct / 100.0
    blended = [(1 - weight) * flat_pct + weight * v for v in hist_normalized]

    # Distribute total sites by blended percentage
    distributed = [v / 100 * total for v in blended]
    arr = [round(v) for v in distributed]

    # Fix rounding drift to match total
    diff = total - sum(arr)
    peak_idx = arr.index(max(arr)) if arr else 0
    order = sorted((i for i in range(days) if i != peak_idx),
                   key=lambda i: abs(i - peak_idx))
    if diff < 0:
        order = order[::-1]
    oi = 0
    safety = 0
    while diff != 0 and safety < 20000:
        t = order[oi % len(order)]
        if diff > 0:
            arr[t] += 1
            diff -= 1
        elif arr[t] > 0:
            arr[t] -= 1
            diff += 1
        oi += 1
        safety += 1
    return arr


# Backward-compat alias
def bell_curve_demand(total, days, peak_ratio, skew_pct=100, cluster='Pune'):
    return end_skewed_demand(total, days, peak_ratio, skew_pct, cluster)


def int_distribute(total, props):
    s = sum(props) or 1
    exact = [total * p / s for p in props]
    r = [int(x) for x in exact]
    diff = total - sum(r)
    fracs = sorted([(i, exact[i] - r[i]) for i in range(len(props))],
                   key=lambda t: -t[1])
    for k in range(diff):
        r[fracs[k % len(fracs)][0]] += 1
    return r


def compute_daily_demand(daily, slab_mix, dd_elig_slabs, elig_pct):
    days = len(daily)
    dd_pairs = []
    sd_slabs = []
    for d in range(days):
        slabs = int_distribute(daily[d], slab_mix)
        max_pairs = int(daily[d] * elig_pct // 2)
        pairs = []
        order = [si for si in [1, 0, 2, 3] if dd_elig_slabs[si]]
        for si in order:
            while slabs[si] >= 2 and len(pairs) < max_pairs:
                pairs.append((si, si))
                slabs[si] -= 2
        while len(pairs) < max_pairs:
            avail = [si for si in order if slabs[si] > 0]
            if len(avail) < 2:
                break
            pairs.append((avail[0], avail[1]))
            slabs[avail[0]] -= 1
            slabs[avail[1]] -= 1
        dd_pairs.append(pairs)
        sd_slabs.append(slabs)
    return dd_pairs, sd_slabs


def _poisson_inverse_cdf(lam, p):
    """Return smallest k such that P(X <= k) >= p, where X ~ Poisson(lam).
    Pure-Python implementation, deterministic.
    """
    if lam <= 0:
        return 0
    # P(X=0) = e^-lam. Iteratively compute CDF.
    import math as _m
    pmf = _m.exp(-lam)
    cdf = pmf
    k = 0
    while cdf < p and k < 1000:
        k += 1
        pmf = pmf * lam / k
        cdf += pmf
    return k


def monte_carlo_slips(pair_count, sd_by_day, sl2, sl1, percentile, runs=None, seed=None):
    """Deterministic Poisson-based slip distribution (replaces old Monte Carlo).

    For each day d (>=1), slips that originate from day d-1's work follow a
    Poisson distribution with mean = (2*pair_count[d-1]*sl2 + sd_by_day[d-1]*sl1).
    Per-day Pxx is the inverse Poisson CDF at the requested percentile.

    `runs` and `seed` are kept for backward compatibility but are ignored.
    Output schema matches the original function exactly.
    """
    days = len(pair_count)
    dd_sites = [p * 2 for p in pair_count]
    total_slips = round(sum(dd_sites) * sl2 + sum(sd_by_day) * sl1)

    if total_slips == 0:
        return {'total_slips': 0, 'expected_sl': [0]*days, 'pxx_sl': [0]*days}

    # Expected slips that land on day d come from day d-1's work
    expected_sl = [0.0] * days
    for d in range(days - 1):
        # day d's work generates slips for day d+1
        lam = dd_sites[d] * sl2 + sd_by_day[d] * sl1
        expected_sl[d + 1] = lam

    # Per-day Pxx via Poisson inverse CDF
    pxx_sl = [0] * days
    for d in range(1, days):
        lam = expected_sl[d]
        if lam > 0:
            pxx_sl[d] = _poisson_inverse_cdf(lam, percentile)

    return {'total_slips': total_slips, 'expected_sl': expected_sl, 'pxx_sl': pxx_sl}


def solve_schedule(v2, v1, days, pair_count, sd_count, peak_day, sl_needed_by_day, total_slips_required, max_working_days, min_working_days, time_limit):
    model = cp_model.CpModel()
    total_v = v2 + v1
    is_dd = [[model.NewBoolVar(f'd_{v}_{d}') for d in range(days)] for v in range(total_v)]
    is_sd = [[model.NewBoolVar(f's_{v}_{d}') for d in range(days)] for v in range(total_v)]
    is_sl = [[model.NewBoolVar(f'l_{v}_{d}') for d in range(days)] for v in range(total_v)]

    for v in range(v2, total_v):
        for d in range(days):
            model.Add(is_dd[v][d] == 0)

    for v in range(total_v):
        for d in range(days):
            model.Add(is_dd[v][d] + is_sd[v][d] + is_sl[v][d] <= 1)

    for d in range(days):
        model.Add(sum(is_dd[v][d] for v in range(v2)) == pair_count[d])
        model.Add(sum(is_sd[v][d] for v in range(total_v)) == sd_count[d])
        # SL: at least sl_needed_by_day (Pxx coverage). May exceed if schedule allows.
        model.Add(sum(is_sl[v][d] for v in range(total_v)) >= sl_needed_by_day[d])

    # HARD CONSTRAINT: total monthly SL must be AT LEAST the deterministic slip count.
    # We use >= (not ==) so the solver always finds a feasible schedule.
    # Excess SL beyond the theoretical count is then converted to idle in post-processing.
    model.Add(sum(is_sl[v][d] for v in range(total_v) for d in range(days)) >= total_slips_required)

    for v in range(v2):
        for d in range(days - 1):
            model.Add(is_dd[v][d] + is_dd[v][d + 1] <= 1)

    # Peak day: do NOT force every vendor to work. Spare vendors may idle.
    # (The original strict constraint was infeasible when total_v > peak work)
    # No additional constraint needed beyond the "at most one activity per day" above.

    for v in range(total_v):
        model.Add(is_sl[v][0] == 0)
        for d in range(1, days):
            model.Add(is_sl[v][d] <= is_dd[v][d - 1] + is_sd[v][d - 1])

    # HARD: each vendor max max_working_days working days (DD + SD + SL count as work)
    # AND min min_working_days to force balanced workload distribution.
    for v in range(total_v):
        work_days = sum(is_dd[v][d] + is_sd[v][d] + is_sl[v][d] for d in range(days))
        model.Add(work_days <= max_working_days)
        model.Add(work_days >= min_working_days)

    # BALANCE DD across 2i vendors: each 2i must do at least (avg × 0.85) DD pairs.
    # This prevents one vendor getting 11 DD while another gets 5.
    if v2 > 0:
        total_dd_pairs = sum(pair_count)
        avg_dd_per_v2 = total_dd_pairs / v2
        min_dd_per_v2 = max(0, int(avg_dd_per_v2 * 0.85))
        max_dd_per_v2 = int(-(-total_dd_pairs // v2) + 1)  # ceil + buffer
        for v in range(v2):
            dd_total = sum(is_dd[v][d] for d in range(days))
            model.Add(dd_total >= min_dd_per_v2)
            model.Add(dd_total <= max_dd_per_v2)

    # No SL distribution objective: solver places SL freely based on feasibility only.
    # The hard constraints (next-day-after-work, total = required count) are sufficient.

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = 8
    solver.parameters.random_seed = 42
    # Removed stop_after_first_solution: search for optimal within time_limit
    # (still bounded by the per-call time budget, so won't run forever)

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, solver.StatusName(status)

    DD, SD, SL = 1, 2, 3
    activity = []
    for v in range(total_v):
        row = []
        for d in range(days):
            if solver.Value(is_dd[v][d]) == 1:
                row.append(DD)
            elif solver.Value(is_sd[v][d]) == 1:
                row.append(SD)
            elif solver.Value(is_sl[v][d]) == 1:
                row.append(SL)
            else:
                row.append(0)
        activity.append(row)
    return activity, solver.StatusName(status)


def assign_slabs(activity, v2, days, dd_pairs_per_day, sd_slabs_per_day,
                 slab_rates_2i, slab_rates_1i, dd_discount):
    total_v = len(activity)
    payouts = [0.0] * total_v
    roster = [[None] * days for _ in range(total_v)]
    DD, SD, SL = 1, 2, 3

    # DD pairs (always done by 2i, use 2i rates with discount)
    for d in range(days):
        dd_vendors = [v for v in range(v2) if activity[v][d] == DD]
        dd_vendors.sort(key=lambda v: payouts[v])
        pair_payouts = []
        for k, (s1, s2) in enumerate(dd_pairs_per_day[d]):
            r1, r2 = slab_rates_2i[s1], slab_rates_2i[s2]
            pp = max(r1, r2) + min(r1, r2) * dd_discount
            pair_payouts.append((pp, (s1, s2)))
        pair_payouts.sort(key=lambda x: -x[0])
        for i, v in enumerate(dd_vendors):
            if i < len(pair_payouts):
                pp, (s1, s2) = pair_payouts[i]
                roster[v][d] = ('DD', (s1, s2))
                payouts[v] += pp
            else:
                roster[v][d] = ('DD', (1, 1))

    # SD: 2i uses 2i rates, 1i uses 1i rates. Assign slabs to vendors regardless of type
    # but the payout depends on the vendor's type. Greedy: highest slab to lowest-paid vendor.
    for d in range(days):
        sd_vendors = [v for v in range(total_v) if activity[v][d] == SD]
        sd_vendors.sort(key=lambda v: payouts[v])
        sd_flat = []
        for si in range(4):
            sd_flat.extend([si] * sd_slabs_per_day[d][si])
        # Sort slabs by max rate so highest-paying slab goes first
        sd_flat.sort(key=lambda si: -max(slab_rates_2i[si], slab_rates_1i[si]))
        for i, v in enumerate(sd_vendors):
            if i < len(sd_flat):
                slab = sd_flat[i]
                rate = slab_rates_2i[slab] if v < v2 else slab_rates_1i[slab]
                roster[v][d] = ('SD', slab)
                payouts[v] += rate
            else:
                roster[v][d] = ('SD', 1)

    for d in range(days):
        for v in range(total_v):
            if activity[v][d] == SL:
                roster[v][d] = ('SL', None)
            elif activity[v][d] == 0:
                roster[v][d] = ('idle', None)

    return roster, payouts


def optimize_slabs_cpsat(activity, v2, days, dd_pairs_per_day, sd_slabs_per_day,
                         slab_rates_2i, slab_rates_1i, dd_discount, cost_2i, cost_1i, time_limit, spread_weight=1):
    """Phase B: given fixed activity schedule, find slab assignments that maximize
    minimum vendor profit. Slab decisions are CP-SAT variables here.
    2i vendors use slab_rates_2i, 1i vendors use slab_rates_1i. DD always uses 2i rates."""
    model = cp_model.CpModel()
    total_v = len(activity)
    DD, SD, SL = 1, 2, 3

    # Decision: for each (day, pair-index k), assign to one DD vendor
    # dd_assign[(d, k, v)] = 1 if vendor v gets pair k on day d
    dd_assign = {}
    pair_payouts_per_day = []
    for d in range(days):
        # Vendors doing DD on day d (already decided by activity)
        dd_vendors = [v for v in range(v2) if activity[v][d] == DD]
        pps = []
        for k, (s1, s2) in enumerate(dd_pairs_per_day[d]):
            r1, r2 = slab_rates_2i[s1], slab_rates_2i[s2]
            pps.append(int(max(r1, r2) + min(r1, r2) * dd_discount))
        pair_payouts_per_day.append(pps)
        # Each pair assigned to exactly one vendor; each vendor gets exactly one pair
        # (Equal counts because activity already decided how many DD on day d)
        if len(dd_vendors) != len(pps):
            # Schedule inconsistency — fallback
            continue
        for k in range(len(pps)):
            for v in dd_vendors:
                dd_assign[(d, k, v)] = model.NewBoolVar(f'ddA_{d}_{k}_{v}')
            model.Add(sum(dd_assign[(d, k, v)] for v in dd_vendors) == 1)
        for v in dd_vendors:
            model.Add(sum(dd_assign[(d, k, v)] for k in range(len(pps))) == 1)

    # SD: each SD vendor on day d gets one slab from the day's pool
    sd_assign = {}
    sd_payouts_per_day = []
    for d in range(days):
        sd_vendors = [v for v in range(total_v) if activity[v][d] == SD]
        flat_slabs = []
        for si in range(4):
            flat_slabs.extend([si] * sd_slabs_per_day[d][si])
        sd_payouts_per_day.append(flat_slabs)
        if len(sd_vendors) != len(flat_slabs):
            continue
        for k in range(len(flat_slabs)):
            for v in sd_vendors:
                sd_assign[(d, k, v)] = model.NewBoolVar(f'sdA_{d}_{k}_{v}')
            model.Add(sum(sd_assign[(d, k, v)] for v in sd_vendors) == 1)
        for v in sd_vendors:
            model.Add(sum(sd_assign[(d, k, v)] for k in range(len(flat_slabs))) == 1)

    # Total payout per vendor
    max_pay = 30 * 60000
    total_payout = [model.NewIntVar(0, max_pay, f'tp_{v}') for v in range(total_v)]
    for v in range(total_v):
        terms = []
        for d in range(days):
            if activity[v][d] == DD and v < v2:
                pps = pair_payouts_per_day[d]
                for k in range(len(pps)):
                    if (d, k, v) in dd_assign:
                        terms.append(dd_assign[(d, k, v)] * pps[k])
            elif activity[v][d] == SD:
                flat = sd_payouts_per_day[d]
                # 2i uses 2i rates, 1i uses 1i rates
                rate_card = slab_rates_2i if v < v2 else slab_rates_1i
                for k in range(len(flat)):
                    if (d, k, v) in sd_assign:
                        terms.append(sd_assign[(d, k, v)] * int(rate_card[flat[k]]))
        if terms:
            model.Add(total_payout[v] == sum(terms))
        else:
            model.Add(total_payout[v] == 0)

    # Objective: equalize profits. Two parts:
    # 1) Maximize the minimum profit (lift the floor)
    # 2) Minimize the spread (max - min) for equal distribution
    # Combined: maximize (2 * min_profit) - (spread_weight * spread)
    # spread_weight=1: balanced (default), spread_weight=3-5: heavy equalization (polish phase)
    min_profit = model.NewIntVar(-max_pay, max_pay, 'min_profit')
    max_profit = model.NewIntVar(-max_pay, max_pay, 'max_profit')
    for v in range(v2):
        model.Add(min_profit <= total_payout[v] - int(cost_2i))
        model.Add(max_profit >= total_payout[v] - int(cost_2i))
    for v in range(v2, total_v):
        model.Add(min_profit <= total_payout[v] - int(cost_1i))
        model.Add(max_profit >= total_payout[v] - int(cost_1i))
    spread = model.NewIntVar(0, 2 * max_pay, 'spread')
    model.Add(spread == max_profit - min_profit)
    model.Maximize(2 * min_profit - spread_weight * spread)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = 8
    solver.parameters.random_seed = 42

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, solver.StatusName(status)

    # Build roster with optimized slabs
    roster = [[None] * days for _ in range(total_v)]
    payouts = [0.0] * total_v
    for d in range(days):
        for v in range(total_v):
            if activity[v][d] == DD and v < v2:
                pps = pair_payouts_per_day[d]
                for k in range(len(pps)):
                    if (d, k, v) in dd_assign and solver.Value(dd_assign[(d, k, v)]) == 1:
                        s1, s2 = dd_pairs_per_day[d][k]
                        roster[v][d] = ('DD', (s1, s2))
                        payouts[v] += pps[k]
                        break
                if roster[v][d] is None:
                    roster[v][d] = ('DD', (1, 1))
            elif activity[v][d] == SD:
                flat = sd_payouts_per_day[d]
                rate_card = slab_rates_2i if v < v2 else slab_rates_1i
                for k in range(len(flat)):
                    if (d, k, v) in sd_assign and solver.Value(sd_assign[(d, k, v)]) == 1:
                        slab = flat[k]
                        roster[v][d] = ('SD', slab)
                        payouts[v] += rate_card[slab]
                        break
                if roster[v][d] is None:
                    roster[v][d] = ('SD', 1)
            elif activity[v][d] == SL:
                roster[v][d] = ('SL', None)
            else:
                roster[v][d] = ('idle', None)
    return (roster, payouts), solver.StatusName(status)


def find_optimal_fast(daily, dd_pairs_per_day, sd_slabs_per_day, pair_count, sd_count,
                      peak_day, sl2, sl1, target_pxx, slab_rates_2i, slab_rates_1i, dd_discount,
                      cost_2i, cost_1i, time_limit_sec, max_working_days, deep_mode=False):
    days = len(daily)
    # lb_v2: max of consecutive DD pairs (no consec constraint) and peak DD
    lb_v2 = max((pair_count[d] + pair_count[d + 1] for d in range(days - 1)), default=0)
    lb_v2 = max(lb_v2, pair_count[peak_day])

    # lb_v1_base: at any day d, total_v must >= pair_count[d] + sd_count[d]
    # (peak day will be tightest, but check all days to be safe)
    # min v1 = max over all days of (pair_count[d] + sd_count[d] - lb_v2)
    # but only counts 2i above pair_count[d] as available for SD
    # Actually simpler: total >= max(pair[d] + sd[d]) for all d
    # So min total = max over d. Then min v1 = max(0, min_total - lb_v2)
    min_total_for_capacity = max(pair_count[d] + sd_count[d] for d in range(days))

    # Also: total vendor-days needed = DD + SD + total_slips. Available = total_v × max_working_days.
    target_total_slips = round(sum(p*2 for p in pair_count) * sl2 + sum(sd_count) * sl1)
    total_workday_demand = sum(pair_count) + sum(sd_count) + target_total_slips
    min_total_from_workdays = -(-total_workday_demand // max_working_days)  # ceil division
    min_total = max(min_total_for_capacity, min_total_from_workdays)
    lb_v1_base = max(0, min_total - lb_v2)
    print(f'[solver] Lower bounds: v2>={lb_v2}, v1>={lb_v1_base}, min_total={min_total} (capacity={min_total_for_capacity}, workdays={min_total_from_workdays}, total_slips={target_total_slips})')

    pxx_levels = [target_pxx]
    for p in [0.50, 0.25, 0.10, 0.0]:
        if p < target_pxx and p not in pxx_levels:
            pxx_levels.append(p)

    # Iteration-based caps. Deep mode (pan-India precompute) uses 5x larger caps
    # because it runs in the background and can afford to search exhaustively.
    if deep_mode:
        SOFT_ITER_CAP = 250
        HARD_ITER_CAP = 500
        per_call_time = 5  # 5s per Phase A iteration instead of 1s
        print(f'[solver] DEEP MODE: caps soft={SOFT_ITER_CAP}, hard={HARD_ITER_CAP}, per-call={per_call_time}s')
    else:
        SOFT_ITER_CAP = 50
        HARD_ITER_CAP = 100
        per_call_time = 1
        print(f'[solver] Iteration caps: soft={SOFT_ITER_CAP} (return first profitable), hard={HARD_ITER_CAP}')

    # Track best partial solution in case no fully-profitable exists
    best_partial = None
    best_partial_score = float('inf')

    # Collect ALL profitable candidates, then pick best at end
    profitable_candidates = []
    iterations_since_improvement = 0  # for smart early stop
    done = False  # flag to break out of nested loops

    # Track every iteration for the diagnostic UI table
    all_attempts = []  # list of {v2, v1, total, status, min_profit, all_profitable}

    iter_count = 0
    # Wider search ranges so we explore all reasonable vendor counts
    for total_extra in range(0, 25):
        if done:
            break
        # Prefer adding 1i (cheaper) before 2i
        for v1_extra in range(0, total_extra + 1):
            if done:
                break
            v2_extra = total_extra - v1_extra
            v2 = lb_v2 + v2_extra
            v1 = lb_v1_base + v1_extra

            # Pre-flight: check basic capacity
            # Every day needs at least pair_count[d] + sd_count[d] vendors working
            min_capacity_per_day = [pair_count[d] + sd_count[d] for d in range(days)]
            max_needed = max(min_capacity_per_day)
            if v2 + v1 < max_needed:
                # Not enough vendors to cover daily demand. Skip all Pxx levels.
                continue
            # 2i vendors must cover DD on every day (only they can do DD)
            if v2 < max(pair_count):
                continue
            # No-consec-DD: need enough 2i for consecutive days
            max_consec_dd = max((pair_count[d] + pair_count[d + 1] for d in range(days - 1)), default=0)
            if v2 < max_consec_dd:
                continue

            # Pre-flight: enough SL capacity? Sum of slack across days 1..29 must >= total_slips.
            # (Day 0 can't host SL because no prior workday.)
            total_slack = sum(max(0, (v2 + v1) - pair_count[d] - sd_count[d]) for d in range(1, days))
            target_total_slips = round(sum(p*2 for p in pair_count) * sl2 + sum(sd_count) * sl1)
            if total_slack < target_total_slips:
                print(f'[solver] skip v2={v2}, v1={v1}: total slack {total_slack} < required slips {target_total_slips}')
                continue

            # Try Pxx levels until one works (lower Pxx = less SL needed = easier)
            iteration_found_any = False
            for pxx in pxx_levels:
                iter_count += 1
                # Iteration-based caps (deterministic across machines)
                if iter_count > HARD_ITER_CAP:
                    print(f'[solver] Hard iteration cap reached: {iter_count} > {HARD_ITER_CAP}')
                    done = True
                    break
                if iter_count > SOFT_ITER_CAP and profitable_candidates:
                    print(f'[solver] Soft iteration cap reached at iter {iter_count} with {len(profitable_candidates)} profitable candidate(s)')
                    done = True
                    break
                # Smart early stop: if we have profitable candidates and N more iterations
                # haven't improved the best one, stop. Adding more vendors only hurts profit.
                if profitable_candidates and iterations_since_improvement >= 8:
                    print(f'[solver] Early stop: {iterations_since_improvement} iterations without improvement')
                    done = True
                    break
                mc = monte_carlo_slips(pair_count, sd_count, sl2, sl1, pxx)
                sl_needed = []
                feas = True
                for d in range(days):
                    slack = v2 + v1 - pair_count[d] - sd_count[d]
                    pxx_val = mc['pxx_sl'][d]
                    if slack < pxx_val:
                        feas = False
                        break
                    expected = mc['expected_sl'][d]
                    if d == 0:
                        sl_needed.append(0)
                    elif slack < pxx_val + 1:
                        sl_needed.append(pxx_val)
                    else:
                        sl_needed.append(min(slack, round(expected)))
                if not feas:
                    # Pxx too high for this vendor count — but lower Pxx might work
                    continue
                # NOTE: do NOT bump sl_needed up to slack max. Slack is just "vendors not
                # doing DD/SD that day" — but SL requires day-before work too, so effective
                # slack is lower. Bumping creates per-day requirements the solver can't meet.
                # Instead: use Pxx-based per-day minimums, and let the total==82 constraint
                # distribute remaining SL where the solver finds capacity.

                # min_working_days disabled. Vendors with no work just idle on those days.
                min_working_days = 0
                print(f'[solver] iter {iter_count}: v2={v2}, v1={v1}, pxx={pxx}, target_total_sl={mc["total_slips"]}, max_work={max_working_days}')
                activity, status = solve_schedule(v2, v1, days, pair_count, sd_count,
                                                  peak_day, sl_needed, mc['total_slips'], max_working_days, min_working_days, per_call_time)
                if activity is None:
                    print(f'[solver]   CP-SAT: {status}')
                    # Record this attempt
                    all_attempts.append({
                        'v2': v2, 'v1': v1, 'total': v2 + v1, 'pxx': pxx,
                        'status': status.lower(), 'min_profit': None, 'all_profitable': False,
                    })
                    # INFEASIBLE here likely means SL slots too tight. Try lower Pxx (less SL).
                    # MODEL_INVALID or other errors won't be fixed by lower Pxx — break out.
                    if status not in ('INFEASIBLE',):
                        break
                    continue
                roster, payouts = assign_slabs(
                    activity, v2, days, dd_pairs_per_day, sd_slabs_per_day,
                    slab_rates_2i, slab_rates_1i, dd_discount
                )
                # Compute profitability stats (Phase A — fast check)
                all_prof = True
                unprof_sum = 0.0
                min_2i_prof = float('inf')
                min_1i_prof = float('inf')
                for v in range(v2):
                    prof = payouts[v] - cost_2i
                    if prof < min_2i_prof:
                        min_2i_prof = prof
                    if prof < -0.5:
                        all_prof = False
                        unprof_sum += -prof
                for v in range(v2, v2 + v1):
                    prof = payouts[v] - cost_1i
                    if prof < min_1i_prof:
                        min_1i_prof = prof
                    if prof < -0.5:
                        all_prof = False
                        unprof_sum += -prof

                print(f'[solver]   Phase A: min_2i=₹{min_2i_prof:.0f}, min_1i=₹{min_1i_prof:.0f}, unprof=₹{unprof_sum:.0f}')

                # Phase B: if Phase A is profitable, run quick CP-SAT slab optimization
                if all_prof:
                    # Phase B per candidate. Deep mode searches longer.
                    phase_b_time = 8 if deep_mode else 2
                    print(f'[solver]   Phase A profitable. Running Phase B ({phase_b_time}s)...')
                    optimized, opt_status = optimize_slabs_cpsat(
                        activity, v2, days, dd_pairs_per_day, sd_slabs_per_day,
                        slab_rates_2i, slab_rates_1i, dd_discount, cost_2i, cost_1i, phase_b_time
                    )
                    if optimized is not None:
                        opt_roster, opt_payouts = optimized
                        # Recompute stats
                        opt_min_2i = min(opt_payouts[v] - cost_2i for v in range(v2))
                        opt_min_1i = min(opt_payouts[v] - cost_1i for v in range(v2, v2 + v1))
                        print(f'[solver]   Phase B done ({opt_status}): min_2i=₹{opt_min_2i:.0f}, min_1i=₹{opt_min_1i:.0f}')
                        # Use optimized result if better
                        if min(opt_min_2i, opt_min_1i) >= min(min_2i_prof, min_1i_prof):
                            roster = opt_roster
                            payouts = opt_payouts
                            print(f'[solver]   Using Phase B result (better min profit)')
                        else:
                            print(f'[solver]   Phase B worse than Phase A, keeping Phase A')
                    else:
                        print(f'[solver]   Phase B failed ({opt_status}), using Phase A')

                # Total profit across all vendors
                total_payout_cand = sum(payouts)
                total_fixed_cand = v2 * cost_2i + v1 * cost_1i
                total_profit_cand = total_payout_cand - total_fixed_cand

                # Per-vendor profits (for scoring stats)
                vendor_profits = []
                for vi in range(v2):
                    vendor_profits.append(payouts[vi] - cost_2i)
                for vi in range(v2, v2 + v1):
                    vendor_profits.append(payouts[vi] - cost_1i)
                # P75 of vendor profits (ascending sort, 75th-percentile vendor)
                vps_sorted = sorted(vendor_profits)
                p75_idx = int(0.75 * (len(vps_sorted) - 1))
                p75_profit = vps_sorted[p75_idx] if vps_sorted else 0
                # 2i vs 1i averages
                profits_2i_list = vendor_profits[:v2]
                profits_1i_list = vendor_profits[v2:]
                avg_2i_profit = sum(profits_2i_list) / len(profits_2i_list) if profits_2i_list else 0
                avg_1i_profit = sum(profits_1i_list) / len(profits_1i_list) if profits_1i_list else 0
                v2_premium_ok = (v2 == 0 or v1 == 0 or avg_2i_profit > avg_1i_profit)

                candidate = {
                    'v2': v2, 'v1': v1, 'roster': roster, 'status': status,
                    'pxx_achieved': pxx, 'sl_needed': sl_needed, 'mc': mc, 'payouts': payouts,
                    'all_profitable': all_prof, 'unprofitable_amount': unprof_sum,
                    'activity': activity,  # for polish phase
                    'total_profit': total_profit_cand,
                    'p75_profit': p75_profit,
                    'avg_2i_profit': avg_2i_profit,
                    'avg_1i_profit': avg_1i_profit,
                    'v2_premium_ok': v2_premium_ok,
                }

                # Compute min for tracking
                min_overall_recorded = min(min_2i_prof, min_1i_prof) if (v2 > 0 and v1 > 0) else (min_2i_prof if v2 > 0 else min_1i_prof)
                all_attempts.append({
                    'v2': v2, 'v1': v1, 'total': v2 + v1, 'pxx': pxx,
                    'status': status, 'min_profit': float(min_overall_recorded), 'all_profitable': bool(all_prof),
                })

                if all_prof:
                    # Compute min profit for ranking
                    min_overall = min(min_2i_prof, min_1i_prof)
                    candidate['min_profit'] = min_overall
                    # Check if this improves over current best (by total profit primary)
                    if profitable_candidates:
                        current_best = min(profitable_candidates, key=lambda c: (-c['total_profit'], c['v2']+c['v1']))
                        current_best_score = (-current_best['total_profit'], current_best['v2']+current_best['v1'])
                        new_score = (-total_profit_cand, v2 + v1)
                        if new_score < current_best_score:
                            iterations_since_improvement = 0
                        else:
                            iterations_since_improvement += 1
                    else:
                        iterations_since_improvement = 0
                    profitable_candidates.append(candidate)
                    print(f'[solver]   PROFITABLE candidate: v2={v2}, v1={v1}, pxx={pxx}, total_profit=₹{total_profit_cand:.0f}, min_profit=₹{min_overall:.0f} (no_improve_streak={iterations_since_improvement})')
                else:
                    # Track best partial
                    if unprof_sum < best_partial_score:
                        best_partial_score = unprof_sum
                        best_partial = candidate
                        print(f'[solver]   new best partial (unprofitable=₹{unprof_sum:.0f})')
                    if profitable_candidates:
                        iterations_since_improvement += 1

    # Done iterating. Pick best candidate.
    if profitable_candidates:
        # Cluster-level scoring (always): (1) highest total profit, (2) fewest vendors, (3) highest min profit
        # Deep mode (pan-India) adds ONE soft tiebreaker: prefer candidates where 2i avg > 1i avg.
        # This sits AFTER total profit and vendor count — it only breaks ties, never overrides.
        if deep_mode:
            def candidate_score(c):
                return (
                    -c['total_profit'],
                    c['v2'] + c['v1'],
                    0 if c.get('v2_premium_ok', True) else 1,  # soft: 2i should earn > 1i
                    -c['min_profit'],
                )
        else:
            def candidate_score(c):
                return (-c['total_profit'], c['v2'] + c['v1'], -c['min_profit'])
        profitable_candidates.sort(key=candidate_score)
        best = profitable_candidates[0]
        print(f'[solver] Searched {iter_count} iterations. Found {len(profitable_candidates)} profitable candidates.')
        if deep_mode:
            premium = "✓ 2i premium" if best.get('v2_premium_ok') else "⚠ no 2i premium"
            print(f'[solver] Initial pick (deep mode): v2={best["v2"]}, v1={best["v1"]}, total_profit=₹{best["total_profit"]:.0f}, 2i avg=₹{best.get("avg_2i_profit", 0):.0f}, 1i avg=₹{best.get("avg_1i_profit", 0):.0f} {premium}')
        else:
            print(f'[solver] Initial pick: v2={best["v2"]}, v1={best["v1"]}, total_profit=₹{best["total_profit"]:.0f}, min_profit=₹{best["min_profit"]:.0f}')

        # POLISH PHASE: deep mode runs polish longer for best refinement.
        polish_time = 60 if deep_mode else 10
        if 'activity' in best:
            print(f'[solver] Polish phase: {polish_time}s with heavy spread weight (=10)')
            polished_result, polished_status = optimize_slabs_cpsat(
                best['activity'], best['v2'], days, dd_pairs_per_day, sd_slabs_per_day,
                slab_rates_2i, slab_rates_1i, dd_discount, cost_2i, cost_1i, polish_time, spread_weight=10
            )
            if polished_result is not None:
                pol_roster, pol_payouts = polished_result
                # CRITICAL: verify every single vendor is profitable
                pol_v2_profits = [pol_payouts[v] - cost_2i for v in range(best['v2'])]
                pol_v1_profits = [pol_payouts[v] - cost_1i for v in range(best['v2'], best['v2'] + best['v1'])]
                pol_min_2i = min(pol_v2_profits) if pol_v2_profits else 0
                pol_min_1i = min(pol_v1_profits) if pol_v1_profits else 0
                pol_max_2i = max(pol_v2_profits) if pol_v2_profits else 0
                pol_max_1i = max(pol_v1_profits) if pol_v1_profits else 0
                pol_min = min(pol_min_2i, pol_min_1i)
                all_profitable = all(p >= 0 for p in pol_v2_profits) and all(p >= 0 for p in pol_v1_profits)
                if all_profitable and pol_min >= best['min_profit'] - 100:
                    print(f'[solver] Polish done ({polished_status}): 2i spread ₹{pol_max_2i - pol_min_2i:.0f}, 1i spread ₹{pol_max_1i - pol_min_1i:.0f}, min profit ₹{pol_min:.0f}')
                    best['roster'] = pol_roster
                    best['payouts'] = pol_payouts
                    best['min_profit'] = pol_min
                else:
                    if not all_profitable:
                        unprofitable = [i for i, p in enumerate(pol_v2_profits + pol_v1_profits) if p < 0]
                        print(f'[solver] Polish made vendors unprofitable: {len(unprofitable)} vendors below ₹0. Keeping pre-polish.')
                    else:
                        print(f'[solver] Polish reduced min profit. Keeping pre-polish.')
            else:
                print(f'[solver] Polish failed ({polished_status}), keeping pre-polish')
        best['all_attempts'] = all_attempts
        return best
    # Loop done without fully profitable. Return best partial if any.
    if best_partial:
        print(f'[solver] Exhausted search. Returning best partial: v2={best_partial["v2"]}, v1={best_partial["v1"]}, unprofitable_sum=₹{best_partial_score:.0f}')
        best_partial['all_attempts'] = all_attempts
        return best_partial
    # No feasible schedule at all — return diagnostics dict so caller can show what was tried
    print(f'[solver] No feasible schedule found in any iteration. Tried {len(all_attempts)} combinations.')
    return {'_diagnostic_only': True, 'all_attempts': all_attempts}


def run_solver(params):
    total_sites = int(params['total_sites'])
    days = int(params.get('days', 30))
    peak_ratio = float(params['peak_ratio'])
    elig_pct = float(params['elig_pct'])
    sl2_rate = float(params['sl2_rate'])
    sl1_rate = float(params['sl1_rate'])
    slab_rates_2i = [float(x) for x in params['slab_rates']]
    # If slab_rates_1i not provided, fall back to same as 2i (backward compatibility)
    slab_rates_1i = [float(x) for x in params.get('slab_rates_1i', params['slab_rates'])]
    slab_mix = [float(x) for x in params['slab_mix']]
    dd_elig_slabs = [bool(x) for x in params['dd_elig_slabs']]
    cost_2i = float(params['cost_2i'])
    cost_1i = float(params['cost_1i'])
    dd_discount = float(params['dd_discount'])
    baseline_per_site = float(params.get('baseline_per_site', 10572))
    target_pxx = float(params.get('target_pxx', 0.75))
    time_limit = int(params.get('time_limit_sec', 30))
    max_working_days = int(params.get('max_working_days', 26))
    skew_pct = float(params.get('skew_pct', 100))
    cluster = params.get('cluster', 'Pune')

    daily = bell_curve_demand(total_sites, days, peak_ratio, skew_pct, cluster)
    dd_pairs, sd_slabs = compute_daily_demand(daily, slab_mix, dd_elig_slabs, elig_pct)
    pair_count = [len(p) for p in dd_pairs]
    sd_count = [sum(s) for s in sd_slabs]
    peak_day = daily.index(max(daily))

    print(f'[solver] daily sum={sum(daily)} expected={total_sites}, peak={peak_day}, max_work_days={max_working_days}')
    print(f'[solver] 2i rates: {slab_rates_2i}')
    print(f'[solver] 1i rates: {slab_rates_1i}')

    # Deep mode: triggered by pan-India precompute, allows longer search per cluster
    deep_mode = bool(params.get('_deep_mode', False))

    result = find_optimal_fast(daily, dd_pairs, sd_slabs, pair_count, sd_count,
                               peak_day, sl2_rate, sl1_rate, target_pxx, slab_rates_2i, slab_rates_1i,
                               dd_discount, cost_2i, cost_1i, time_limit, max_working_days, deep_mode=deep_mode)

    if result is None or result.get('_diagnostic_only'):
        diag_attempts = result.get('all_attempts', []) if result else []
        return {
            'ok': False,
            'reason': 'No feasible schedule found in any vendor combination tried. The constraints (slip count, peak load, DD limits) are too tight at these settings. Try: lowering slip%, raising DD discount, lowering peak ratio, or reducing total sites.',
            'daily': daily,
            'pair_count_per_day': pair_count,
            'sd_count_per_day': sd_count,
            'peak_day': peak_day,
            'total_slips': round(sum([p*2 for p in pair_count]) * sl2_rate + sum(sd_count) * sl1_rate),
            'all_attempts': diag_attempts,
        }

    v2 = result['v2']
    v1 = result['v1']
    total_v = v2 + v1
    roster = result['roster']

    # POST-PROCESS: redistribute which vendor owns each SL cell, based on
    # probabilistic slip ownership (which vendor's site actually slipped).
    #
    # Logic:
    #   1. Solver placed SL on certain days — keep those days (the columns).
    #   2. For each individual site done in the schedule, coin-flip (seeded) whether
    #      it slipped. The vendor that did that site is "owed" 1 SL recovery.
    #   3. Sum slip ownership per vendor → desired SL count per vendor.
    #   4. The solver placed SL cells on specific (vendor, day) positions. We
    #      reshuffle these cells across vendors — each day, the SL count on that
    #      day stays the same, but WHICH vendors have SL that day changes to
    #      match the desired ownership counts.
    #
    # The total SL count in the roster stays exactly what the solver gave us.
    # No SL is gained or lost. We just change rows, not columns.
    import random as _random
    rng = _random.Random(42)

    # Step 1: simulate which sites slip → count owned slips per vendor
    desired_sl_per_vendor = [0] * total_v
    for vi in range(total_v):
        for d in range(days):
            cell = roster[vi][d]
            if cell[0] == 'DD':
                for _ in range(2):  # 2 sites per DD pair
                    if rng.random() < sl2_rate:
                        desired_sl_per_vendor[vi] += 1
            elif cell[0] == 'SD':
                if rng.random() < sl1_rate:
                    desired_sl_per_vendor[vi] += 1

    sim_total = sum(desired_sl_per_vendor)
    print(f'[post] Simulated slip ownership: {sim_total} total slips owned by vendors')

    # Step 2: find all SL cells in the solver roster, grouped by day
    sl_days_to_vendors = {}  # day -> list of vendor indices that have SL that day
    for d in range(days):
        sl_days_to_vendors[d] = [
            vi for vi in range(total_v) if roster[vi][d][0] == 'SL'
        ]
    solver_total_sl = sum(len(v) for v in sl_days_to_vendors.values())
    theoretical_sl = result['mc']['total_slips']
    print(f'[post] Solver placed {solver_total_sl} SL cells; theoretical = {theoretical_sl}')

    # Step 2b: TRIM excess SL down to theoretical count.
    # The solver uses >= so it may pad. We drop the excess (convert to idle in solver
    # roster) so total SL matches the deterministic count from slip rates.
    # Drop from later days first (less critical recovery-wise), and from days with
    # excess cells beyond what's needed for Pxx coverage.
    excess = solver_total_sl - theoretical_sl
    if excess > 0:
        # Build a flat list of (day, vendor) pairs to drop, prioritizing latest days
        drop_candidates = []
        for d in sorted(sl_days_to_vendors.keys(), reverse=True):  # latest days first
            for vi in sl_days_to_vendors[d]:
                drop_candidates.append((d, vi))
        for d, vi in drop_candidates[:excess]:
            roster[vi][d] = ('idle', None)
            sl_days_to_vendors[d].remove(vi)
        solver_total_sl = sum(len(v) for v in sl_days_to_vendors.values())
        print(f'[post] Trimmed {excess} excess SL cells. Now {solver_total_sl} cells.')

    # Step 3: reconcile desired vs available. The solver gave us N slots,
    # ownership totals to M (M could be different due to coin-flip variance).
    # We'll allocate the N slots to vendors as proportionally to desired as possible,
    # capped by each vendor's actual desired count.
    if solver_total_sl != sim_total and solver_total_sl > 0:
        # Scale desired counts to sum to solver_total_sl
        scale = solver_total_sl / sim_total if sim_total > 0 else 0
        scaled = [d * scale for d in desired_sl_per_vendor]
        # Round to integers, fix drift
        new_counts = [int(round(s)) for s in scaled]
        diff = solver_total_sl - sum(new_counts)
        # Distribute the diff to highest-fractional-part vendors
        fracs = sorted(range(total_v), key=lambda i: -(scaled[i] - int(scaled[i])))
        for i in range(abs(diff)):
            idx = fracs[i % len(fracs)]
            new_counts[idx] += 1 if diff > 0 else -1
            if new_counts[idx] < 0:
                new_counts[idx] = 0
        desired_sl_per_vendor = new_counts
        print(f'[post] Scaled vendor SL targets to fit {solver_total_sl} solver slots')

    # Step 4: redistribute SL cells day-by-day
    # For each day, we have a count of SL slots. We pick which vendors get SL that day.
    # Constraints: vendor must have worked the previous day (for d > 0); vendor must
    # not already have DD/SD on that day.
    # Strategy: prioritize vendors with high remaining-desired counts.
    new_roster = [[(cell[0], cell[1]) for cell in row] for row in roster]
    # Wipe all SL cells first; we'll rebuild
    for vi in range(total_v):
        for d in range(days):
            if new_roster[vi][d][0] == 'SL':
                new_roster[vi][d] = ('idle', None)

    remaining_desired = desired_sl_per_vendor[:]
    sl_placed = 0
    sl_failed_to_place = 0

    for d in sorted(sl_days_to_vendors.keys()):
        slots_today = len(sl_days_to_vendors[d])
        if slots_today == 0:
            continue
        # Find eligible vendors for SL on this day:
        # - Not already doing DD or SD that day
        # - Worked previous day (DD or SD) — required by SL rule
        # - For day 0: SL not allowed at all (no prior day)
        eligible = []
        if d > 0:
            for vi in range(total_v):
                if new_roster[vi][d][0] in ('DD', 'SD', 'SL'):
                    continue
                prev = new_roster[vi][d - 1][0]
                if prev not in ('DD', 'SD'):
                    continue
                eligible.append(vi)
        # Sort eligible by remaining desired count (descending)
        eligible.sort(key=lambda vi: -remaining_desired[vi])
        # Place SL for top `slots_today` vendors
        for vi in eligible[:slots_today]:
            new_roster[vi][d] = ('SL', None)
            sl_placed += 1
            if remaining_desired[vi] > 0:
                remaining_desired[vi] -= 1
        # If fewer eligible vendors than slots needed, those SL slots are lost
        # (extremely rare; solver should have made these placeable)
        if len(eligible) < slots_today:
            sl_failed_to_place += slots_today - len(eligible)

    roster = new_roster
    final_sl = sum(1 for vi in range(total_v) for d in range(days) if roster[vi][d][0] == 'SL')
    print(f'[post] Redistributed SL: {sl_placed} placed, {sl_failed_to_place} slots lost, final count {final_sl}')

    vendors = []
    for vi in range(total_v):
        is_v2 = vi < v2
        dd_d = sd_d = sl_d = idle_d = 0
        payout = 0
        # 2i uses 2i rates; 1i uses 1i rates (1i can't do DD, only SD)
        rates = slab_rates_2i if is_v2 else slab_rates_1i
        for d in range(days):
            cell = roster[vi][d]
            if cell[0] == 'DD':
                s1, s2 = cell[1]
                # DD always uses 2i rates with discount
                r1, r2 = slab_rates_2i[s1], slab_rates_2i[s2]
                payout += max(r1, r2) + min(r1, r2) * dd_discount
                dd_d += 1
            elif cell[0] == 'SD':
                payout += rates[cell[1]]
                sd_d += 1
            elif cell[0] == 'SL':
                sl_d += 1
            else:
                idle_d += 1
        fixed = cost_2i if is_v2 else cost_1i
        profit = payout - fixed
        vendors.append({
            'name': f'V{vi+1:02d}',
            'type': '2-Install' if is_v2 else '1-Install',
            'dd_days': dd_d, 'sd_sites': sd_d, 'sl_days': sl_d, 'idle_days': idle_d,
            'sites': dd_d * 2 + sd_d, 'fixed_cost': fixed,
            'payout': round(payout), 'profit': round(profit),
        })

    slab_labels = ['S1', 'S2', 'S3', 'S4']
    roster_out = []
    for vi in range(total_v):
        row = []
        for d in range(days):
            cell = roster[vi][d]
            if cell[0] == 'DD':
                s1, s2 = cell[1]
                row.append({'type': 'DD', 'label': f'{slab_labels[s1]}+{slab_labels[s2]}'})
            elif cell[0] == 'SD':
                row.append({'type': 'SD', 'label': slab_labels[cell[1]]})
            elif cell[0] == 'SL':
                row.append({'type': 'SL', 'label': ''})
            else:
                row.append({'type': 'idle', 'label': ''})
        roster_out.append(row)

    total_payout = sum(v['payout'] for v in vendors)
    total_cost = sum(v['fixed_cost'] for v in vendors)
    total_profit = sum(v['profit'] for v in vendors)
    baseline = total_sites * baseline_per_site
    savings_pct = 100 * (baseline - total_payout) / baseline if baseline > 0 else 0

    v2_list = [v for v in vendors if v['type'] == '2-Install']
    v1_list = [v for v in vendors if v['type'] == '1-Install']
    avg_2i = sum(v['profit'] for v in v2_list) / len(v2_list) if v2_list else 0
    avg_1i = sum(v['profit'] for v in v1_list) / len(v1_list) if v1_list else 0
    min_2i = min((v['profit'] for v in v2_list), default=0)
    min_1i = min((v['profit'] for v in v1_list), default=0)

    # Compute slip breakdown by vendor type
    # 2i sites (from DD): sum(pair_count) * 2; slips = those * sl2_rate
    # 1i sites (from SD): sum(sd_count); slips on 1i-handled sites are tracked separately
    # In our model, SD can be done by either 2i or 1i vendor — but slip rate applies to site type, not vendor
    # Approximate: slips proportional to (sites of type) × (slip rate of type)
    dd_sites_total = sum(pair_count) * 2
    sd_sites_total = sum(sd_count)
    slips_2i_est = round(dd_sites_total * sl2_rate)
    slips_1i_est = round(sd_sites_total * sl1_rate)

    return {
        'ok': True,
        'all_profitable': result.get('all_profitable', True),
        'unprofitable_amount': round(result.get('unprofitable_amount', 0)),
        'status': result['status'],
        'v2': v2, 'v1': v1, 'total_v': total_v,
        'daily': daily,
        'pair_count_per_day': pair_count,
        'sd_count_per_day': sd_count,
        'sl_needed_by_day': result['sl_needed'],
        'peak_day': peak_day,
        'roster': roster_out,
        'vendors': vendors,
        'total_payout': total_payout,
        'total_cost': total_cost,
        'total_profit': total_profit,
        'savings_pct': round(savings_pct, 1),
        'avg_2i': round(avg_2i),
        'avg_1i': round(avg_1i),
        'min_2i': round(min_2i),
        'min_1i': round(min_1i),
        'total_slips': sum(v['sl_days'] for v in vendors),
        'slips_theoretical': result['mc']['total_slips'],
        'pxx_target': target_pxx,
        'pxx_achieved': result['pxx_achieved'],
        'all_attempts': result.get('all_attempts', []),
        # New for PDF
        'slab_rates_2i': slab_rates_2i,
        'slab_rates_1i': slab_rates_1i,
        'slips_from_2i': slips_2i_est,
        'slips_from_1i': slips_1i_est,
        'sl2_rate': sl2_rate,
        'sl1_rate': sl1_rate,
        'baseline_per_site': baseline_per_site,
        'peak_ratio': peak_ratio,
        'skew_pct': skew_pct,
    }


# ============================================================
# PDF REPORT GENERATION
# ============================================================

BRAND_COLOR = colors.HexColor('#131ac3') if REPORTLAB_OK else None
BRAND_LIGHT = colors.HexColor('#e3e4f8') if REPORTLAB_OK else None
ACCENT_GREEN = colors.HexColor('#187a3b') if REPORTLAB_OK else None
ACCENT_RED = colors.HexColor('#c00000') if REPORTLAB_OK else None
GREY_LIGHT = colors.HexColor('#f5f5f7') if REPORTLAB_OK else None


def build_pdf_report(data, city_name):
    """CFO-grade PDF report. Editorial design, minimal palette.
    Uses 'Rs' instead of ₹ to avoid font glyph issues.
    """
    if not REPORTLAB_OK:
        raise RuntimeError('reportlab not installed. Run: pip3 install reportlab --break-system-packages')

    # ---- Page geometry ----
    PAGE_W, PAGE_H = A4
    MARGIN = 20 * mm
    CONTENT_W = PAGE_W - 2 * MARGIN

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=20 * mm, bottomMargin=20 * mm,
        title=f'Vendor Plan - {city_name}',
        author='SolarSquare Energy',
    )

    # ---- Palette ----
    INK = colors.HexColor('#0f1419')
    SLATE = colors.HexColor('#5e6470')
    MUTED = colors.HexColor('#9aa0a8')
    HAIRLINE = colors.HexColor('#dadce0')
    ACCENT = colors.HexColor('#131ac3')
    SOFT = colors.HexColor('#f8fafc')
    POSITIVE = colors.HexColor('#1a7f37')
    NEGATIVE = colors.HexColor('#b91c1c')
    PAPER = colors.HexColor('#ffffff')

    SERIF = 'Times-Roman'
    SERIF_BOLD = 'Times-Bold'
    SERIF_ITALIC = 'Times-Italic'
    SANS = FONT_REGULAR
    SANS_BOLD = FONT_BOLD

    # ---- Helpers ----
    def P(text, font=SANS, size=10, color=INK, align=TA_LEFT, leading=None):
        return Paragraph(text, ParagraphStyle(
            'p', fontName=font, fontSize=size,
            leading=leading or size * 1.4, textColor=color, alignment=align,
        ))

    def rs(v, with_unit=True):
        """Format INR using 'Rs' to avoid PDF glyph issues."""
        v = round(v)
        sign = '-' if v < 0 else ''
        v = abs(v)
        if v >= 1e7: out = f'{sign}Rs {v/1e7:.2f} Cr'
        elif v >= 1e5: out = f'{sign}Rs {v/1e5:.2f} L'
        elif v >= 1e3: out = f'{sign}Rs {v/1e3:.1f}K'
        else: out = f'{sign}Rs {v}'
        return out if with_unit else out.replace('Rs ', '')

    def rs_full(v):
        """Indian number system with Rs prefix."""
        v = round(v)
        sign = '-' if v < 0 else ''
        v = abs(v)
        if v < 1000: return f'{sign}Rs {v}'
        s = str(v)
        last_three = s[-3:]
        rest = s[:-3]
        parts = []
        while len(rest) > 2:
            parts.append(rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.append(rest)
        return f'{sign}Rs {",".join(reversed(parts))},{last_three}'

    def pct(arr, p):
        if not arr: return None
        s = sorted(arr)
        if p <= 0: return s[0]
        if p >= 100: return s[-1]
        idx = (p / 100.0) * (len(s) - 1)
        lo = int(idx); hi = lo + 1 if lo + 1 < len(s) else lo
        return s[lo] + (s[hi] - s[lo]) * (idx - lo)

    now = datetime.datetime.now()
    date_str = now.strftime('%d %B %Y')

    # Derived
    daily = data['daily']
    peak_day = data['peak_day']
    n_days = len(daily)
    total_v = data['v2'] + data['v1']
    total_sites = sum(daily)
    avg_daily = total_sites / n_days
    peak_to_bau = max(daily) / avg_daily if avg_daily else 1
    pair_counts = data['pair_count_per_day']
    sd_counts = data['sd_count_per_day']
    total_dd_pairs = sum(pair_counts)
    total_dd_sites = total_dd_pairs * 2
    total_sd_sites = sum(sd_counts)

    # Slippage breakdown (deterministic from slip rates and site counts)
    # We don't have sl2/sl1 explicitly in data but we can back-compute approximately
    # Read these from the params if surfaced; otherwise use defaults
    total_slips = data['total_slips']

    profits_2i = [v['profit'] for v in data['vendors'] if v['type'] == '2-Install']
    profits_1i = [v['profit'] for v in data['vendors'] if v['type'] == '1-Install']
    payouts_2i = [v['payout'] for v in data['vendors'] if v['type'] == '2-Install']
    payouts_1i = [v['payout'] for v in data['vendors'] if v['type'] == '1-Install']
    avg_2i_profit = sum(profits_2i) / len(profits_2i) if profits_2i else 0
    avg_1i_profit = sum(profits_1i) / len(profits_1i) if profits_1i else 0
    avg_2i_payout = sum(payouts_2i) / len(payouts_2i) if payouts_2i else 0
    avg_1i_payout = sum(payouts_1i) / len(payouts_1i) if payouts_1i else 0

    story = []

    # ============================================================
    # PAGE 1 — COVER
    # ============================================================
    story.append(P('SOLARSQUARE', font=SANS_BOLD, size=8, color=ACCENT))
    story.append(Spacer(1, 4 * mm))
    story.append(P('Vendor Plan', font=SERIF, size=38, color=INK, leading=42))
    story.append(P(f'<i>{city_name}</i>', font=SERIF_ITALIC, size=18, color=SLATE, leading=22))
    story.append(Spacer(1, 10 * mm))

    # Meta line
    meta = Table([[
        P(date_str, font=SANS, size=9, color=SLATE),
        P(f'Status: {data.get("status", "—")}', font=SANS, size=9, color=SLATE, align=TA_CENTER),
        P(f'{total_v} vendors  |  {total_sites} sites  |  {data["savings_pct"]}% savings',
          font=SANS, size=9, color=SLATE, align=TA_RIGHT),
    ]], colWidths=[CONTENT_W/3]*3)
    meta.setStyle(TableStyle([
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('LINEABOVE', (0,0), (-1,0), 0.5, HAIRLINE),
        ('LINEBELOW', (0,0), (-1,0), 0.5, HAIRLINE),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(meta)
    story.append(Spacer(1, 12 * mm))

    # KPI grid 2 rows x 3 cols (no boxes - just labels + big numbers)
    def kpi(label, value, sub=None, accent=False):
        col = ACCENT if accent else INK
        rows = [
            [P(label.upper(), font=SANS_BOLD, size=7, color=MUTED, leading=10)],
            [P(value, font=SERIF, size=22, color=col, leading=26)],
        ]
        if sub:
            rows.append([P(sub, font=SANS, size=8, color=SLATE, leading=11)])
        t = Table(rows, colWidths=[CONTENT_W/3 - 4*mm])
        t.setStyle(TableStyle([
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,0), 3),
            ('BOTTOMPADDING', (0,1), (-1,-1), 2),
        ]))
        return t

    row1 = Table([[
        kpi('Total vendors', str(total_v), sub=f'{data["v2"]} two-install + {data["v1"]} one-install'),
        kpi('Monthly payout', rs(data['total_payout']), sub=rs_full(data['total_payout'])),
        kpi('SSE savings', f'{data["savings_pct"]}%', sub=f'vs Rs 10,572/site baseline', accent=True),
    ]], colWidths=[CONTENT_W/3]*3)
    row1.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP'),
                              ('LEFTPADDING', (0,0), (-1,-1), 0),
                              ('RIGHTPADDING', (0,0), (-1,-1), 0)]))
    story.append(row1)
    story.append(Spacer(1, 10 * mm))

    row2 = Table([[
        kpi('Total sites', str(total_sites), sub=f'Peak/BAU ratio {peak_to_bau:.2f}x'),
        kpi('Total slippages', str(total_slips), sub=f'recovered across {n_days} days'),
        kpi('Net profit', rs(data['total_profit']),
            sub=f'{round(100*data["total_profit"]/data["total_payout"], 1) if data.get("total_payout") else 0}% margin',
            accent=data['total_profit'] >= 0),
    ]], colWidths=[CONTENT_W/3]*3)
    row2.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP'),
                              ('LEFTPADDING', (0,0), (-1,-1), 0),
                              ('RIGHTPADDING', (0,0), (-1,-1), 0)]))
    story.append(row2)

    # ============================================================
    # PAGE 2 — DEMAND PROFILE (end-of-month skew)
    # ============================================================
    story.append(PageBreak())
    story.append(P('Demand Profile', font=SERIF, size=22, color=INK, leading=26))
    story.append(Spacer(1, 2 * mm))
    # Describe the stress + skew settings
    pr = data.get('peak_ratio', 1.6)
    sk = data.get('skew_pct', 100)
    stress_label = 'P50 (typical)' if pr <= 1.45 else ('P75 (stressed)' if pr <= 1.55 else 'P90 (planning)')
    if sk <= 5:
        skew_label = 'flat distribution (every day equal)'
    elif sk >= 95:
        skew_label = 'full historical skew'
    else:
        skew_label = f'{int(sk)}% historical / {100-int(sk)}% flat blend'
    story.append(P(
        f'Stress level: <b>{stress_label}</b>. Skew: <b>{skew_label}</b>. '
        f'Peak day: <b>D{peak_day+1}</b> with {max(daily)} sites. '
        f'Average: {avg_daily:.1f} sites/day. Day 1: {daily[0]} sites.',
        font=SANS, size=10, color=SLATE
    ))
    story.append(Spacer(1, 10 * mm))

    # Bar chart - vertical bars with site counts BELOW
    max_d = max(daily) if daily else 1
    BAR_AREA_H = 55 * mm
    bar_w = (CONTENT_W / n_days) - 0.5 * mm

    def make_bar(value, is_peak):
        h = max(1, int((value / max_d) * BAR_AREA_H))
        col = ACCENT if is_peak else INK
        return Table([['']], colWidths=[bar_w], rowHeights=[h],
                     style=TableStyle([('BACKGROUND', (0,0), (-1,-1), col)]))

    bar_cells = []
    for i, v in enumerate(daily):
        # Wrap each bar in a bottom-aligned cell
        wrapper = Table([[make_bar(v, i == peak_day)]],
                        colWidths=[CONTENT_W/n_days], rowHeights=[BAR_AREA_H + 2*mm],
                        style=TableStyle([
                            ('VALIGN', (0,0), (-1,-1), 'BOTTOM'),
                            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                            ('LEFTPADDING', (0,0), (-1,-1), 0),
                            ('RIGHTPADDING', (0,0), (-1,-1), 0),
                            ('TOPPADDING', (0,0), (-1,-1), 0),
                            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
                        ]))
        bar_cells.append(wrapper)

    bar_row = Table([bar_cells], colWidths=[CONTENT_W/n_days]*n_days)
    bar_row.setStyle(TableStyle([
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('VALIGN', (0,0), (-1,-1), 'BOTTOM'),
        ('LINEBELOW', (0,0), (-1,-1), 0.75, INK),
    ]))
    story.append(bar_row)

    # NUMBER OF INSTALLS row directly under each bar
    install_row = Table(
        [[P(str(v), font=SANS_BOLD if i == peak_day else SANS,
            size=6, color=ACCENT if i == peak_day else INK, align=TA_CENTER)
          for i, v in enumerate(daily)]],
        colWidths=[CONTENT_W/n_days]*n_days,
        rowHeights=[4.5*mm],
    )
    install_row.setStyle(TableStyle([
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 1),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(install_row)

    # Day labels (every 5 days + peak + endpoints)
    day_labels = []
    for i in range(n_days):
        show = (i + 1) % 5 == 0 or i == 0 or i == n_days - 1 or i == peak_day
        day_labels.append(P(f'D{i+1}' if show else '',
                            font=SANS, size=6,
                            color=ACCENT if i == peak_day else MUTED,
                            align=TA_CENTER))
    label_row = Table([day_labels], colWidths=[CONTENT_W/n_days]*n_days, rowHeights=[4*mm])
    label_row.setStyle(TableStyle([
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(label_row)

    # Demand summary stats
    story.append(Spacer(1, 8 * mm))
    stats = Table([
        [P('TOTAL SITES', font=SANS_BOLD, size=7, color=MUTED),
         P('PEAK DAY', font=SANS_BOLD, size=7, color=MUTED),
         P('AVERAGE/DAY', font=SANS_BOLD, size=7, color=MUTED),
         P('PEAK/BAU', font=SANS_BOLD, size=7, color=MUTED),
         P('DD PAIRS', font=SANS_BOLD, size=7, color=MUTED),
         P('SD SITES', font=SANS_BOLD, size=7, color=MUTED)],
        [P(str(total_sites), font=SERIF, size=18, color=INK),
         P(f'{max(daily)}', font=SERIF, size=18, color=ACCENT),
         P(f'{avg_daily:.1f}', font=SERIF, size=18, color=INK),
         P(f'{peak_to_bau:.2f}x', font=SERIF, size=18, color=INK),
         P(str(total_dd_pairs), font=SERIF, size=18, color=INK),
         P(str(total_sd_sites), font=SERIF, size=18, color=INK)],
        [P('across month', font=SANS, size=7, color=SLATE),
         P(f'on day {peak_day+1}', font=SANS, size=7, color=SLATE),
         P('sites', font=SANS, size=7, color=SLATE),
         P('ratio', font=SANS, size=7, color=SLATE),
         P(f'= {total_dd_sites} sites', font=SANS, size=7, color=SLATE),
         P('single days', font=SANS, size=7, color=SLATE)],
    ], colWidths=[CONTENT_W/6]*6)
    stats.setStyle(TableStyle([
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LINEABOVE', (0,0), (-1,0), 0.5, HAIRLINE),
        ('LINEBELOW', (0,-1), (-1,-1), 0.5, HAIRLINE),
    ]))
    story.append(stats)

    # ============================================================
    # PAGE 3 — RATE CARDS & SSE SAVINGS DERIVATION
    # ============================================================
    story.append(PageBreak())
    story.append(P('Rate Cards', font=SERIF, size=22, color=INK, leading=26))
    story.append(Spacer(1, 2 * mm))
    story.append(P(
        f'Per-site payment rates by slab. 2-Install vendors receive a discounted rate '
        f'on the second site in a double-install day (DD discount applied).',
        font=SANS, size=10, color=SLATE
    ))
    story.append(Spacer(1, 8 * mm))

    # Two side-by-side rate cards
    # Pull rates from data if available, fall back to defaults
    rates_2i = data.get('slab_rates_2i', [8000, 8500, 10000, 15000])
    rates_1i = data.get('slab_rates_1i', rates_2i)  # if not provided, same as 2i

    rate_header = [
        P('SLAB', font=SANS_BOLD, size=7, color=PAPER),
        P('2-INSTALL', font=SANS_BOLD, size=7, color=PAPER, align=TA_RIGHT),
        P('1-INSTALL', font=SANS_BOLD, size=7, color=PAPER, align=TA_RIGHT),
        P('DELTA', font=SANS_BOLD, size=7, color=PAPER, align=TA_RIGHT),
    ]
    rate_rows = [rate_header]
    for i, label in enumerate(['S1', 'S2', 'S3', 'S4']):
        r2 = rates_2i[i] if i < len(rates_2i) else 0
        r1 = rates_1i[i] if i < len(rates_1i) else r2
        delta = r1 - r2
        delta_str = '—' if delta == 0 else (f'+{rs(delta)}' if delta > 0 else f'{rs(delta)}')
        rate_rows.append([
            P(label, font=SANS, size=10, color=INK),
            P(rs_full(r2), font=SANS, size=10, color=INK, align=TA_RIGHT),
            P(rs_full(r1), font=SANS, size=10, color=INK, align=TA_RIGHT),
            P(delta_str, font=SANS, size=10, color=SLATE if delta == 0 else (POSITIVE if delta > 0 else NEGATIVE), align=TA_RIGHT),
        ])
    rate_table = Table(rate_rows, colWidths=[CONTENT_W*0.15, CONTENT_W*0.30, CONTENT_W*0.30, CONTENT_W*0.25])
    rate_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), ACCENT),
        ('LINEBELOW', (0,1), (-1,-1), 0.25, HAIRLINE),
        ('LINEABOVE', (0,1), (-1,1), 0.5, INK),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(rate_table)
    story.append(Spacer(1, 4 * mm))
    story.append(P('DD discount: second site in a pair pays 0.7x of its full slab rate.',
                   font=SANS, size=8, color=MUTED))

    # ---- SSE SAVINGS DERIVATION ----
    story.append(Spacer(1, 14 * mm))
    story.append(P('SSE Savings Derivation', font=SERIF, size=18, color=INK, leading=22))
    story.append(Spacer(1, 4 * mm))

    baseline_per_site = 10572
    baseline_total = total_sites * baseline_per_site
    modeled_total = data['total_payout']
    savings_abs = baseline_total - modeled_total
    savings_pct = (savings_abs / baseline_total * 100) if baseline_total else 0

    deriv_rows = [
        [P('Baseline payout', font=SANS, size=10, color=INK),
         P(f'{total_sites} sites x Rs 10,572/site', font=SANS, size=9, color=SLATE),
         P(rs_full(baseline_total), font=SANS, size=11, color=INK, align=TA_RIGHT)],
        [P('Modelled payout', font=SANS, size=10, color=INK),
         P('per optimised vendor plan', font=SANS, size=9, color=SLATE),
         P(rs_full(modeled_total), font=SANS, size=11, color=INK, align=TA_RIGHT)],
        [P('SSE Savings', font=SANS_BOLD, size=11, color=ACCENT),
         P(f'{savings_pct:.1f}% reduction', font=SANS_BOLD, size=10, color=ACCENT),
         P(rs_full(savings_abs), font=SANS_BOLD, size=13, color=ACCENT, align=TA_RIGHT)],
    ]
    deriv_table = Table(deriv_rows, colWidths=[CONTENT_W*0.30, CONTENT_W*0.40, CONTENT_W*0.30])
    deriv_table.setStyle(TableStyle([
        ('LINEABOVE', (0,0), (-1,0), 0.5, HAIRLINE),
        ('LINEBELOW', (0,0), (-1,-2), 0.25, HAIRLINE),
        ('LINEABOVE', (0,-1), (-1,-1), 1, ACCENT),
        ('LINEBELOW', (0,-1), (-1,-1), 1, ACCENT),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(deriv_table)

    # ============================================================
    # PAGE 4 — VENDOR EARNINGS & PROFIT DISTRIBUTION
    # ============================================================
    story.append(PageBreak())
    story.append(P('Vendor Earnings', font=SERIF, size=22, color=INK, leading=26))
    story.append(Spacer(1, 2 * mm))
    story.append(P(
        f'Monthly profit per vendor (payout minus fixed cost), by percentile across the pool. '
        f'P0 is the worst-paid vendor, P100 is the best.',
        font=SANS, size=10, color=SLATE
    ))
    story.append(Spacer(1, 8 * mm))

    def prof_cell(v):
        if v is None: return P('—', font=SANS, size=10, color=MUTED, align=TA_RIGHT)
        c = POSITIVE if v >= 0 else NEGATIVE
        return P(rs(v), font=SANS, size=10, color=c, align=TA_RIGHT)

    def pay_cell(v):
        if v is None: return P('—', font=SANS, size=10, color=MUTED, align=TA_RIGHT)
        return P(rs(v), font=SANS, size=10, color=INK, align=TA_RIGHT)

    earnings_header = [
        P('', font=SANS_BOLD, size=8, color=MUTED),
        P('2-INSTALL\nPAYOUT', font=SANS_BOLD, size=7, color=MUTED, align=TA_RIGHT),
        P('2-INSTALL\nPROFIT', font=SANS_BOLD, size=7, color=MUTED, align=TA_RIGHT),
        P('1-INSTALL\nPAYOUT', font=SANS_BOLD, size=7, color=MUTED, align=TA_RIGHT),
        P('1-INSTALL\nPROFIT', font=SANS_BOLD, size=7, color=MUTED, align=TA_RIGHT),
    ]
    earn_rows = [earnings_header]
    for label, p in [('Min (P0)', 0), ('P25', 25), ('Median (P50)', 50),
                     ('P75', 75), ('P90', 90), ('Max (P100)', 100)]:
        earn_rows.append([
            P(label, font=SANS, size=10, color=INK),
            pay_cell(pct(payouts_2i, p)),
            prof_cell(pct(profits_2i, p)),
            pay_cell(pct(payouts_1i, p)),
            prof_cell(pct(profits_1i, p)),
        ])
    earn_rows.append([
        P('Average', font=SANS_BOLD, size=10, color=INK),
        P(rs(avg_2i_payout), font=SANS_BOLD, size=11, color=INK, align=TA_RIGHT),
        P(rs(avg_2i_profit), font=SANS_BOLD, size=11, color=ACCENT, align=TA_RIGHT),
        P(rs(avg_1i_payout), font=SANS_BOLD, size=11, color=INK, align=TA_RIGHT),
        P(rs(avg_1i_profit), font=SANS_BOLD, size=11, color=ACCENT, align=TA_RIGHT),
    ])

    earn_table = Table(earn_rows, colWidths=[CONTENT_W*0.24] + [CONTENT_W*0.19]*4)
    earn_table.setStyle(TableStyle([
        ('LINEBELOW', (0,0), (-1,0), 0.75, INK),
        ('LINEBELOW', (0,1), (-1,-2), 0.25, HAIRLINE),
        ('LINEABOVE', (0,-1), (-1,-1), 0.75, INK),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 9),
        ('BOTTOMPADDING', (0,0), (-1,-1), 9),
    ]))
    story.append(earn_table)

    # Slippages box
    story.append(Spacer(1, 12 * mm))
    story.append(P('Slippages', font=SERIF, size=18, color=INK, leading=22))
    story.append(Spacer(1, 4 * mm))
    # Calculate 2i vs 1i slippages
    # Slips from 2i sites (DD): 2 * sum(pair_count) * sl_rate_2i (we don't have rate directly here)
    # We can approximate from total_slips and DD/SD site split
    sl_share_2i_sites = total_dd_sites / (total_dd_sites + total_sd_sites) if (total_dd_sites + total_sd_sites) else 0
    # If we don't have sl rates explicitly, approximate proportionally
    # Better: pull from the data['daily'] if surfaced. For now, estimate
    # Note: in practice slip rate on 2i and 1i could be different. We show the split if data has it.
    slips_from_2i = data.get('slips_from_2i', round(total_slips * sl_share_2i_sites))
    slips_from_1i = total_slips - slips_from_2i

    slip_data = [
        [P('SLIPS FROM 2-INSTALL', font=SANS_BOLD, size=7, color=MUTED),
         P('SLIPS FROM 1-INSTALL', font=SANS_BOLD, size=7, color=MUTED),
         P('TOTAL SLIPS', font=SANS_BOLD, size=7, color=MUTED)],
        [P(str(slips_from_2i), font=SERIF, size=26, color=INK),
         P(str(slips_from_1i), font=SERIF, size=26, color=INK),
         P(str(total_slips), font=SERIF, size=26, color=ACCENT)],
        [P(f'from {total_dd_sites} DD sites', font=SANS, size=8, color=SLATE),
         P(f'from {total_sd_sites} SD sites', font=SANS, size=8, color=SLATE),
         P('recovered across the month', font=SANS, size=8, color=SLATE)],
    ]
    slip_table = Table(slip_data, colWidths=[CONTENT_W/3]*3)
    slip_table.setStyle(TableStyle([
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LINEABOVE', (0,0), (-1,0), 0.5, HAIRLINE),
        ('LINEBELOW', (0,-1), (-1,-1), 0.5, HAIRLINE),
    ]))
    story.append(slip_table)

    # ============================================================
    # PAGE 5 — PER-VENDOR P&L
    # ============================================================
    story.append(PageBreak())
    story.append(P('Per-Vendor P&amp;L', font=SERIF, size=22, color=INK, leading=26))
    story.append(Spacer(1, 2 * mm))
    story.append(P(f'Detailed monthly economics for all {total_v} vendors.',
                   font=SANS, size=10, color=SLATE))
    story.append(Spacer(1, 6 * mm))

    header_row = [
        P('VENDOR', font=SANS_BOLD, size=7, color=MUTED),
        P('TYPE', font=SANS_BOLD, size=7, color=MUTED),
        P('DD', font=SANS_BOLD, size=7, color=MUTED, align=TA_RIGHT),
        P('SD', font=SANS_BOLD, size=7, color=MUTED, align=TA_RIGHT),
        P('SL', font=SANS_BOLD, size=7, color=MUTED, align=TA_RIGHT),
        P('IDLE', font=SANS_BOLD, size=7, color=MUTED, align=TA_RIGHT),
        P('SITES', font=SANS_BOLD, size=7, color=MUTED, align=TA_RIGHT),
        P('FIXED', font=SANS_BOLD, size=7, color=MUTED, align=TA_RIGHT),
        P('PAYOUT', font=SANS_BOLD, size=7, color=MUTED, align=TA_RIGHT),
        P('PROFIT', font=SANS_BOLD, size=7, color=MUTED, align=TA_RIGHT),
    ]
    pnl_rows = [header_row]
    for v in data['vendors']:
        c = POSITIVE if v['profit'] >= 0 else NEGATIVE
        pnl_rows.append([
            P(v['name'], font=SANS, size=9, color=INK),
            P(v['type'].replace('-Install', '-Inst'), font=SANS, size=9, color=SLATE),
            P(str(v['dd_days']), font=SANS, size=9, color=INK, align=TA_RIGHT),
            P(str(v['sd_sites']), font=SANS, size=9, color=INK, align=TA_RIGHT),
            P(str(v['sl_days']), font=SANS, size=9, color=INK, align=TA_RIGHT),
            P(str(v['idle_days']), font=SANS, size=9, color=MUTED, align=TA_RIGHT),
            P(str(v['sites']), font=SANS, size=9, color=INK, align=TA_RIGHT),
            P(rs(v['fixed_cost']), font=SANS, size=9, color=SLATE, align=TA_RIGHT),
            P(rs(v['payout']), font=SANS, size=9, color=INK, align=TA_RIGHT),
            P(rs(v['profit']), font=SANS_BOLD, size=10, color=c, align=TA_RIGHT),
        ])
    # Totals row
    tot_dd = sum(v['dd_days'] for v in data['vendors'])
    tot_sd = sum(v['sd_sites'] for v in data['vendors'])
    tot_sl = sum(v['sl_days'] for v in data['vendors'])
    tot_idle = sum(v['idle_days'] for v in data['vendors'])
    tot_sites = sum(v['sites'] for v in data['vendors'])
    pnl_rows.append([
        P('Total', font=SANS_BOLD, size=10, color=INK),
        P(f'{total_v} vendors', font=SANS, size=9, color=SLATE),
        P(str(tot_dd), font=SANS_BOLD, size=10, color=INK, align=TA_RIGHT),
        P(str(tot_sd), font=SANS_BOLD, size=10, color=INK, align=TA_RIGHT),
        P(str(tot_sl), font=SANS_BOLD, size=10, color=INK, align=TA_RIGHT),
        P(str(tot_idle), font=SANS_BOLD, size=10, color=MUTED, align=TA_RIGHT),
        P(str(tot_sites), font=SANS_BOLD, size=10, color=INK, align=TA_RIGHT),
        P(rs(data['total_cost']), font=SANS_BOLD, size=10, color=INK, align=TA_RIGHT),
        P(rs(data['total_payout']), font=SANS_BOLD, size=10, color=INK, align=TA_RIGHT),
        P(rs(data['total_profit']), font=SANS_BOLD, size=11,
          color=POSITIVE if data['total_profit'] >= 0 else NEGATIVE, align=TA_RIGHT),
    ])

    col_widths = [16, 17, 8, 8, 8, 9, 11, 22, 25, 28]
    total_units = sum(col_widths)
    col_widths = [w / total_units * CONTENT_W for w in col_widths]

    pnl_table = Table(pnl_rows, colWidths=col_widths, repeatRows=1)
    pnl_table.setStyle(TableStyle([
        ('LINEBELOW', (0,0), (-1,0), 0.75, INK),
        ('LINEBELOW', (0,1), (-1,-2), 0.25, HAIRLINE),
        ('LINEABOVE', (0,-1), (-1,-1), 0.75, INK),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(pnl_table)

    story.append(Spacer(1, 4 * mm))
    story.append(P(
        'DD = double install day (2 sites). SD = single install day. SL = slip recovery. '
        'Profit = Payout - Fixed cost.',
        font=SANS, size=8, color=MUTED
    ))

    # ============================================================
    # PAGE 6 — INSTALL CALENDAR (end-of-month skew visible)
    # ============================================================
    story.append(PageBreak())
    story.append(P('Install Calendar', font=SERIF, size=22, color=INK, leading=26))
    story.append(Spacer(1, 2 * mm))
    story.append(P(
        f'Daily roster across all {total_v} vendors. '
        f'D = double install, S = single install, s = slip recovery, dot = rest. '
        f'Peak day (D{peak_day+1}) highlighted in blue.',
        font=SANS, size=10, color=SLATE
    ))
    story.append(Spacer(1, 6 * mm))

    roster = data['roster']
    n_vendors = len(roster)

    cal_header = [P('DAY', font=SANS_BOLD, size=7, color=PAPER, align=TA_LEFT)]
    cal_header.append(P('SITES', font=SANS_BOLD, size=7, color=PAPER, align=TA_RIGHT))
    for v in data['vendors']:
        cal_header.append(P(v['name'], font=SANS_BOLD, size=6, color=PAPER, align=TA_CENTER))
    cal_rows = [cal_header]

    for d in range(n_days):
        is_peak = (d == peak_day)
        row = [
            P(f'D{d+1}', font=SANS_BOLD if is_peak else SANS,
              size=7, color=ACCENT if is_peak else SLATE),
            P(str(daily[d]), font=SANS_BOLD if is_peak else SANS,
              size=7, color=ACCENT if is_peak else INK, align=TA_RIGHT),
        ]
        for vi in range(n_vendors):
            cell = roster[vi][d]
            kind = cell.get('type', 'idle')
            if kind == 'DD':
                row.append(P('D', font=SANS_BOLD, size=8, color=ACCENT, align=TA_CENTER))
            elif kind == 'SD':
                row.append(P('S', font=SANS, size=8, color=INK, align=TA_CENTER))
            elif kind == 'SL':
                row.append(P('s', font=SANS_BOLD, size=8, color=NEGATIVE, align=TA_CENTER))
            else:
                row.append(P('·', font=SANS, size=8, color=MUTED, align=TA_CENTER))
        cal_rows.append(row)

    day_col = 9 * mm
    sites_col = 11 * mm
    vendor_col = (CONTENT_W - day_col - sites_col) / max(1, n_vendors)
    col_widths_cal = [day_col, sites_col] + [vendor_col] * n_vendors

    cal_table = Table(cal_rows, colWidths=col_widths_cal, repeatRows=1)
    cstyle = [
        ('BACKGROUND', (0,0), (-1,0), ACCENT),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,0), (-1,-1), 2),
        ('RIGHTPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LINEBELOW', (0,0), (-1,0), 0.5, ACCENT),
        ('LINEBELOW', (0,1), (-1,-1), 0.15, HAIRLINE),
        # Highlight peak row left rail
        ('BACKGROUND', (0, peak_day+1), (1, peak_day+1), colors.HexColor('#eef0ff')),
    ]
    cal_table.setStyle(TableStyle(cstyle))
    story.append(cal_table)

    # ---- Page footer decoration ----
    def add_page_decoration(canvas, doc_):
        canvas.saveState()
        canvas.setStrokeColor(ACCENT)
        canvas.setLineWidth(1.5)
        canvas.line(MARGIN, PAGE_H - 12*mm, MARGIN + 28*mm, PAGE_H - 12*mm)
        canvas.setStrokeColor(HAIRLINE)
        canvas.setLineWidth(0.3)
        canvas.line(MARGIN, 14*mm, PAGE_W - MARGIN, 14*mm)
        canvas.setFont(SANS, 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN, 10*mm, 'SolarSquare Vendor Plan')
        canvas.drawCentredString(PAGE_W/2, 10*mm, city_name)
        canvas.drawRightString(PAGE_W - MARGIN, 10*mm, f'{doc_.page}  ·  {date_str}')
        canvas.restoreState()

    doc.build(story, onFirstPage=add_page_decoration, onLaterPages=add_page_decoration)
    return buf.getvalue()



@app.route('/report', methods=['POST'])
def report_route():
    if not REPORTLAB_OK:
        return jsonify({'ok': False, 'reason': 'reportlab not installed on server. Run: pip3 install reportlab --break-system-packages'}), 500
    try:
        body = request.get_json()
        city = (body.get('city') or '').strip() or 'Unspecified City'
        data = body.get('data')
        if not data or not data.get('ok'):
            return jsonify({'ok': False, 'reason': 'no solution data provided'}), 400
        pdf_bytes = build_pdf_report(data, city)
        safe_city = ''.join(c for c in city if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_') or 'city'
        date_str = datetime.datetime.now().strftime('%Y%m%d')
        filename = f'SolarSquare_VendorPlan_{safe_city}_{date_str}.pdf'
        return Response(
            pdf_bytes,
            mimetype='application/pdf',
            headers={'Content-Disposition': f'attachment; filename="{filename}"'}
        )
    except Exception as e:
        tb = tb_mod.format_exc()
        print(f'[/report] ERROR: {tb}')
        return jsonify({'ok': False, 'reason': str(e), 'traceback': tb}), 500


@app.route('/')
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'index.html')


@app.route('/health')
def health():
    return jsonify({'ortools_loaded': ORTOOLS_OK, 'ortools_error': ORTOOLS_ERR})


@app.route('/clusters')
def clusters_route():
    """Return cluster catalog for the UI dropdown."""
    catalog = []
    for name, info in sorted(CLUSTER_DATA.items()):
        catalog.append({
            'name': name,
            'avg_monthly': info['avg_monthly'],
            'april_2026': info.get('april_2026', info['avg_monthly']),
            'pan_india_skew': info.get('pan_india_skew', 70),
            'baseline_per_site': info.get('baseline_per_site', 10572),
            'total_12mo': info['total_12mo'],
            'months_used': info['months_used'],
            'slab_mix': info['slab_mix'],
            'is_noisy': info['months_used'] <= 2,
        })
    return jsonify({'clusters': catalog})


# ============================================================================
# PAN-INDIA PRECOMPUTE & CACHE
# ============================================================================
# Runs solver for every cluster at startup, caches results.
# All clusters use: P75 demand, cluster-specific skew (60/70/80 by data quality),
# April 2026 volume, fixed slip rates (2i=10%, 1i=20%), default slab rates.
# Cache is in-memory; cleared on app restart.
PAN_INDIA_CACHE = {
    'status': 'not_started',  # 'not_started', 'computing', 'ready', 'error'
    'progress': 0,             # 0-100
    'current_cluster': None,
    'results': [],             # list of per-cluster result dicts
    'started_at': None,
    'completed_at': None,
}

# Default inputs for pan-India runs (cluster-independent)
# Guardrails per AOP framing:
#   - Daily demand confidence: P75 (no cluster runs below P50)
#   - Peak shaving: max 30% across all clusters (pan_india_skew = 70 = 30% shaving)
#   - Slip rates: 2i = 10%, 1i = 20% (conservative for AOP commitments)
#   - Per-cluster slab mix and baseline auto-loaded from clusters.py
PAN_INDIA_DEFAULTS = {
    'days': 30,
    'peak_ratio': 1.5,           # P75 (between 1.45 and 1.55)
    'sl2_rate': 0.10,             # 2i slip rate
    'sl1_rate': 0.20,             # 1i slip rate
    'elig_pct': 0.50,             # 2i eligibility
    'target_pxx': 0.75,
    'slab_rates': [8000, 8500, 10000, 15000],
    'slab_rates_1i': [8000, 8500, 10000, 15000],
    'dd_elig_slabs': [True, True, False, False],
    'cost_2i': 180000,
    'cost_1i': 140000,
    'dd_discount': 0.7,
    'baseline_per_site': 10572,
    'time_limit_sec': 30,
    'max_working_days': 26,
}


def compute_pan_india_for_cluster(cluster_name):
    """Run the solver for one cluster with pan-India defaults."""
    info = CLUSTER_DATA[cluster_name]
    volume = info.get('april_2026', info['avg_monthly'])
    if volume < 30:
        print(f'[pan-india/{cluster_name}] SKIPPED: only {volume} installs in April')
        return {
            'cluster': cluster_name,
            'status': 'skipped_low_volume',
            'reason': f'Only {volume} installs in April — too low for stable solve',
            'volume': volume,
        }

    params = dict(PAN_INDIA_DEFAULTS)
    params['total_sites'] = volume
    params['cluster'] = cluster_name
    params['skew_pct'] = info.get('pan_india_skew', 70)
    # Use cluster's actual slab mix
    params['slab_mix'] = info['slab_mix']
    # Use cluster's actual baseline Rs/site (P50 of Jan-Mar 2026 actuals)
    cluster_baseline = info.get('baseline_per_site', PAN_INDIA_DEFAULTS['baseline_per_site'])
    params['baseline_per_site'] = cluster_baseline
    # DEEP MODE: pan-India runs in background and can afford to search much harder
    # for the truly optimal solution per cluster. Override iteration/time caps.
    params['_deep_mode'] = True

    # Detailed pre-solve log so you can verify inputs
    print(f'\n[pan-india/{cluster_name}] ============================================')
    print(f'[pan-india/{cluster_name}] INPUT PARAMS:')
    print(f'[pan-india/{cluster_name}]   total_sites:    {params["total_sites"]} (April 2026)')
    print(f'[pan-india/{cluster_name}]   days:           {params["days"]}')
    print(f'[pan-india/{cluster_name}]   peak_ratio:     {params["peak_ratio"]} (P75)')
    print(f'[pan-india/{cluster_name}]   skew_pct:       {params["skew_pct"]}% (data quality: {info["months_used"]} months)')
    print(f'[pan-india/{cluster_name}]   slab_mix:       S1={params["slab_mix"][0]} S2={params["slab_mix"][1]} S3={params["slab_mix"][2]} S4={params["slab_mix"][3]}')
    print(f'[pan-india/{cluster_name}]   slab_rates_2i:  {params["slab_rates"]}')
    print(f'[pan-india/{cluster_name}]   slab_rates_1i:  {params["slab_rates_1i"]}')
    print(f'[pan-india/{cluster_name}]   dd_elig_slabs:  S1={params["dd_elig_slabs"][0]} S2={params["dd_elig_slabs"][1]} S3={params["dd_elig_slabs"][2]} S4={params["dd_elig_slabs"][3]}')
    print(f'[pan-india/{cluster_name}]   elig_pct:       {params["elig_pct"]*100}%')
    print(f'[pan-india/{cluster_name}]   sl2_rate:       {params["sl2_rate"]*100}%  (2i slip)')
    print(f'[pan-india/{cluster_name}]   sl1_rate:       {params["sl1_rate"]*100}%  (1i slip)')
    print(f'[pan-india/{cluster_name}]   cost_2i:        Rs {params["cost_2i"]:,.0f}')
    print(f'[pan-india/{cluster_name}]   cost_1i:        Rs {params["cost_1i"]:,.0f}')
    print(f'[pan-india/{cluster_name}]   dd_discount:    {params["dd_discount"]}')
    print(f'[pan-india/{cluster_name}]   baseline/site:  Rs {cluster_baseline:,.0f}  (P50 of Jan-Mar 2026 actuals)')
    print(f'[pan-india/{cluster_name}]   max_work_days:  {params["max_working_days"]}')

    import time as _time
    t_start = _time.time()
    try:
        result = run_solver(params)
        elapsed = _time.time() - t_start
        if result is None or result.get('_diagnostic_only'):
            print(f'[pan-india/{cluster_name}] INFEASIBLE after {elapsed:.1f}s')
            return {
                'cluster': cluster_name,
                'status': 'infeasible',
                'volume': volume,
                'skew': params['skew_pct'],
            }
        # Extract just what we need for the pan-India table
        vendors = result.get('vendors', [])
        v2 = result.get('v2', 0)
        v1 = result.get('v1', 0)
        # Profit distributions
        profits_2i = sorted([v['profit'] for v in vendors if v['type'] == '2-Install'])
        profits_1i = sorted([v['profit'] for v in vendors if v['type'] == '1-Install'])
        def pxx(arr, p):
            if not arr:
                return 0
            idx = (p / 100) * (len(arr) - 1)
            lo = int(idx); hi = min(lo+1, len(arr)-1)
            frac = idx - lo
            return arr[lo] + (arr[hi] - arr[lo]) * frac
        dist_2i = {f'P{p}': round(pxx(profits_2i, p)) for p in [0, 25, 50, 75, 90, 100]}
        dist_1i = {f'P{p}': round(pxx(profits_1i, p)) for p in [0, 25, 50, 75, 90, 100]}
        avg_2i = round(sum(profits_2i) / len(profits_2i)) if profits_2i else 0
        avg_1i = round(sum(profits_1i) / len(profits_1i)) if profits_1i else 0
        # Compute SSE savings using THIS CLUSTER's baseline (not the global default)
        total_payout = result.get('total_payout', 0)
        baseline_total = volume * cluster_baseline
        savings = baseline_total - total_payout
        savings_pct = (savings / baseline_total * 100) if baseline_total > 0 else 0

        # Detailed result log
        print(f'[pan-india/{cluster_name}] RESULT after {elapsed:.1f}s:')
        print(f'[pan-india/{cluster_name}]   Vendors:        {v2} (2i) + {v1} (1i) = {v2+v1} total')
        print(f'[pan-india/{cluster_name}]   Modelled cost:  Rs {total_payout:,.0f}')
        print(f'[pan-india/{cluster_name}]   Baseline cost:  Rs {baseline_total:,.0f}')
        print(f'[pan-india/{cluster_name}]   Savings:        Rs {savings:,.0f} ({savings_pct:.1f}%)')
        if profits_2i:
            print(f'[pan-india/{cluster_name}]   2i profits:     min=Rs {min(profits_2i):,.0f} median=Rs {profits_2i[len(profits_2i)//2]:,.0f} max=Rs {max(profits_2i):,.0f}')
        if profits_1i:
            print(f'[pan-india/{cluster_name}]   1i profits:     min=Rs {min(profits_1i):,.0f} median=Rs {profits_1i[len(profits_1i)//2]:,.0f} max=Rs {max(profits_1i):,.0f}')

        # Sanity: profitability check
        unprofitable_vendors = [(v['name'], v['profit']) for v in vendors if v['profit'] < 0]
        unprofitable_amount = sum(-p for _, p in unprofitable_vendors)
        all_profitable = len(unprofitable_vendors) == 0
        if not all_profitable:
            print(f'[pan-india/{cluster_name}]   ⚠ UNPROFITABLE VENDORS: {unprofitable_vendors}')
            print(f'[pan-india/{cluster_name}]   ⚠ Total loss: Rs {unprofitable_amount:,.0f}')

        # Sanity: total payout matches sum of vendor payouts
        sum_vendor_payouts = sum(v['payout'] for v in vendors)
        if abs(sum_vendor_payouts - total_payout) > 1:
            print(f'[pan-india/{cluster_name}]   ⚠ PAYOUT MISMATCH: result={total_payout} vs sum(vendor.payout)={sum_vendor_payouts}')

        # Determine final status:
        # - 'ok' only if ALL vendors are profitable
        # - 'partial' if some vendors lose money (solver returned a degraded solution)
        final_status = 'ok' if all_profitable else 'partial'
        if not all_profitable:
            print(f'[pan-india/{cluster_name}] STATUS: PARTIAL (some vendors unprofitable — solution not viable)')

        return {
            'cluster': cluster_name,
            'status': final_status,
            'volume': volume,
            'skew': params['skew_pct'],
            'v2': v2,
            'v1': v1,
            'total_v': v2 + v1,
            'total_payout': round(total_payout),
            'baseline_cost': round(baseline_total),
            'savings_rs': round(savings),
            'savings_pct': round(savings_pct, 1),
            'dist_2i': dist_2i,
            'dist_1i': dist_1i,
            'avg_2i': avg_2i,
            'avg_1i': avg_1i,
            'months_used': info['months_used'],
            'slab_mix': info['slab_mix'],
            'is_noisy': info['months_used'] <= 2,
            'elapsed_sec': round(elapsed, 2),
            'all_profitable': all_profitable,
            'unprofitable_count': len(unprofitable_vendors),
            'unprofitable_amount': round(unprofitable_amount),
        }
    except Exception as e:
        elapsed = _time.time() - t_start
        print(f'[pan-india/{cluster_name}] ERROR after {elapsed:.1f}s: {e}')
        import traceback
        traceback.print_exc()
        return {
            'cluster': cluster_name,
            'status': 'error',
            'error': str(e)[:200],
            'volume': volume,
        }


def compute_pan_india_all():
    """Run all clusters. Called at startup in a background thread."""
    global PAN_INDIA_CACHE
    PAN_INDIA_CACHE['status'] = 'computing'
    PAN_INDIA_CACHE['started_at'] = datetime.datetime.now().isoformat()
    PAN_INDIA_CACHE['results'] = []
    PAN_INDIA_CACHE['progress'] = 0
    cluster_names = sorted(CLUSTER_DATA.keys())
    n = len(cluster_names)
    for idx, name in enumerate(cluster_names):
        PAN_INDIA_CACHE['current_cluster'] = name
        print(f'[pan-india] [{idx+1}/{n}] Solving {name}...')
        res = compute_pan_india_for_cluster(name)
        PAN_INDIA_CACHE['results'].append(res)
        PAN_INDIA_CACHE['progress'] = round((idx + 1) / n * 100)
        print(f'[pan-india] [{idx+1}/{n}] {name}: {res.get("status")} '
              f'(v={res.get("total_v", "-")}, savings={res.get("savings_pct", "-")}%)')
    PAN_INDIA_CACHE['status'] = 'ready'
    PAN_INDIA_CACHE['current_cluster'] = None
    PAN_INDIA_CACHE['completed_at'] = datetime.datetime.now().isoformat()
    # Compute totals
    # Only 'ok' clusters count toward the headline savings number (partial clusters
    # have unprofitable vendors and aren't real solutions).
    ok = [r for r in PAN_INDIA_CACHE['results'] if r['status'] == 'ok']
    partial = [r for r in PAN_INDIA_CACHE['results'] if r['status'] == 'partial']
    infeasible = [r for r in PAN_INDIA_CACHE['results'] if r['status'] == 'infeasible']
    skipped = [r for r in PAN_INDIA_CACHE['results'] if r['status'] == 'skipped_low_volume']
    errored = [r for r in PAN_INDIA_CACHE['results'] if r['status'] == 'error']

    total_vendors = sum(r['total_v'] for r in ok)
    total_volume = sum(r['volume'] for r in ok)
    total_payout = sum(r['total_payout'] for r in ok)
    total_baseline = sum(r['baseline_cost'] for r in ok)
    savings_pct = ((total_baseline - total_payout) / total_baseline * 100) if total_baseline > 0 else 0

    # Also compute "with partial" totals so the UI can show the full picture
    ok_and_partial = ok + partial
    total_vendors_with_partial = sum(r['total_v'] for r in ok_and_partial)
    total_volume_with_partial = sum(r['volume'] for r in ok_and_partial)
    total_unprofitable = sum(r.get('unprofitable_amount', 0) for r in partial)

    PAN_INDIA_CACHE['totals'] = {
        'clusters_ok': len(ok),
        'clusters_partial': len(partial),
        'clusters_infeasible': len(infeasible),
        'clusters_skipped': len(skipped),
        'clusters_error': len(errored),
        'clusters_total': n,
        'total_vendors': total_vendors,
        'total_vendors_with_partial': total_vendors_with_partial,
        'total_volume': total_volume,
        'total_volume_with_partial': total_volume_with_partial,
        'total_payout': round(total_payout),
        'total_baseline': round(total_baseline),
        'total_savings_rs': round(total_baseline - total_payout),
        'total_savings_pct': round(savings_pct, 1),
        'total_unprofitable_amount': round(total_unprofitable),
    }
    print(f'[pan-india] DONE. {len(ok)}/{n} fully solved, {len(partial)} partial, '
          f'{len(infeasible)} infeasible, {len(skipped)} skipped, {len(errored)} errored. '
          f'Headline: {total_vendors} vendors, {savings_pct:.1f}% SSE savings.')


@app.route('/pan_india')
def pan_india_route():
    """Return cached pan-India results."""
    return jsonify(PAN_INDIA_CACHE)


@app.route('/pan_india/refresh', methods=['POST'])
def pan_india_refresh_route():
    """Trigger a recompute (in background)."""
    import threading
    if PAN_INDIA_CACHE['status'] == 'computing':
        return jsonify({'ok': False, 'reason': 'Already computing'}), 200
    t = threading.Thread(target=compute_pan_india_all, daemon=True)
    t.start()
    return jsonify({'ok': True, 'message': 'Pan-India compute started in background'})


@app.route('/solve', methods=['POST'])
def solve_route():
    if not ORTOOLS_OK:
        return jsonify({
            'ok': False,
            'reason': f'OR-Tools missing: {ORTOOLS_ERR}'
        }), 200
    try:
        params = request.get_json(force=True, silent=False)
        if params is None:
            return jsonify({'ok': False, 'reason': 'No JSON body'}), 200
        print(f'[/solve] received request')
        result = run_solver(params)
        return jsonify(result), 200
    except Exception as e:
        tb = tb_mod.format_exc()
        print('=' * 60)
        print('ERROR in /solve:')
        print(tb)
        print('=' * 60)
        return jsonify({
            'ok': False, 'reason': f'{type(e).__name__}: {e}', 'traceback': tb,
        }), 200


@app.errorhandler(404)
def handle_404(e):
    return jsonify({'ok': False, 'reason': f'404: {request.path}'}), 200


@app.errorhandler(500)
def handle_500(e):
    return jsonify({'ok': False, 'reason': f'500: {e}', 'traceback': tb_mod.format_exc()}), 200


if __name__ == '__main__':
    print('=' * 60)
    print('SolarSquare CP-SAT Solver (v34 — FAST)')
    print(f'OR-Tools loaded: {ORTOOLS_OK}')
    if not ORTOOLS_OK:
        print(f'  Error: {ORTOOLS_ERR}')
    print('Open http://localhost:5000 in browser')
    print('=' * 60)

    # Auto-trigger pan-India compute in background so cache is warm for demos
    if ORTOOLS_OK and os.environ.get('SKIP_PAN_INDIA') != '1':
        import threading
        print('[startup] Launching pan-India precompute in background...')
        print('[startup] (Set SKIP_PAN_INDIA=1 env to skip this)')
        threading.Thread(target=compute_pan_india_all, daemon=True).start()

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
