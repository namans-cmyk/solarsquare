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
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False
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


def bell_curve_demand(total, days, peak_ratio):
    if peak_ratio <= 1.001:
        base = total // days
        rem = total - base * days
        arr = [base] * days
        mid = days // 2
        for i in range(rem):
            arr[(i + mid) % days] += 1
        return arr
    mu = (days - 1) / 2
    peak_value = round((total / days) * peak_ratio)
    sigma = days / (2 + (peak_ratio - 1) * 4)
    raw = [math.exp(-0.5 * ((i - mu) / sigma) ** 2) for i in range(days)]
    max_raw = max(raw)
    arr = [round(x * peak_value / max_raw) for x in raw]
    diff = total - sum(arr)
    peak_idx = round(mu)
    order = sorted((i for i in range(days) if i != peak_idx),
                   key=lambda i: (abs(i - mu), i))
    if diff > 0:
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


def monte_carlo_slips(pair_count, sd_by_day, sl2, sl1, percentile, runs=500, seed=42):
    rng = random.Random(seed)
    days = len(pair_count)
    dd_sites = [p * 2 for p in pair_count]
    total_slips = round(sum(dd_sites) * sl2 + sum(sd_by_day) * sl1)
    weights = []
    w_sum = 0.0
    for d in range(days - 1):
        w = dd_sites[d] * sl2 + sd_by_day[d] * sl1
        weights.append(w)
        w_sum += w
    if w_sum == 0:
        return {'total_slips': 0, 'expected_sl': [0]*days, 'pxx_sl': [0]*days}
    cum = []
    acc = 0.0
    for w in weights:
        acc += w / w_sum
        cum.append(acc)
    expected_sl = [0.0] * days
    for d in range(days - 1):
        expected_sl[d + 1] = total_slips * weights[d] / w_sum
    daily_samples = [[] for _ in range(days)]
    for _ in range(runs):
        day_counts = [0] * days
        for _s in range(total_slips):
            u = rng.random()
            src = days - 2
            for i, c in enumerate(cum):
                if u <= c:
                    src = i
                    break
            day_counts[src + 1] += 1
        for d in range(days):
            daily_samples[d].append(day_counts[d])
    pxx_sl = []
    for arr in daily_samples:
        arr.sort()
        idx = min(len(arr) - 1, int(len(arr) * percentile))
        pxx_sl.append(arr[idx])
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

    # HARD CONSTRAINT: total monthly SL must equal the deterministic slip count.
    # All 87 (or whatever) slips must get recovered somewhere in the month.
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

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = 8
    solver.parameters.stop_after_first_solution = True

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
                      cost_2i, cost_1i, time_limit_sec, max_working_days):
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

    # Wall-clock cap: stop searching after time_limit_sec
    import time
    search_start = time.time()
    HARD_CAP = float(time_limit_sec)
    print(f'[solver] Hard cap: {HARD_CAP}s wall clock')

    # Tight budgets to fit within HARD_CAP. Phase A: 3s feasibility, Phase B: 12s optimization.
    per_call_time = 3

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
                # Wall-clock cap
                elapsed = time.time() - search_start
                if elapsed > HARD_CAP:
                    print(f'[solver] Wall-clock cap reached: {elapsed:.1f}s > {HARD_CAP}s')
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
                while sum(sl_needed) < mc['total_slips']:
                    bumped = False
                    for d in sorted(range(days), key=lambda i: -mc['pxx_sl'][i]):
                        slack = v2 + v1 - pair_count[d] - sd_count[d]
                        if sl_needed[d] < slack:
                            sl_needed[d] += 1
                            bumped = True
                            break
                    if not bumped:
                        break
                # min_working_days: force balanced workload.
                # Total work = DD + SD + SL = sum(pair_count) + sum(sd_count) + sum(sl_needed)
                # Min per vendor = floor((total_work - max_slack) / total_v)
                # where max_slack lets some vendors work slightly more than others.
                # Simpler: aim for ~90% of avg, capped at max_working_days
                total_work = sum(pair_count) + sum(sd_count) + sum(sl_needed)
                avg_work = total_work / (v2 + v1)
                min_working_days = max(0, int(avg_work * 0.9))
                # Don't exceed max
                min_working_days = min(min_working_days, max_working_days - 2)
                print(f'[solver] iter {iter_count}: v2={v2}, v1={v1}, pxx={pxx}, target_total_sl={mc["total_slips"]}, min_work={min_working_days}, max_work={max_working_days}')
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
                    # Phase B per candidate is QUICK — just enough to differentiate candidates.
                    # The winning candidate gets a full polish at the end.
                    remaining = HARD_CAP - (time.time() - search_start)
                    phase_b_time = min(5, max(2, int((remaining - 30) / 3)))  # save 30s for polish
                    print(f'[solver]   Phase A profitable. Running quick Phase B ({phase_b_time}s)...')
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

                candidate = {
                    'v2': v2, 'v1': v1, 'roster': roster, 'status': status,
                    'pxx_achieved': pxx, 'sl_needed': sl_needed, 'mc': mc, 'payouts': payouts,
                    'all_profitable': all_prof, 'unprofitable_amount': unprof_sum,
                    'activity': activity,  # for polish phase
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
                    # Check if this improves over current best
                    if profitable_candidates:
                        current_best = min(profitable_candidates, key=lambda c: (c['v2']+c['v1'], -c['min_profit']))
                        current_best_score = (current_best['v2']+current_best['v1'], -current_best['min_profit'])
                        new_score = (v2 + v1, -min_overall)
                        if new_score < current_best_score:
                            iterations_since_improvement = 0
                        else:
                            iterations_since_improvement += 1
                    else:
                        iterations_since_improvement = 0
                    profitable_candidates.append(candidate)
                    print(f'[solver]   PROFITABLE candidate: v2={v2}, v1={v1}, pxx={pxx}, min_profit=₹{min_overall:.0f} (no_improve_streak={iterations_since_improvement})')
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
        # Sort by: (1) fewest vendors, (2) highest min profit, (3) tightest spread
        def candidate_score(c):
            spread = max(c['payouts'][:c['v2']]) - min(c['payouts'][:c['v2']]) if c['v2'] > 0 else 0
            return (c['v2'] + c['v1'], -c['min_profit'], spread)
        profitable_candidates.sort(key=candidate_score)
        best = profitable_candidates[0]
        print(f'[solver] Searched {iter_count} iterations. Found {len(profitable_candidates)} profitable candidates.')
        print(f'[solver] Initial pick: v2={best["v2"]}, v1={best["v1"]}, min_profit=₹{best["min_profit"]:.0f}')

        # POLISH PHASE: spend remaining budget squeezing the spread on the winner.
        remaining = HARD_CAP - (time.time() - search_start)
        polish_time = max(5, int(remaining - 3))
        if polish_time >= 5 and 'activity' in best:
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
    return None


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

    daily = bell_curve_demand(total_sites, days, peak_ratio)
    dd_pairs, sd_slabs = compute_daily_demand(daily, slab_mix, dd_elig_slabs, elig_pct)
    pair_count = [len(p) for p in dd_pairs]
    sd_count = [sum(s) for s in sd_slabs]
    peak_day = daily.index(max(daily))

    print(f'[solver] daily sum={sum(daily)} expected={total_sites}, peak={peak_day}, max_work_days={max_working_days}')
    print(f'[solver] 2i rates: {slab_rates_2i}')
    print(f'[solver] 1i rates: {slab_rates_1i}')

    result = find_optimal_fast(daily, dd_pairs, sd_slabs, pair_count, sd_count,
                               peak_day, sl2_rate, sl1_rate, target_pxx, slab_rates_2i, slab_rates_1i,
                               dd_discount, cost_2i, cost_1i, time_limit, max_working_days)

    if result is None:
        return {
            'ok': False,
            'reason': 'No profitable solution found AND no feasible schedule found. At this slip rate + cost combination, the math doesn\'t work. Try: lowering slip%, raising DD discount, lowering 2i fixed cost, or lowering peak ratio.',
            'daily': daily,
            'pair_count_per_day': pair_count,
            'sd_count_per_day': sd_count,
            'peak_day': peak_day,
            'total_slips': round(sum([p*2 for p in pair_count]) * sl2_rate + sum(sd_count) * sl1_rate),
        }

    v2 = result['v2']
    v1 = result['v1']
    total_v = v2 + v1
    roster = result['roster']

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
        'total_slips': result['mc']['total_slips'],
        'pxx_target': target_pxx,
        'pxx_achieved': result['pxx_achieved'],
        'all_attempts': result.get('all_attempts', []),
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
    """Generate a PDF report from solver result data. Returns bytes."""
    if not REPORTLAB_OK:
        raise RuntimeError('reportlab not installed. Run: pip3 install reportlab --break-system-packages')

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=landscape(A4),
        leftMargin=12 * mm, rightMargin=12 * mm,
        topMargin=12 * mm, bottomMargin=12 * mm,
        title=f'SolarSquare Vendor Plan — {city_name}'
    )

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle('h1', parent=styles['Heading1'], textColor=BRAND_COLOR,
                        fontSize=22, spaceAfter=4, leading=26)
    h2 = ParagraphStyle('h2', parent=styles['Heading2'], textColor=BRAND_COLOR,
                        fontSize=13, spaceAfter=6, spaceBefore=10, leading=16)
    body = ParagraphStyle('body', parent=styles['BodyText'], fontSize=9, leading=12)
    sub = ParagraphStyle('sub', parent=styles['BodyText'], fontSize=9, textColor=colors.HexColor('#666666'),
                         spaceAfter=8)
    small = ParagraphStyle('small', parent=styles['BodyText'], fontSize=8, textColor=colors.HexColor('#888888'))

    story = []

    # ---- HEADER ----
    story.append(Paragraph(f'SolarSquare Vendor Plan — {city_name}', h1))
    now = datetime.datetime.now().strftime('%d %B %Y, %H:%M')
    story.append(Paragraph(f'Generated {now}  |  Status: {data["status"]}  |  Pxx achieved: P{round(data["pxx_achieved"]*100)}', sub))

    # ---- KPI CARDS (top row) ----
    def fmt_inr(v):
        v = abs(v); 
        if v >= 1e7: return f'₹{v/1e7:.2f}Cr'
        if v >= 1e5: return f'₹{v/1e5:.2f}L'
        if v >= 1e3: return f'₹{v/1e3:.0f}K'
        return f'₹{v:.0f}'

    kpi_data = [[
        Paragraph(f'<b>{data["v2"]}</b>', ParagraphStyle('kn', fontSize=20, textColor=BRAND_COLOR, alignment=TA_CENTER)),
        Paragraph(f'<b>{data["v1"]}</b>', ParagraphStyle('kn', fontSize=20, textColor=BRAND_COLOR, alignment=TA_CENTER)),
        Paragraph(f'<b>{data["v2"]+data["v1"]}</b>', ParagraphStyle('kn', fontSize=20, textColor=BRAND_COLOR, alignment=TA_CENTER)),
        Paragraph(f'<b>{fmt_inr(data["total_payout"])}</b>', ParagraphStyle('kn', fontSize=20, textColor=BRAND_COLOR, alignment=TA_CENTER)),
        Paragraph(f'<b>{data["savings_pct"]}%</b>', ParagraphStyle('kn', fontSize=20, textColor=ACCENT_GREEN, alignment=TA_CENTER)),
        Paragraph(f'<b>{data["total_slips"]}</b>', ParagraphStyle('kn', fontSize=20, textColor=BRAND_COLOR, alignment=TA_CENTER)),
    ], [
        Paragraph('2-Install vendors', ParagraphStyle('kl', fontSize=8, textColor=colors.grey, alignment=TA_CENTER)),
        Paragraph('1-Install vendors', ParagraphStyle('kl', fontSize=8, textColor=colors.grey, alignment=TA_CENTER)),
        Paragraph('Total vendors', ParagraphStyle('kl', fontSize=8, textColor=colors.grey, alignment=TA_CENTER)),
        Paragraph('Modelled payout', ParagraphStyle('kl', fontSize=8, textColor=colors.grey, alignment=TA_CENTER)),
        Paragraph('SSE savings', ParagraphStyle('kl', fontSize=8, textColor=colors.grey, alignment=TA_CENTER)),
        Paragraph('Total slips/mo', ParagraphStyle('kl', fontSize=8, textColor=colors.grey, alignment=TA_CENTER)),
    ]]
    kpi_table = Table(kpi_data, colWidths=[44*mm]*6, rowHeights=[14*mm, 6*mm])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BRAND_LIGHT),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOX', (0,0), (-1,-1), 0.5, BRAND_COLOR),
        ('LINEAFTER', (0,0), (-2,-1), 0.3, colors.white),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(kpi_table)

    # Profit distribution by percentile (replaces the simple avg/min strip)
    story.append(Spacer(1, 6))

    def pct(arr, p):
        if not arr:
            return None
        s = sorted(arr)
        if p <= 0: return s[0]
        if p >= 100: return s[-1]
        idx = (p / 100.0) * (len(s) - 1)
        lo = int(idx)
        hi = lo + 1 if lo + 1 < len(s) else lo
        return s[lo] + (s[hi] - s[lo]) * (idx - lo)

    profits_2i = [v['profit'] for v in data['vendors'] if v['type'] == '2-Install']
    profits_1i = [v['profit'] for v in data['vendors'] if v['type'] == '1-Install']
    avg_2i = sum(profits_2i) / len(profits_2i) if profits_2i else None
    avg_1i = sum(profits_1i) / len(profits_1i) if profits_1i else None

    def cell(v):
        if v is None: return '—'
        if v < 0: return f'<font color="red">-{fmt_inr(abs(v))}</font>'
        return f'<font color="green">{fmt_inr(v)}</font>'

    pct_styles = ParagraphStyle('pctc', fontSize=9, alignment=TA_RIGHT)
    pct_rows = [['Percentile', '2-Install', '1-Install']]
    for label, p in [('P0 (min)', 0), ('P25', 25), ('P50 (median)', 50), ('P75', 75), ('P90', 90), ('P100 (max)', 100)]:
        pct_rows.append([
            label,
            Paragraph(cell(pct(profits_2i, p)), pct_styles),
            Paragraph(cell(pct(profits_1i, p)), pct_styles),
        ])
    pct_rows.append([
        Paragraph('<b>Average</b>', ParagraphStyle('pavg', fontSize=9, alignment=TA_LEFT)),
        Paragraph(f'<b>{cell(avg_2i)}</b>', pct_styles),
        Paragraph(f'<b>{cell(avg_1i)}</b>', pct_styles),
    ])
    pct_table = Table(pct_rows, colWidths=[60*mm, 90*mm, 90*mm])
    pct_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), BRAND_COLOR),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ALIGN', (1,0), (-1,-1), 'RIGHT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.25, colors.lightgrey),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.white, GREY_LIGHT]),
        ('BACKGROUND', (0,-1), (-1,-1), BRAND_LIGHT),
        ('LINEABOVE', (0,-1), (-1,-1), 1.2, BRAND_COLOR),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(Paragraph('Profit Distribution', h2))
    story.append(pct_table)

    # ---- PER-VENDOR P&L ----
    story.append(Paragraph('Per-Vendor P&amp;L', h2))
    pnl_header = ['Vendor', 'Type', 'DD', 'SD', 'SL', 'Idle', 'Sites', 'Fixed', 'Payout', 'Profit']
    pnl_rows = [pnl_header]
    for v in data['vendors']:
        profit_color = 'green' if v['profit'] >= 0 else 'red'
        pnl_rows.append([
            v['name'],
            v['type'],
            str(v['dd_days']),
            str(v['sd_sites']),
            str(v['sl_days']),
            str(v['idle_days']),
            str(v['sites']),
            fmt_inr(v['fixed_cost']),
            fmt_inr(v['payout']),
            Paragraph(f'<font color="{profit_color}">{"-" if v["profit"]<0 else ""}{fmt_inr(v["profit"])}</font>',
                      ParagraphStyle('pp', fontSize=8, alignment=TA_RIGHT)),
        ])
    pnl_table = Table(pnl_rows, colWidths=[18*mm, 22*mm, 12*mm, 12*mm, 12*mm, 12*mm, 14*mm, 22*mm, 24*mm, 24*mm], repeatRows=1)
    pnl_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), BRAND_COLOR),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('ALIGN', (2,0), (-1,-1), 'RIGHT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.25, colors.lightgrey),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, GREY_LIGHT]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(pnl_table)

    # ---- DAILY DEMAND ----
    story.append(PageBreak())
    story.append(Paragraph('Daily Demand', h2))
    daily = data['daily']
    peak_day = data['peak_day']
    # Build demand table with bars rendered as filled cells
    max_d = max(daily) if daily else 1
    bar_rows = [['Day'] + [str(i+1) for i in range(len(daily))]]
    bar_rows.append(['Sites'] + [str(v) for v in daily])
    bar_table = Table(bar_rows, colWidths=[14*mm] + [(270/len(daily))*mm]*len(daily), rowHeights=[6*mm, 6*mm])
    style = [
        ('FONTSIZE', (0,0), (-1,-1), 7),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BACKGROUND', (0,0), (0,-1), BRAND_LIGHT),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.25, colors.lightgrey),
    ]
    style.append(('BACKGROUND', (peak_day+1, 0), (peak_day+1, -1), colors.HexColor('#ef9f27')))
    bar_table.setStyle(TableStyle(style))
    story.append(bar_table)
    story.append(Spacer(1, 4))
    story.append(Paragraph(f'<i>Peak day: Day {peak_day+1} with {daily[peak_day]} sites (highlighted orange).</i>', small))

    # ---- VENDOR ROSTER CALENDAR ----
    story.append(Paragraph('Vendor Roster Calendar', h2))
    roster = data['roster']
    n_vendors = len(roster)
    days = len(daily)
    # Header: Day numbers
    cal_header = ['Day'] + [v['name'] for v in data['vendors']]
    cal_rows = [cal_header]
    cell_styles = []
    for d in range(days):
        row = [f'D{d+1}']
        for vi in range(n_vendors):
            cell = roster[vi][d]
            # cell is a dict: {'type': 'DD'|'SD'|'SL'|'idle', 'label': '...'}
            kind = cell.get('type', 'idle')
            cell_label = cell.get('label', '')
            if kind == 'DD':
                label = f'DD\n{cell_label}'
            elif kind == 'SD':
                label = f'SD\n{cell_label}'
            elif kind == 'SL':
                label = 'SL'
            else:
                label = '—'
            row.append(label)
        cal_rows.append(row)
    col_count = len(cal_header)
    col_widths = [12*mm] + [(265 / max(1, n_vendors)) * mm] * n_vendors
    cal_table = Table(cal_rows, colWidths=col_widths, repeatRows=1)
    cstyle = [
        ('FONTSIZE', (0,0), (-1,-1), 6),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BACKGROUND', (0,0), (-1,0), BRAND_COLOR),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BACKGROUND', (0,1), (0,-1), BRAND_LIGHT),
        ('FONTNAME', (0,1), (0,-1), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.2, colors.lightgrey),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]
    # Color code DD (green), SD (light blue), SL (red), idle (grey)
    for d in range(days):
        row_idx = d + 1
        for vi in range(n_vendors):
            col_idx = vi + 1
            cell = roster[vi][d]
            kind = cell.get('type', 'idle')
            if kind == 'DD':
                cstyle.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), colors.HexColor('#c8e6c9')))
            elif kind == 'SD':
                cstyle.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), colors.HexColor('#e1ecf9')))
            elif kind == 'SL':
                cstyle.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), colors.HexColor('#fad4d4')))
    cal_table.setStyle(TableStyle(cstyle))
    story.append(cal_table)

    # ---- FOOTER ----
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        f'<i>SolarSquare 2-Install-A-Day Vendor Optimizer  |  Generated {now}</i>',
        small
    ))

    doc.build(story)
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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
