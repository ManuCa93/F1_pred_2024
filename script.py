import fastf1
import fastf1.plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
import os
import logging

# ================= CONFIGURATION =================
if not os.path.exists('cache'):
    os.makedirs('cache')

fastf1.Cache.enable_cache('cache')
fastf1.plotting.setup_mpl()  # misc_mpl_mods è deprecato
logging.getLogger('fastf1').setLevel(logging.ERROR)
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")

# ================= DEFAULT PARAMETERS =================
YEAR = 2024
FALLBACK_PENALTY_FP = 18
FALLBACK_PENALTY_FORM = 12

# ================= HELPERS =================
def get_rep_session(event_row):
    return 'Practice 1' if event_row['EventFormat'] == 'sprint' else 'Practice 2'

def to_seconds(td):
    if pd.isna(td): return np.nan
    if isinstance(td, pd.Timedelta): return td.total_seconds()
    try: return float(td)
    except: return np.nan

def load_weekend_pace_seconds(year, gp_name, session_name, min_laps=5):
    try:
        session = fastf1.get_session(year, gp_name, session_name)
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        laps = session.laps.pick_quicklaps()
        if laps.empty: return {}
        laps = laps.copy()
        laps['LapTime_s'] = laps['LapTime'].apply(to_seconds)
        grouped = laps.groupby('Abbreviation')['LapTime_s'].agg(['median', 'count'])
        grouped = grouped[grouped['count'] >= min_laps]
        grouped = grouped.dropna(subset=['median'])
        return grouped['median'].to_dict()
    except Exception:
        return {}

def compute_recent_form(year, current_gp_name, window=3):
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        idxs = schedule.index[schedule['EventName'] == current_gp_name].tolist()
        if not idxs: return {}
        cur_idx = idxs[0]
        if cur_idx == 0: return {}
        start = max(0, cur_idx - window)
        past = schedule.iloc[start:cur_idx]
        pos_lists = defaultdict(list)
        for _, ev in past.iterrows():
            try:
                race = fastf1.get_session(year, ev['EventName'], 'Race')
                race.load(laps=False, telemetry=False, weather=False, messages=False)
                res = race.results
                for pos, abbr in enumerate(res['Abbreviation'].tolist(), start=1):
                    pos_lists[abbr].append(pos)
            except: continue
        return {d: np.mean(v) for d, v in pos_lists.items() if len(v) > 0}
    except Exception:
        return {}

def normalize_scores(score_dict):
    if not score_dict: return {}
    vals = np.array(list(score_dict.values()), dtype=float)
    mn, mx = vals.min(), vals.max()
    if np.isclose(mn, mx): return {k: 0.5 for k in score_dict.keys()}
    return {k: (v - mn) / (mx - mn) for k, v in score_dict.items()}

def blend(fp_dict, form_dict, w_fp=0.6, w_form=0.4):
    fp_sorted = sorted(fp_dict.items(), key=lambda x: x[1])
    fp_rank = {drv: i+1 for i, (drv, _) in enumerate(fp_sorted)}
    form_sorted = sorted(form_dict.items(), key=lambda x: x[1])
    form_rank = {drv: i+1 for i, (drv, _) in enumerate(form_sorted)}
    drivers = set(fp_rank.keys()) | set(form_rank.keys())
    if not drivers: return []
    raw_fp = {d: fp_rank.get(d, FALLBACK_PENALTY_FP) for d in drivers}
    raw_form = {d: form_rank.get(d, FALLBACK_PENALTY_FORM) for d in drivers}
    norm_fp = normalize_scores(raw_fp)
    norm_form = normalize_scores(raw_form)
    blended = {d: (norm_fp.get(d, 1.0) * w_fp) + (norm_form.get(d, 1.0) * w_form) for d in drivers}
    return [d for d, _ in sorted(blended.items(), key=lambda x: x[1])]

def get_real_podium(year, gp_name):
    try:
        race = fastf1.get_session(year, gp_name, 'Race')
        race.load(laps=False, telemetry=False, weather=False, messages=False)
        return race.results.iloc[:3]['Abbreviation'].tolist()
    except:
        return None

# ================= MAIN BACKTEST =================
def run_clean_backtest(year=YEAR, window_form=2, min_laps_longrun=5, w_fp=0.9, w_form=0.1, verbose=True):
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    schedule = schedule[schedule['EventDate'] < pd.Timestamp.now()]

    results = []
    driver_stats = defaultdict(lambda: {'real_wins': 0, 'pred_wins': 0, 'real_pod': 0, 'pred_pod': 0})

    if verbose:
        print(f"\n{'GP NAME':<22} | {'PREDICTED':<15} | {'REALITY':<15} | ACCURACY")
        print("-" * 75)

    for _, ev in schedule.iterrows():
        gp = ev['EventName']
        rep = get_rep_session(ev)

        fp_map = load_weekend_pace_seconds(year, gp, rep)
        form_map = compute_recent_form(year, gp, window=window_form)  # <-- pass window_form

        if len(fp_map) < 3 and not form_map: 
            continue

        final_rank = blend(fp_map, form_map, w_fp=w_fp, w_form=w_form)
        if not final_rank and form_map:
            final_rank = [d for d, _ in sorted(form_map.items(), key=lambda x: x[1])]
        if not final_rank: 
            continue

        pred = final_rank[:3]
        real = get_real_podium(year, gp)
        if not real: 
            continue

        match_count = len(set(pred) & set(real))
        accuracy_pct = (match_count / 3) * 100

        # --- UPDATE DRIVER STATS ---
        real_winner = real[0]
        pred_winner = pred[0]
        driver_stats[real_winner]['real_wins'] += 1
        if real_winner == pred_winner:
            driver_stats[real_winner]['pred_wins'] += 1

        for drv in real:
            driver_stats[drv]['real_pod'] += 1
            if drv in pred:
                driver_stats[drv]['pred_pod'] += 1

        if verbose:
            print(f"{gp[:20]:<22} | {', '.join(pred):<15} | {', '.join(real):<15} | {match_count}/3 ({accuracy_pct:.0f}%)")

        results.append({
            'GP': gp,
            'Podium_Accuracy': match_count,
            'Winner_Correct': pred[0] == real[0]
        })

    return pd.DataFrame(results), driver_stats

# ================= ANALYSIS =================
def analyze_driver_stats(driver_stats):
    if not driver_stats: return
    data = []
    for drv, stats in driver_stats.items():
        if stats['real_pod'] > 0:
            row = {'Driver': drv, **stats}
            data.append(row)
    df = pd.DataFrame(data)
    df['Win_Acc_Pct'] = np.where(df['real_wins']>0, df['pred_wins']/df['real_wins']*100, 0)
    df['Pod_Acc_Pct'] = df['pred_pod']/df['real_pod']*100
    df = df.sort_values(by='real_pod', ascending=False)

    print("\n" + "="*30)
    print(" DRIVER PERFORMANCE ANALYSIS ")
    print("="*30)
    print(f"{'DRIVER':<8} | {'WINS (Pred/Real)':<20} | {'ACCURACY':<10} || {'PODIUMS (Pred/Real)':<20} | {'ACCURACY':<10}")
    print("-"*85)
    for _, r in df.iterrows():
        print(f"{r['Driver']:<8} | {r['pred_wins']}/{r['real_wins']:<18} | {r['Win_Acc_Pct']:>6.1f}%   || {r['pred_pod']}/{r['real_pod']:<18} | {r['Pod_Acc_Pct']:>6.1f}%")

def plot_summary(df_results, driver_stats):
    if df_results.empty: return
    print("\n" + "="*30)
    print(" GLOBAL MODEL ACCURACY ")
    print("="*30)
    print(f"Total Races: {len(df_results)}")
    print(f"Winner Correct: {df_results['Winner_Correct'].mean()*100:.1f}%")
    print(f"Podium Accuracy: {df_results['Podium_Accuracy'].sum()/(len(df_results)*3)*100:.1f}%")
    analyze_driver_stats(driver_stats)

# ================= MAIN =================
if __name__ == "__main__":
    df_res, drv_stats = run_clean_backtest()
    plot_summary(df_res, drv_stats)
