import fastf1
import pandas as pd
import numpy as np
import os
import logging
import warnings
import itertools
from collections import defaultdict
from tqdm import tqdm

# ================= CONFIGURATION =================
if not os.path.exists('cache'):
    os.makedirs('cache')

fastf1.Cache.enable_cache('cache')
logging.getLogger('fastf1').setLevel(logging.ERROR)
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")

# ================= PARAMETRI =================
YEAR = 2024
WINDOWS_TO_TEST = [2, 3, 4, 5]
MIN_LAPS_TO_TEST = [3, 4, 5, 6]
FP_WEIGHTS_TO_TEST = np.arange(0.0, 1.01, 0.05)

# Penalità (IDENTICHE A SCRIPT.PY)
FALLBACK_PENALTY_FP = 18
FALLBACK_PENALTY_FORM = 12

# ================= FUNZIONI ORIGINALI (COPIA-INCOLLA DA SCRIPT.PY) =================

def to_seconds(td):
    if pd.isna(td): return np.nan
    if isinstance(td, pd.Timedelta): return td.total_seconds()
    try: return float(td)
    except: return np.nan

def get_rep_session(event_row):
    return 'Practice 1' if event_row['EventFormat'] == 'sprint' else 'Practice 2'

def normalize_scores(score_dict):
    if not score_dict: return {}
    vals = np.array(list(score_dict.values()), dtype=float)
    mn, mx = vals.min(), vals.max()
    if np.isclose(mn, mx): return {k: 0.5 for k in score_dict.keys()}
    return {k: (v - mn) / (mx - mn) for k, v in score_dict.items()}

def blend(fp_dict, form_dict, w_fp, w_form):
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

# ================= LOADING FUNZIONI (MODIFICATE PER PRE-CALCOLO) =================

def load_pace_exact(session, min_laps):
    """Calcola il passo usando la sessione GIA caricata (per velocità)"""
    try:
        laps = session.laps.pick_quicklaps()
        if laps.empty: return {}
        laps = laps.copy()
        laps['LapTime_s'] = laps['LapTime'].apply(to_seconds)
        # USIAMO ABBREVIATION COME NEL TUO SCRIPT FUNZIONANTE
        grouped = laps.groupby('Abbreviation')['LapTime_s'].agg(['median', 'count'])
        grouped = grouped[grouped['count'] >= min_laps]
        grouped = grouped.dropna(subset=['median'])
        return grouped['median'].to_dict()
    except:
        return {}

def compute_form_exact(target_gp_name, schedule, results_cache, window):
    """Calcola la forma usando la cache dei risultati (per velocità)"""
    try:
        idxs = schedule.index[schedule['EventName'] == target_gp_name].tolist()
        if not idxs: return {}
        cur_idx = idxs[0]
        if cur_idx == 0: return {}
        start = max(0, cur_idx - window)
        past = schedule.iloc[start:cur_idx]
        
        pos_lists = defaultdict(list)
        for _, ev in past.iterrows():
            gp = ev['EventName']
            res = results_cache.get(gp, [])
            for pos, abbr in enumerate(res, start=1):
                pos_lists[abbr].append(pos)
        return {d: np.mean(v) for d, v in pos_lists.items() if len(v) > 0}
    except:
        return {}

# ================= PRE-LOADING MASSIVO =================

def preload_everything(year):
    print(f"📥 CARICAMENTO E PRE-CALCOLO {year} (Attendere...)\n")
    
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    schedule = schedule[schedule['EventDate'] < pd.Timestamp.now()]
    
    # 1. Carica tutti i risultati gara (necessari per la Forma)
    results_cache = {}
    print("   -> Scaricamento Risultati Gara...")
    for _, ev in tqdm(schedule.iterrows(), total=len(schedule), desc="Race Results"):
        gp = ev['EventName']
        try:
            r = fastf1.get_session(year, gp, 'Race')
            r.load(laps=False, telemetry=False, weather=False, messages=False)
            results_cache[gp] = r.results['Abbreviation'].tolist()
        except:
            results_cache[gp] = []

    # 2. Pre-calcola TUTTO (FP per ogni min_laps, Forma per ogni window)
    # cache[gp] = { 'real': [...], 'fp_vars': {3: {}, 4:{}...}, 'form_vars': {2: {}, 3:{}...} }
    final_cache = {}
    calendar = []

    print("   -> Scaricamento FP e Pre-calcolo Variabili...")
    for _, ev in tqdm(schedule.iterrows(), total=len(schedule), desc="Processing GPs"):
        gp = ev['EventName']
        calendar.append(gp)
        rep = get_rep_session(ev)
        
        entry = {
            'real': results_cache.get(gp, [])[:3], # Podio reale
            'fp_vars': {},
            'form_vars': {}
        }
        
        # A. Pre-calcolo FP (carica sessione una volta, calcola N varianti)
        try:
            s = fastf1.get_session(year, gp, rep)
            s.load(laps=True, telemetry=False, weather=False, messages=False)
            
            for ml in MIN_LAPS_TO_TEST:
                entry['fp_vars'][ml] = load_pace_exact(s, ml)
        except:
            pass # Se fallisce, restano dizionari vuoti

        # B. Pre-calcolo Forma
        for win in WINDOWS_TO_TEST:
            entry['form_vars'][win] = compute_form_exact(gp, schedule, results_cache, win)
            
        final_cache[gp] = entry

    print("\n✅ TUTTO PRONTO. LOGICA DUPLICATA AL 100%. AVVIO GRID SEARCH.\n")
    return final_cache, calendar

# ================= GRID SEARCH LOOP =================

def run_optimizer():
    # 1. Preload
    cache, calendar = preload_everything(YEAR)
    
    all_combinations = list(itertools.product(WINDOWS_TO_TEST, MIN_LAPS_TO_TEST, FP_WEIGHTS_TO_TEST))
    results_log = []

    # 2. Loop Matematico (Veloce)
    for window, min_laps, w_fp in tqdm(all_combinations, desc="Grid Search"):
        w_form = round(1.0 - w_fp, 2)
        
        total = 0
        correct = 0
        
        for gp in calendar:
            data = cache[gp]
            real = data['real']
            
            # Recupera i dati pre-calcolati (senza ricalcolare nulla)
            fp_map = data['fp_vars'].get(min_laps, {})
            form_map = data['form_vars'].get(window, {})
            
            # --- LOGICA IDENTICA A SCRIPT.PY ---
            if len(fp_map) < 3 and not form_map: continue
            if not real: continue # Se non abbiamo il risultato reale non possiamo verificare

            final_rank = blend(fp_map, form_map, w_fp, w_form)
            
            if not final_rank and form_map:
                final_rank = [d for d, _ in sorted(form_map.items(), key=lambda x: x[1])]
            
            if not final_rank: continue
            
            pred = final_rank[:3]
            
            # Score
            matches = len(set(pred) & set(real))
            correct += matches
            total += 3
            
        acc = (correct / total * 100) if total > 0 else 0
        
        results_log.append({
            'Window': window, 'Min_Laps': min_laps, 'W_FP': w_fp, 'W_Form': w_form, 'Accuracy': acc
        })

    # 3. Risultati
    df = pd.DataFrame(results_log).sort_values(by='Accuracy', ascending=False)

    print("\n" + "="*40)
    print(" 🏆 TOP 20 CONFIGURAZIONI ")
    print("="*40)
    print(df.head(100).to_string(index=False))
    
    best = df.iloc[0]
    print(f"\n✅ BEST: Window={int(best['Window'])}, MinLaps={int(best['Min_Laps'])}, FP={best['W_FP']:.2f}, Form={best['W_Form']:.2f}")
    print(f"🚀 ACCURACY: {best['Accuracy']:.2f}%")

if __name__ == "__main__":
    run_optimizer()