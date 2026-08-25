"""Random survival forest benchmark and completion probability surface."""
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from lifelines import AalenJohansenFitter
from sksurv.util import Surv

from utils import C, DERIVED, SLATE_CMAP, TAB, save_figure, set_style, df_to_md

warnings.filterwarnings('ignore')

set_style()
d = pd.read_pickle(DERIVED / 'analysis_dataset.pkl')

d['E'] = (d['event'] == 1)
FEATS = ['log_mw', 'q_year', 'log_backlog', 'dc_market']
X = pd.get_dummies(d[FEATS + ['tech', 'region']], columns=['tech', 'region']).astype(float)
y = Surv.from_arrays(d['E'].values, d['T_years'].values)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=d['E'])
rsf = RandomSurvivalForest(n_estimators=100, min_samples_leaf=40, max_features='sqrt',
                           n_jobs=2, random_state=42).fit(Xtr, ytr)
sub, ysub = Xte.iloc[:2500].copy(), yte[:2500]
base = rsf.score(sub, ysub)
print(f'RSF holdout C-index: {base:.3f}')

GROUPS = {'Project size (log MW)': ['log_mw'], 'Entry year': ['q_year'],
          'Regional queue backlog': ['log_backlog'], 'Data center market county': ['dc_market'],
          'Technology': [c for c in X.columns if c.startswith('tech_')],
          'Region / ISO': [c for c in X.columns if c.startswith('region_')]}
rng = np.random.RandomState(0)
imp = []
for gname, cols in GROUPS.items():
    drops = []
    for _ in range(3):
        Xp = sub.copy()
        Xp[cols] = Xp[cols].values[rng.permutation(len(Xp))]
        drops.append(base - rsf.score(Xp, ysub))
    imp.append({'Predictor': gname, 'Importance': float(np.mean(drops)),
                'SD': float(np.std(drops))})
    print(f'  {gname}: {np.mean(drops):.4f}')
impdf = pd.DataFrame(imp).sort_values('Importance', ascending=False)
df_to_md(impdf.round(4), TAB / 'table_rsf_importance.md', index=False)
json.dump({'c_index_holdout': float(base), 'n_train': int(len(Xtr)), 'n_test': int(len(Xte))},
          open(TAB / 'rsf_fit.json', 'w'), indent=1)

MIN_AT_RISK = 25   # cells are far smaller than whole cohorts
def cif5(g, h=5):
    ts = np.sort(g['T_years'].values)[::-1]
    if len(ts) < MIN_AT_RISK or ts[MIN_AT_RISK - 1] < h:
        return np.nan
    aj = AalenJohansenFitter(calculate_variance=False)
    aj.fit(g['T_years'], g['event'], event_of_interest=1)
    c = aj.cumulative_density_
    idx = c.index[c.index <= h]
    return float(c.loc[idx[-1]].iloc[0]) if len(idx) else np.nan

panel = d[d['q_year'].between(2010, 2018)]
rows = []
for (reg, tech), g in panel.groupby(['region', 'tech']):
    if len(g) < 50:
        continue
    rows.append({'region': reg, 'tech': tech, 'p5': cif5(g), 'n': len(g)})
hm = pd.DataFrame(rows)
grid = hm.pivot(index='region', columns='tech', values='p5')
cnt = hm.pivot(index='region', columns='tech', values='n')
grid = grid.reindex(index=grid.mean(axis=1).sort_values(ascending=False).index)
cnt = cnt.reindex(index=grid.index)
df_to_md((grid * 100).round(1), TAB / 'table_completion_surface.md', floatfmt='.1f')

fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.3), gridspec_kw={'width_ratios': [1.15, 1]})

ax = axes[0]
ax.barh(impdf['Predictor'][::-1], impdf['Importance'][::-1], color=C['blue'], height=0.55)
ax.errorbar(impdf['Importance'][::-1], range(len(impdf)), xerr=impdf['SD'][::-1],
            fmt='none', ecolor=C['ink'], elinewidth=0.8, capsize=2)
for i, v in enumerate(impdf['Importance'][::-1]):
    ax.text(v + 0.0035, i, f'{v:.3f}', va='center', fontsize=7, color=C['mut'])
ax.set_xlabel('Permutation importance (loss in C-index)')
ax.set_title(f'(a) Predictor importance, C-index {base:.3f}', loc='left')
ax.set_xlim(0, max(impdf['Importance']) * 1.28)

ax = axes[1]
im = ax.imshow(grid.values * 100, cmap=SLATE_CMAP, vmin=0, vmax=np.nanmax(grid.values) * 100,
               aspect='auto')
ax.set_xticks(range(grid.shape[1])); ax.set_xticklabels(grid.columns, rotation=35, ha='right')
ax.set_yticks(range(grid.shape[0])); ax.set_yticklabels(grid.index)
vmax = np.nanmax(grid.values) * 100
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        v = grid.values[i, j] * 100
        if not np.isnan(v):
            ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=7.2,
                    color='white' if v > vmax * 0.58 else C['ink'])
        else:
            ax.text(j, i, '-', ha='center', va='center', fontsize=7.2, color='#c8c8c4')
ax.set_title('(b) Five-year completion probability (%)', loc='left')
ax.set_xlabel('Entries 2010-2018; cells with <50 projects omitted', fontsize=6.8, color=C['mut'])
plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
plt.tight_layout()
save_figure(fig, 'fig05_survival_forest')
plt.close()
print('\nCompletion surface (%):\n', (grid * 100).round(1).to_string())
