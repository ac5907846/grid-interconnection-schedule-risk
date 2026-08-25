"""Aalen-Johansen cumulative incidence by cohort, region, and technology."""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import AalenJohansenFitter
from lifelines.statistics import logrank_test

from utils import C, COHORT_COLORS, DERIVED, TAB, save_figure, set_style, df_to_md

warnings.filterwarnings('ignore')

set_style()
d = pd.read_pickle(DERIVED / 'analysis_dataset.pkl')
MIN_AT_RISK = 100


def max_horizon(g):
    """Longest duration at which at least MIN_AT_RISK projects remain at risk."""
    ts = np.sort(g['T_years'].values)[::-1]
    if len(ts) < MIN_AT_RISK:
        return float(ts[-1])
    return float(ts[MIN_AT_RISK - 1])


def cif_curve(g, event):
    aj = AalenJohansenFitter(calculate_variance=False)
    aj.fit(g['T_years'], g['event'], event_of_interest=event)
    c = aj.cumulative_density_
    return c.index.values, c.values.ravel()


def cif_at(g, event, horizons=(2, 3, 5, 8)):
    x, y = cif_curve(g, event)
    hmax = max_horizon(g)
    out = {}
    for h in horizons:
        if h > hmax:
            out[h] = np.nan
            continue
        idx = np.where(x <= h)[0]
        out[h] = float(y[idx[-1]]) if len(idx) else np.nan
    return out


COHORTS = ['2008-13', '2014-17', '2018-20', '2021-22', '2023-25']
fig = plt.figure(figsize=(7.2, 4.1))
gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.0], hspace=0.22, wspace=0.16)
XMAX, XT = 10, [0, 2, 4, 6, 8, 10]
XLIM = (-0.55, 10.4)

for col, (ev, name) in enumerate([(1, 'Commercial operation'), (2, 'Withdrawal')]):
    ax = fig.add_subplot(gs[0, col])
    for coh in COHORTS:
        g = d[d['cohort'] == coh]
        x, y = cif_curve(g, ev)
        hmax = min(max_horizon(g), XMAX)
        keep = x <= hmax
        ax.step(x[keep], y[keep], where='post', color=COHORT_COLORS[coh], lw=1.7,
                label=coh if ev == 1 else None)
        ax.plot(x[keep][-1], y[keep][-1], 'o', ms=3.2, color=COHORT_COLORS[coh],
                mec='white', mew=0.7)
    ax.set_xlim(*XLIM); ax.set_ylim(0, 0.80)
    ax.set_xticks(XT); ax.set_xticklabels([])
    ax.set_title(f'({"ab"[ev - 1]}) {name}', loc='left')
    ax.grid(axis='y')
    if col == 0:
        ax.set_ylabel('Cumulative incidence')
        ax.legend(frameon=False, title='Entry cohort', title_fontsize=7.5, loc='upper left')
    else:
        ax.set_yticklabels([])

    axr = fig.add_subplot(gs[1, col])
    for i, coh in enumerate(COHORTS):
        g = d[d['cohort'] == coh]
        ts = g['T_years'].values
        yy = len(COHORTS) - 1 - i
        for xt in XT:
            n_at = int((ts >= xt).sum())
            axr.text(xt, yy, f'{n_at:,}' if n_at >= 100 else '-', ha='center', va='center',
                     fontsize=6.3, color=COHORT_COLORS[coh])

    axr.set_xlim(*XLIM); axr.set_ylim(-0.6, len(COHORTS) - 0.4)
    axr.set_xticks(XT); axr.set_xlabel('Years since interconnection request')
    axr.set_yticks([])
    for sp in ['left', 'bottom']:
        axr.spines[sp].set_visible(False)
    axr.set_title('Projects at risk (rows in legend order, top to bottom)' if col == 0
                  else 'Projects at risk', loc='left', fontsize=7, color=C['mut'], pad=2)

save_figure(fig, 'fig03_cumulative_incidence')
plt.close()

rows = []
for coh in ['2000-07', '2008-13', '2014-17', '2018-20', '2021-22', '2023-25']:
    g = d[d['cohort'] == coh]
    ch, wh = cif_at(g, 1), cif_at(g, 2)
    rows.append({'Cohort': coh, 'Projects': len(g), 'Follow-up limit (yr)': round(max_horizon(g), 1),
                 'P(COD) 2y': ch[2], 'P(COD) 3y': ch[3], 'P(COD) 5y': ch[5], 'P(COD) 8y': ch[8],
                 'P(WD) 2y': wh[2], 'P(WD) 3y': wh[3], 'P(WD) 5y': wh[5], 'P(WD) 8y': wh[8]})
t = pd.DataFrame(rows).set_index('Cohort')
df_to_md(t.round(3), TAB / 'table_cif_by_cohort.md')
print(t.round(3).to_string(), '\n')

rows = []
for reg, g in d[d['q_year'].between(2014, 2020)].groupby('region'):
    ch, wh = cif_at(g, 1), cif_at(g, 2)
    rows.append({'Region': reg, 'Projects': len(g), 'Median MW': g['mw'].median(),
                 'P(COD) 3y': ch[3], 'P(COD) 5y': ch[5], 'P(WD) 3y': wh[3], 'P(WD) 5y': wh[5]})
tr = pd.DataFrame(rows).set_index('Region').sort_values('P(COD) 5y', ascending=False)
df_to_md(tr.round(3), TAB / 'table_cif_by_region.md')
print(tr.round(3).to_string(), '\n')

rows = []
for tech, g in d[d['q_year'].between(2014, 2020)].groupby('tech'):
    ch, wh = cif_at(g, 1), cif_at(g, 2)
    rows.append({'Technology': tech, 'Projects': len(g), 'Median MW': g['mw'].median(),
                 'P(COD) 3y': ch[3], 'P(COD) 5y': ch[5], 'P(WD) 5y': wh[5]})
tt = pd.DataFrame(rows).set_index('Technology').sort_values('P(COD) 5y', ascending=False)
df_to_md(tt.round(3), TAB / 'table_cif_by_tech.md')
print(tt.round(3).to_string(), '\n')

early = d[d['cohort'].isin(['2008-13', '2014-17'])]
late = d[d['cohort'].isin(['2018-20', '2021-22'])]
lr = logrank_test(early['T_years'], late['T_years'],
                  (early['event'] == 1).astype(int), (late['event'] == 1).astype(int))
(TAB / 'logrank_test.md').write_text(
    '# Log-rank test, completion hazard\n\n'
    'Groups: entry cohorts 2008-2017 versus 2018-2022. The competing event (withdrawal) '
    'is treated as censoring, giving the cause-specific comparison.\n\n'
    f'- Test statistic: {lr.test_statistic:.1f}\n'
    f'- p-value: {lr.p_value:.3e}\n'
    f'- n (2008-2017) = {len(early):,}; n (2018-2022) = {len(late):,}\n')
print(f'Log-rank early vs late: chi2 = {lr.test_statistic:.1f}, p = {lr.p_value:.3e}')
