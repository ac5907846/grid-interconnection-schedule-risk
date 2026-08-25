"""Cause-specific Cox and accelerated failure time models."""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import (CoxPHFitter, LogLogisticAFTFitter, LogNormalAFTFitter,
                       WeibullAFTFitter)

from utils import C, DERIVED, TAB, save_figure, set_style, df_to_md

warnings.filterwarnings('ignore')

set_style()
d = pd.read_pickle(DERIVED / 'analysis_dataset.pkl')

BASE = {'tech': 'Solar', 'region': 'MISO', 'cohort': '2014-17'}
COVS = ['log_mw', 'log_backlog', 'dc_market']


def design(df):
    X = pd.get_dummies(df[['T_years'] + COVS + ['tech', 'region', 'cohort']],
                       columns=['tech', 'region', 'cohort'], drop_first=False)
    for k, v in BASE.items():
        col = f'{k}_{v}'
        if col in X:
            X = X.drop(columns=col)
    return X.astype(float)


results = {}
for ev, name in [(1, 'completion'), (2, 'withdrawal')]:
    df = d.copy()
    df['E'] = (df['event'] == ev).astype(int)   # cause-specific: competing event censored
    X = design(df); X['E'] = df['E'].values
    m = CoxPHFitter(penalizer=0.01).fit(X, duration_col='T_years', event_col='E')
    s = m.summary[['coef', 'exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]
    results[name] = s
    if name == 'completion':
        cidx_completion = m.concordance_index_
    else:
        cidx_withdrawal = m.concordance_index_
    df_to_md(s.round(4), TAB / f'table_cox_{name}.md', floatfmt='.4f')
    print(f'--- Cause-specific Cox, {name}: n = {len(X):,}, events = {int(X["E"].sum()):,}, '
          f'C-index = {m.concordance_index_:.3f}')
    print(s.round(3).to_string(), '\n')

df = d.copy(); df['E'] = (df['event'] == 1).astype(int)
Xa = design(df); Xa['E'] = df['E'].values
CAND = [('Weibull', WeibullAFTFitter), ('Log-normal', LogNormalAFTFitter),
        ('Log-logistic', LogLogisticAFTFitter)]
fits, fitstat = {}, []
for name, F in CAND:
    m = F(penalizer=0.01).fit(Xa, duration_col='T_years', event_col='E')
    fits[name] = m
    fitstat.append({'Distribution': name, 'Log-likelihood': m.log_likelihood_, 'AIC': m.AIC_})
fitstat = pd.DataFrame(fitstat).set_index('Distribution').sort_values('AIC')
best = fitstat.index[0]
aft = fits[best]
df_to_md(fitstat.round(1), TAB / 'table_aft_model_selection.md', floatfmt='.1f')
print('\n--- AFT distribution selection (lower AIC is better)')
print(fitstat.round(1).to_string())
print(f'Primary specification: {best}')


def time_ratios(m):
    lvl = [l for l in ['alpha_', 'mu_', 'lambda_'] if l in m.params_.index.get_level_values(0)][0]
    s = m.summary.loc[lvl]
    return pd.DataFrame({
        'Time ratio': np.exp(s['coef']),
        'Lower 95%': np.exp(s['coef lower 95%']),
        'Upper 95%': np.exp(s['coef upper 95%']),
        'p': s['p']})


tr = time_ratios(aft)
df_to_md(tr.round(3), TAB / 'table_aft_completion.md')
df_to_md(time_ratios(fits['Weibull']).round(3), TAB / 'table_aft_completion_weibull.md')
print(f'\n--- {best} accelerated failure time model (time ratios)')
print(tr.round(3).to_string())
(TAB / 'model_fit.md').write_text(
    '# Model fit statistics\n\n'
    f'- Cause-specific Cox, completion: C-index = {cidx_completion:.3f}\n'
    f'- Cause-specific Cox, withdrawal: C-index = {cidx_withdrawal:.3f}\n'
    f'- Accelerated failure time, selected distribution: {best}\n'
    + fitstat.round(1).to_markdown() + '\n\n'
    'The Weibull specification is retained as a comparison because its shape parameter has '
    'a direct interpretation, but the log-logistic fits materially better and is used for '
    'the reported time ratios. A09 confirms all three distributions agree in direction.\n')

LABELS = [
    ('log_mw', 'Project size (log MW)'),
    ('log_backlog', 'Regional queue backlog (log)'),
    ('dc_market', 'Data center market county'),
    ('tech_Gas', 'Gas (vs solar)'),
    ('tech_Wind', 'Wind (vs solar)'),
    ('tech_Battery', 'Battery (vs solar)'),
    ('tech_Hybrid', 'Hybrid (vs solar)'),
    ('tech_Nuclear', 'Nuclear (vs solar)'),
    ('region_ERCOT', 'ERCOT (vs MISO)'),
    ('region_PJM', 'PJM (vs MISO)'),
    ('region_CAISO', 'CAISO (vs MISO)'),
    ('region_Southeast', 'Southeast (vs MISO)'),
    ('region_West', 'West non-ISO (vs MISO)'),
    ('region_NYISO', 'NYISO (vs MISO)'),
    ('region_ISO-NE', 'ISO-NE (vs MISO)'),
    ('cohort_2008-13', 'Entered 2008-13 (vs 2014-17)'),
    ('cohort_2018-20', 'Entered 2018-20 (vs 2014-17)'),
    ('cohort_2021-22', 'Entered 2021-22 (vs 2014-17)'),
    ('cohort_2023-25', 'Entered 2023-25 (vs 2014-17)'),
]
fig, axes = plt.subplots(1, 2, figsize=(7.3, 4.6), sharey=True)
EVENT_COL = {'completion': C['green'], 'withdrawal': C['red']}
for ax, name, ttl in [(axes[0], 'completion', '(a) Completion hazard'),
                      (axes[1], 'withdrawal', '(b) Withdrawal hazard')]:
    s = results[name]
    ax.axvline(1, color='#999999', lw=0.8, ls=':')
    for i, (k, _) in enumerate(LABELS):
        if k not in s.index:
            continue
        y = len(LABELS) - 1 - i
        hr, lo, hi = s.loc[k, 'exp(coef)'], s.loc[k, 'exp(coef) lower 95%'], s.loc[k, 'exp(coef) upper 95%']
        sig = s.loc[k, 'p'] < 0.05
        col = EVENT_COL[name] if sig else C['silver']
        ax.plot([lo, hi], [y, y], color=col, lw=1.3, alpha=0.8, solid_capstyle='round')
        ax.plot(hr, y, 'o', ms=4.2, color=col, mec='white', mew=0.7, zorder=3)
    ax.set_xscale('log')
    ax.set_xticks([0.1, 0.25, 0.5, 1, 2, 4])
    ax.set_xticklabels(['0.1', '0.25', '0.5', '1', '2', '4'])
    ax.set_xlim(0.06, 6)
    ax.set_xlabel('Hazard ratio (log scale)')
    ax.set_title(ttl, loc='left')
    ax.grid(axis='x', alpha=0.5)
axes[0].set_yticks(range(len(LABELS)))
axes[0].set_yticklabels([lab for _, lab in LABELS][::-1])
axes[0].set_ylim(-0.7, len(LABELS) - 0.3)
axes[1].text(0.02, 0.985, 'Filled color markers: p < 0.05; grey: not significant.\nBars are 95% confidence intervals.',
             transform=axes[1].transAxes, ha='left', va='top', fontsize=6.5, color=C['mut'])
plt.tight_layout()
save_figure(fig, 'fig04_hazard_ratios')
plt.close()
print('\nfigure written.')
