"""Proportional hazards tests, stratified refits, Fine-Gray models, confounder checks."""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import AalenJohansenFitter, CoxPHFitter
from lifelines.statistics import proportional_hazard_test

from utils import C, COHORT_COLORS, DERIVED, TAB, save_figure, set_style, df_to_md

warnings.filterwarnings('ignore')

set_style()
d = pd.read_pickle(DERIVED / 'analysis_dataset.pkl')

BASE = {'tech': 'Solar', 'region': 'MISO', 'cohort': '2014-17'}
COVS = ['log_mw', 'log_backlog', 'dc_market']


def design(df, drop_cohort=False):
    cols = COVS + ['tech', 'region'] + ([] if drop_cohort else ['cohort'])
    X = pd.get_dummies(df[['T_years'] + cols],
                       columns=[c for c in ['tech', 'region', 'cohort'] if c in cols],
                       drop_first=False)
    for k, v in BASE.items():
        X = X.drop(columns=f'{k}_{v}', errors='ignore')
    X = X.astype(float)
    keep = [c for c in X.columns if c == 'T_years' or X[c].nunique() > 1]
    return X[keep]


def fit_cox(df, event=1, drop_cohort=False):
    df = df.copy()
    df['E'] = (df['event'] == event).astype(int)
    X = design(df, drop_cohort); X['E'] = df['E'].values
    m = CoxPHFitter(penalizer=0.01).fit(X, duration_col='T_years', event_col='E')
    return m, X


m_full, X_full = fit_cox(d, 1)
ph = proportional_hazard_test(m_full, X_full, time_transform='rank')
ph_tbl = ph.summary.copy()
ph_tbl = ph_tbl[[c for c in ['test_statistic', 'p'] if c in ph_tbl.columns]]
ph_tbl.columns = ['Test statistic', 'p']
if isinstance(ph_tbl.index, pd.MultiIndex):
    ph_tbl.index = [i[0] for i in ph_tbl.index]
ph_tbl = ph_tbl.sort_values('p')
df_to_md(ph_tbl.round(4), TAB / 'table_ph_test.md', floatfmt='.4f')
viol = ph_tbl[ph_tbl['p'] < 0.05]
print(f'Proportional hazards test: {len(viol)} of {len(ph_tbl)} covariates violate at p < 0.05')
print(ph_tbl.head(10).round(4).to_string())

strat_rows = []
for strat in ['region', 'tech']:
    df = d.copy(); df['E'] = (df['event'] == 1).astype(int)
    X = design(df, drop_cohort=False)
    X = X[[c for c in X.columns if not c.startswith(f'{strat}_')]]
    X['E'] = df['E'].values
    X[strat] = df[strat].values
    m = CoxPHFitter(penalizer=0.01).fit(X, duration_col='T_years', event_col='E',
                                        strata=[strat])
    for k in [c for c in m.summary.index if c.startswith('cohort_')]:
        strat_rows.append({'Model': f'Stratified by {strat}', 'Covariate': k,
                           'Hazard ratio': m.summary.loc[k, 'exp(coef)'],
                           'p': m.summary.loc[k, 'p']})
for k in [c for c in m_full.summary.index if c.startswith('cohort_')]:
    strat_rows.append({'Model': 'Unstratified (Table 5)', 'Covariate': k,
                       'Hazard ratio': m_full.summary.loc[k, 'exp(coef)'],
                       'p': m_full.summary.loc[k, 'p']})
strat = pd.DataFrame(strat_rows).pivot(index='Covariate', columns='Model',
                                       values='Hazard ratio')
df_to_md(strat.round(3), TAB / 'table_stratified_cohort_effects.md')
print('\nCohort hazard ratios, unstratified versus stratified:')
print(strat.round(3).to_string())

PRE = d[d['q_year'] <= 2022]
m_pre, _ = fit_cox(PRE, 1)
comp_rows = []
for k in m_full.summary.index:
    if k in m_pre.summary.index:
        comp_rows.append({
            'Covariate': k,
            'Full sample HR': m_full.summary.loc[k, 'exp(coef)'],
            'Full p': m_full.summary.loc[k, 'p'],
            'Pre-reform HR': m_pre.summary.loc[k, 'exp(coef)'],
            'Pre-reform p': m_pre.summary.loc[k, 'p']})
cmp = pd.DataFrame(comp_rows).set_index('Covariate')
df_to_md(cmp.round(3), TAB / 'table_prereform_comparison.md')
print(f'\nPre-reform sample (entries through 2022): n = {len(PRE):,}, '
      f'events = {int((PRE["event"] == 1).sum()):,}')
print(cmp.loc[[i for i in cmp.index if i.startswith("cohort_") or i == "log_mw"]]
      .round(3).to_string())

late = d[(d['event'] == 1) & (d['end_date'].dt.year >= 2024)]
entry_mix = late['q_year'].describe(percentiles=[0.1, 0.5, 0.9])
(TAB / 'reform_exposure.md').write_text(
    '# Exposure of the realized duration series to FERC Order No. 2023\n\n'
    'Projects reaching commercial operation in 2024 and 2025 entered the queue years '
    'earlier, so their durations were largely determined under the pre-reform regime.\n\n'
    f'- Completions in 2024-2025: n = {len(late):,}\n'
    f'- Queue entry year, 10th percentile: {entry_mix["10%"]:.0f}\n'
    f'- Queue entry year, median: {entry_mix["50%"]:.0f}\n'
    f'- Queue entry year, 90th percentile: {entry_mix["90%"]:.0f}\n'
    f'- Share entering before Order No. 2023 (2023 or earlier entry year): '
    f'{(late["q_year"] <= 2023).mean() * 100:.1f} percent\n')
print(f'\nCompletions 2024-2025: median entry year {entry_mix["50%"]:.0f}, '
      f'{(late["q_year"] <= 2023).mean() * 100:.1f} percent entered before the reform')

pre_covid = d[d['q_year'] <= 2019]
rows = []
for coh in ['2008-13', '2014-17', '2018-20']:
    g = pre_covid[pre_covid['cohort'] == coh]
    if len(g) < 200:
        continue
    aj = AalenJohansenFitter(calculate_variance=False)
    aj.fit(g['T_years'], g['event'], event_of_interest=1)
    c = aj.cumulative_density_
    vals = {}
    for h in [3, 5]:
        i2 = c.index[c.index <= h]
        vals[f'P(COD) {h}y'] = float(c.loc[i2[-1]].iloc[0]) if len(i2) else np.nan
    rows.append({'Cohort (entries through 2019 only)': coh, 'Projects': len(g), **vals})
covid = pd.DataFrame(rows).set_index('Cohort (entries through 2019 only)')
df_to_md(covid.round(3), TAB / 'table_precovid_cohorts.md')
print('\nCompletion probability among cohorts that entered before 2020:')
print(covid.round(3).to_string())

comp = d[d['event'] == 1].copy()
comp['cod_q'] = comp['end_date'].dt.to_period('Q')
qs = comp.groupby('cod_q')['T_years'].median()
qs = qs[(qs.index >= pd.Period('2005Q1')) & (qs.index <= pd.Period('2025Q4'))]
xs = np.array([p.start_time.year + (p.quarter - 1) / 4 for p in qs.index])

fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.0),
                         gridspec_kw={'width_ratios': [1.35, 1.0]})

ax = axes[0]
ax.plot(xs, qs.values, color=C['silver'], lw=1.2, alpha=0.9)
ax.plot(xs, pd.Series(qs.values).rolling(4, center=True, min_periods=2).mean(),
        color=C['ink'], lw=2.0)
EVENTS = [(2020.2, 'COVID-19', C['graph'], ':', 2.05, 'right'),
          (2021.0, 'Structural break 2021Q1', C['blue'], '-', 1.25, 'right'),
          (2023.58, 'FERC Order No. 2023', C['green'], '--', 0.45, 'right')]
for x, lab, col, ls, ytxt, ha in EVENTS:
    ax.axvline(x, color=col, lw=1.1, ls=ls)
    ax.annotate(lab, xy=(x - 0.25, ytxt), fontsize=6.8, color=col, ha=ha, va='bottom')
ax.set_ylabel('Median request-to-COD (years)')
ax.set_xlabel('Year of commercial operation')
ax.set_title('(a) Duration series against the confounders', loc='left')
ax.set_ylim(0, 7.0); ax.grid(axis='y')
ax.text(0.03, 0.90, 'Thin line: quarterly median\nThick line: 4-quarter moving average',
        transform=ax.transAxes, fontsize=6.5, color=C['mut'], va='top')

ax = axes[1]
lab = {'cohort_2000-07': '2000-07', 'cohort_2008-13': '2008-13',
       'cohort_2018-20': '2018-20', 'cohort_2021-22': '2021-22', 'cohort_2023-25': '2023-25'}
ks = [k for k in lab if k in cmp.index]
y = np.arange(len(ks))
ax.axvline(1, color='#999999', lw=0.8, ls=':')
ax.plot(cmp.loc[ks, 'Full sample HR'], y + 0.13, 'o', ms=5, color=C['blue'],
        label='Full sample', mec='white', mew=0.7)
ax.plot(cmp.loc[ks, 'Pre-reform HR'], y - 0.13, 's', ms=4.6, color=C['green'],
        label='Entries through 2022 only', mec='white', mew=0.7)
ax.set_yticks(y); ax.set_yticklabels([lab[k] for k in ks])
ax.set_xscale('log')
ax.set_xlim(0.2, 2.4)
ax.set_xticks([0.25, 0.5, 1, 2])
ax.set_xticklabels(['0.25', '0.5', '1', '2'])
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.tick_params(axis='x', which='minor', length=0)
ax.set_xlabel('Completion hazard ratio\n(reference: entered 2014-17)')
ax.set_title('(b) Sensitivity to the reform', loc='left')
ax.legend(frameon=False, loc='lower right', fontsize=7)
ax.grid(axis='x', alpha=0.5)

plt.tight_layout()
save_figure(fig, 'figS1_identification')
plt.close()
print('\nfigure written.')
