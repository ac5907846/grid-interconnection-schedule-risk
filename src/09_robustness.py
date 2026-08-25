"""Temporal validation, imputation sensitivity, alternative specifications."""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import (CoxPHFitter, LogLogisticAFTFitter, LogNormalAFTFitter,
                       WeibullAFTFitter)
from lifelines import AalenJohansenFitter
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

from utils import C, DATA, DERIVED, TAB, save_figure, set_style, df_to_md

warnings.filterwarnings('ignore')

set_style()
d = pd.read_pickle(DERIVED / 'analysis_dataset.pkl')
BASE = {'tech': 'Solar', 'region': 'MISO', 'cohort': '2014-17'}
CENSOR = pd.Timestamp('2025-12-31')


def design(df, cohort=True, year=False):
    cols = ['log_mw', 'log_backlog', 'dc_market', 'tech', 'region']
    cols += ['cohort'] if cohort else []
    cols += ['q_year'] if year else []
    X = pd.get_dummies(df[['T_years'] + cols],
                       columns=[c for c in ['tech', 'region', 'cohort'] if c in cols])
    for k, v in BASE.items():
        X = X.drop(columns=f'{k}_{v}', errors='ignore')
    X = X.astype(float)
    return X[[c for c in X.columns if c == 'T_years' or X[c].nunique() > 1]]


def cox(df, cohort=True, year=False):
    df = df.copy(); df['E'] = (df['event'] == 1).astype(int)
    X = design(df, cohort, year); X['E'] = df['E'].values
    return CoxPHFitter(penalizer=0.01).fit(X, duration_col='T_years', event_col='E')


def cif_at(g, h=5, event=1, min_at_risk=100):
    ts = np.sort(g['T_years'].values)[::-1]
    if len(ts) < min_at_risk or ts[min_at_risk - 1] < h:
        return np.nan
    aj = AalenJohansenFitter(calculate_variance=False)
    aj.fit(g['T_years'], g['event'], event_of_interest=event)
    c = aj.cumulative_density_
    i2 = c.index[c.index <= h]
    return float(c.loc[i2[-1]].iloc[0]) if len(i2) else np.nan


SRC = DATA / 'lbnl_queue_thru2025.xlsx'
raw = pd.read_excel(SRC, sheet_name='03. Complete Queue Data', header=1)
raw = raw[(raw['project_type'] == 'Generation') & raw['q_date'].notna()]
raw['mw'] = raw[['mw_1', 'mw_2', 'mw_3']].sum(axis=1, min_count=1)
raw = raw[raw['mw'].notna() & (raw['mw'] > 0)]
raw = raw[raw['q_year'].between(2000, 2025)]
miss = raw[((raw['q_status'] == 'withdrawn') & raw['wd_date'].isna()) |
           ((raw['q_status'] == 'operational') & raw['on_date'].isna())].copy()

TECH = {'Solar': 'Solar', 'Wind': 'Wind', 'Offshore Wind': 'Wind', 'Battery': 'Battery',
        'Solar+Battery': 'Hybrid', 'Wind+Battery': 'Hybrid', 'Gas+Battery': 'Hybrid',
        'Gas': 'Gas', 'Nuclear': 'Nuclear'}
miss['tech'] = miss['type_clean'].map(TECH).fillna('Other')
bal = pd.DataFrame({
    'Excluded records': pd.concat([miss['region'].value_counts(normalize=True),
                                   miss['tech'].value_counts(normalize=True)]) * 100,
    'Analysis sample': pd.concat([d['region'].value_counts(normalize=True),
                                  d['tech'].value_counts(normalize=True)]) * 100})
df_to_md(bal.dropna().round(1), TAB / 'table_missing_balance.md', floatfmt='.1f')

med_wd = d.loc[d['event'] == 2, 'T_years'].median()
med_cod = d.loc[d['event'] == 1, 'T_years'].median()
scen_rows = []
for name, tw, tc in [
        ('Baseline (excluded)', None, None),
        ('Imputed at observed medians', med_wd, med_cod),
        ('Imputed early (25th percentile)',
         d.loc[d['event'] == 2, 'T_years'].quantile(0.25),
         d.loc[d['event'] == 1, 'T_years'].quantile(0.25)),
        ('Imputed late (75th percentile)',
         d.loc[d['event'] == 2, 'T_years'].quantile(0.75),
         d.loc[d['event'] == 1, 'T_years'].quantile(0.75))]:
    if tw is None:
        dd = d.copy()
    else:
        add = miss.copy()
        add['event'] = np.where(add['q_status'] == 'withdrawn', 2, 1)
        add['T_years'] = np.where(add['event'] == 2, tw, tc)
        add['cohort'] = pd.cut(add['q_year'], bins=[1999, 2007, 2013, 2017, 2020, 2022, 2025],
                               labels=['2000-07', '2008-13', '2014-17', '2018-20',
                                       '2021-22', '2023-25'])
        dd = pd.concat([d[['T_years', 'event', 'cohort']], add[['T_years', 'event', 'cohort']]],
                       ignore_index=True)
    row = {'Scenario': name, 'n': len(dd)}
    for coh in ['2008-13', '2014-17', '2018-20']:
        row[f'P(COD) 5y, {coh}'] = cif_at(dd[dd['cohort'] == coh])
    scen_rows.append(row)
scen = pd.DataFrame(scen_rows).set_index('Scenario')
df_to_md(scen.round(3), TAB / 'table_missing_sensitivity.md')
print('Missing-data sensitivity, five-year completion probability by cohort:')
print(scen.round(3).to_string())

d['E'] = (d['event'] == 1)
FEATS = ['log_mw', 'q_year', 'log_backlog', 'dc_market']
X = pd.get_dummies(d[FEATS + ['tech', 'region']], columns=['tech', 'region']).astype(float)
y = Surv.from_arrays(d['E'].values, d['T_years'].values)
tr = (d['q_year'] <= 2017).values
rsf = RandomSurvivalForest(n_estimators=100, min_samples_leaf=40, max_features='sqrt',
                           n_jobs=2, random_state=42).fit(X[tr], y[tr])
c_temporal = rsf.score(X[~tr].iloc[:3000], y[~tr][:3000])
cph_t = cox(d[d['q_year'] <= 2017], cohort=False, year=True)
val = {'RSF, temporal split (train entries <= 2017, test 2018+)': float(c_temporal),
       'RSF, random split (A05)': 0.864,
       'Cox, in-sample (A04)': 0.844,
       'Cox trained on entries <= 2017, in-sample': float(cph_t.concordance_index_)}
(TAB / 'temporal_validation.md').write_text(
    '# Temporal validation of the predictive models\n\n'
    'A random train and test split allows a model to learn from projects that entered '
    'after those it predicts. Refitting on a strictly temporal split removes that leakage.\n\n'
    + '\n'.join(f'- {k}: C-index = {v:.3f}' for k, v in val.items()) + '\n')
print('\nTemporal validation:')
for k, v in val.items():
    print(f'  {k}: {v:.3f}')

aft_rows = []
dd = d.copy(); dd['E'] = (dd['event'] == 1).astype(int)
Xa = design(dd); Xa['E'] = dd['E'].values
fits = {}
for name, F in [('Weibull', WeibullAFTFitter), ('Log-normal', LogNormalAFTFitter),
                ('Log-logistic', LogLogisticAFTFitter)]:
    m = F(penalizer=0.01).fit(Xa, duration_col='T_years', event_col='E')
    fits[name] = m
    aft_rows.append({'Distribution': name, 'Log-likelihood': m.log_likelihood_,
                     'AIC': m.AIC_, 'Concordance': m.concordance_index_})
aft = pd.DataFrame(aft_rows).set_index('Distribution').sort_values('AIC')
df_to_md(aft.round(1), TAB / 'table_aft_distributions.md', floatfmt='.1f')
print('\nAccelerated failure time distribution comparison:')
print(aft.round(1).to_string())

tr_rows = {}
for name, m in fits.items():
    p = m.params_
    lvl = 'lambda_' if 'lambda_' in p.index.get_level_values(0) else \
          ('mu_' if 'mu_' in p.index.get_level_values(0) else 'alpha_')
    tr_rows[name] = np.exp(p.loc[lvl])
trc = pd.DataFrame(tr_rows)
trc = trc.loc[[i for i in trc.index if i.startswith('cohort_') or i in
               ('log_mw', 'tech_Gas', 'region_ERCOT', 'region_ISO-NE')]]
df_to_md(trc.round(3), TAB / 'table_time_ratios_by_distribution.md')
print('\nTime ratios under each distribution:')
print(trc.round(3).to_string())

alt = d.copy()
alt['cohort'] = pd.cut(alt['q_year'], bins=[1999, 2009, 2015, 2019, 2022, 2025],
                       labels=['2000-09', '2010-15', '2016-19', '2020-22', '2023-25'])
BASE_ALT = dict(BASE); BASE_ALT['cohort'] = '2010-15'
alt2 = alt.copy(); alt2['E'] = (alt2['event'] == 1).astype(int)
Xb = pd.get_dummies(alt2[['T_years', 'log_mw', 'log_backlog', 'dc_market',
                          'tech', 'region', 'cohort']],
                    columns=['tech', 'region', 'cohort']).astype(float)
for k, v in BASE_ALT.items():
    Xb = Xb.drop(columns=f'{k}_{v}', errors='ignore')
Xb['E'] = alt2['E'].values
m_alt = CoxPHFitter(penalizer=0.01).fit(Xb, duration_col='T_years', event_col='E')

m_year = cox(d, cohort=False, year=True)
yr_hr = m_year.summary.loc['q_year', 'exp(coef)']
spec = pd.DataFrame({
    'Hazard ratio': list(m_alt.summary.loc[[i for i in m_alt.summary.index
                                            if i.startswith('cohort_')], 'exp(coef)'])
    + [yr_hr],
    'p': list(m_alt.summary.loc[[i for i in m_alt.summary.index
                                 if i.startswith('cohort_')], 'p']) +
         [m_year.summary.loc['q_year', 'p']]},
    index=[i for i in m_alt.summary.index if i.startswith('cohort_')]
          + ['Continuous entry year (per year)'])
df_to_md(spec.round(4), TAB / 'table_alternative_specifications.md', floatfmt='.4f')
print('\nAlternative cohort boundaries and continuous year:')
print(spec.round(3).to_string())
print(f'\nContinuous specification: each additional entry year multiplies the completion '
      f'hazard by {yr_hr:.3f}, or {(1 - yr_hr) * 100:.1f} percent per year.')

fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))

ax = axes[0]
cols_s = [c for c in scen.columns if c.startswith('P(COD)')]
xlab = [c.split(', ')[1] for c in cols_s]
marks = ['o', 's', '^', 'v']
for i, (name, r) in enumerate(scen.iterrows()):
    ax.plot(range(len(cols_s)), [r[c] for c in cols_s], marks[i % 4] + '-', ms=5,
            lw=1.3, label=name, color=[C['ink'], C['blue'], C['green'], C['silver']][i % 4],
            alpha=0.9)
ax.set_xticks(range(len(cols_s))); ax.set_xticklabels(xlab)
ax.set_ylabel('P(commercial operation by 5y)')
ax.set_xlabel('Entry cohort')
ax.set_title('(a) Missing event dates', loc='left')
ax.legend(frameon=False, fontsize=6.2, loc='upper right')
ax.set_ylim(0, 0.30); ax.grid(axis='y')

ax = axes[1]
names = list(val.keys()); vals = list(val.values())
short = ['RSF temporal', 'RSF random', 'Cox in-sample', 'Cox pre-2018']
ax.barh(short[::-1], vals[::-1], color=[C['blue'], C['silver'], C['silver'], C['graph']][::-1],
        height=0.55)
for i, v in enumerate(vals[::-1]):
    ax.text(v + 0.006, i, f'{v:.3f}', va='center', fontsize=7.2, color=C['mut'])
ax.set_xlim(0.5, 1.0); ax.set_xlabel('Concordance index')
ax.set_title('(b) Predictive validation', loc='left')

ax = axes[2]
ks = [i for i in trc.index if i.startswith('cohort_')]
xpos = np.arange(len(ks))
for j, (name, col) in enumerate(zip(trc.columns, [C['blue'], C['green'], C['graph']])):
    ax.plot(xpos + (j - 1) * 0.16, trc.loc[ks, name], 'o', ms=5, color=col,
            label=name, mec='white', mew=0.7)
ax.axhline(1, color='#999999', lw=0.8, ls=':')
ax.set_xticks(xpos)
ax.set_xticklabels([k.replace('cohort_', '') for k in ks], rotation=35, ha='right')
ax.set_ylabel('Time ratio')
ax.set_title('(c) Parametric form', loc='left')
ax.legend(frameon=False, fontsize=6.5, loc='upper left')
ax.grid(axis='y')

plt.tight_layout()
save_figure(fig, 'figS2_robustness')
plt.close()
print('\nfigure written.')
