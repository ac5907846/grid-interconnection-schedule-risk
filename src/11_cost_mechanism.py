"""Interconnection study costs merged onto the queue; late-entry hazard models."""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from utils import C, DATA, DERIVED, TAB, save_figure, set_style, df_to_md

warnings.filterwarnings('ignore')

COSTS = DATA / 'costs'
set_style()

TO24_21 = 313.69 / 270.97
TO24_22 = 313.69 / 292.65


def norm_id(s):
    return s.astype(str).str.strip().str.upper()


frames = []

pjm = pd.read_excel(COSTS / 'pjm_costs_2022_clean_data.xlsx', sheet_name='data')
frames.append(pd.DataFrame({
    'region': 'PJM', 'q_id_s': norm_id(pjm['Project #']),
    'study_date': pd.to_datetime(pjm['Study Date'], errors='coerce'),
    'poi_kw': pd.to_numeric(pjm['$2022 POI Cost/kW'], errors='coerce') * TO24_22,
    'net_kw': pd.to_numeric(pjm['$2022 Network Cost/kW'], errors='coerce') * TO24_22,
    'tot_kw': pd.to_numeric(pjm['$2022 Total Cost/kW'], errors='coerce') * TO24_22}))

miso = pd.read_excel(COSTS / 'miso_costs_2021_clean_data.xlsx', sheet_name='data')
frames.append(pd.DataFrame({
    'region': 'MISO', 'q_id_s': norm_id(miso['Project #']),
    'study_date': pd.to_datetime(miso['Study Date'], errors='coerce'),
    'poi_kw': pd.to_numeric(miso['Real POI/kW'], errors='coerce') * TO24_21,
    'net_kw': pd.to_numeric(miso['Real Network/kW'], errors='coerce') * TO24_21,
    'tot_kw': pd.to_numeric(miso['Real Total/kW'], errors='coerce') * TO24_21}))

spp = pd.read_excel(COSTS / 'spp_costs_2023_clean_data.xlsx', sheet_name='data')
frames.append(pd.DataFrame({
    'region': 'SPP', 'q_id_s': norm_id(spp['Project #']),
    'study_date': pd.to_datetime(spp['Study Date'], errors='coerce'),
    'poi_kw': pd.to_numeric(spp['$2022 POI Cost/kW'], errors='coerce') * TO24_22,
    'net_kw': pd.to_numeric(spp['$2022 Network Cost/kW'], errors='coerce') * TO24_22,
    'tot_kw': pd.to_numeric(spp['$2022 Total Cost/kW'], errors='coerce') * TO24_22}))

ny = pd.read_excel(COSTS / 'nyiso_costs_2022_clean_data.xlsx', sheet_name='data')
ny_ids = norm_id(ny['Queue ID'])
frames.append(pd.DataFrame({
    'region': 'NYISO', 'q_id_s': ny_ids, 'q_id_alt': norm_id(ny['Queue ID 2']),
    'study_date': pd.to_datetime(ny['Study Date'], errors='coerce'),
    'poi_kw': pd.to_numeric(ny['$2022 POI Cost/kW'], errors='coerce') * TO24_22,
    'net_kw': pd.to_numeric(ny['$2022 Network Cost/kW'], errors='coerce') * TO24_22,
    'tot_kw': pd.to_numeric(ny['$2022 Total Cost/kW'], errors='coerce') * TO24_22}))

ne = pd.read_excel(COSTS / 'isone_costs_2021_clean_data.xlsx', sheet_name='data')
frames.append(pd.DataFrame({
    'region': 'ISO-NE', 'q_id_s': norm_id(ne['Queue ID 1']),
    'q_id_alt': norm_id(ne['Queue ID 2']),
    'study_date': pd.to_datetime(ne['Study Date'], errors='coerce'),
    'poi_kw': pd.to_numeric(ne['$2022 POI Cost/kW'], errors='coerce') * TO24_22,
    'net_kw': pd.to_numeric(ne['$2022 Network Cost/kW'], errors='coerce') * TO24_22,
    'tot_kw': pd.to_numeric(ne['$2022 Total Cost/kW'], errors='coerce') * TO24_22}))

ba = pd.read_excel(COSTS / 'ba_costs_2024_clean_data.xlsx',
                   sheet_name='Project Cost Data', header=0)
ba = ba[ba['Project # in Queued Up'].notna()].copy()
ba_f = pd.DataFrame({
    'region': 'BA', 'q_id_s': norm_id(ba['Project # in Queued Up']),
    'state': ba['State'].astype(str).str.strip().str.upper(),
    'study_date': pd.to_datetime(ba['Study Date'], errors='coerce'),
    'poi_kw': pd.to_numeric(ba['$2024 POI Cost/kW'], errors='coerce'),
    'net_kw': pd.to_numeric(ba['$2024 Network Cost/kW'], errors='coerce'),
    'tot_kw': pd.to_numeric(ba['$2024 Total Cost/kW'], errors='coerce')})
ba_f = (ba_f.dropna(subset=['tot_kw']).sort_values('study_date')
        .groupby(['q_id_s', 'state'], as_index=False).last())
frames.append(ba_f)

cost = pd.concat(frames, ignore_index=True)
cost = cost.dropna(subset=['tot_kw'])
cost['net_kw'] = cost['net_kw'].fillna(0)

d = pd.read_pickle(DERIVED / 'analysis_dataset.pkl')
d['q_id_s'] = norm_id(d['q_id'])
d['state_s'] = d['state'].astype(str).str.strip().str.upper()

merged = []
for reg in ['PJM', 'MISO', 'SPP', 'NYISO', 'ISO-NE']:
    left = d[d['region'] == reg]
    right = cost[cost['region'] == reg].drop_duplicates('q_id_s')
    m = left.merge(right[['q_id_s', 'study_date', 'poi_kw', 'net_kw', 'tot_kw']],
                   on='q_id_s', how='inner')
    if 'q_id_alt' in cost.columns:
        alt = cost[(cost['region'] == reg) & cost.get('q_id_alt').notna()]
        if len(alt):
            unmatched = left[~left['q_id_s'].isin(m['q_id_s'])]
            m2 = unmatched.merge(
                alt.drop_duplicates('q_id_alt')[
                    ['q_id_alt', 'study_date', 'poi_kw', 'net_kw', 'tot_kw']],
                left_on='q_id_s', right_on='q_id_alt', how='inner').drop(columns='q_id_alt')
            m = pd.concat([m, m2], ignore_index=True)
    m['cost_region'] = reg
    merged.append(m)

left = d[d['region'].isin(['Southeast', 'West'])]
right = cost[cost['region'] == 'BA']
m = left.merge(right[['q_id_s', 'state', 'study_date', 'poi_kw', 'net_kw', 'tot_kw']],
               left_on=['q_id_s', 'state_s'], right_on=['q_id_s', 'state'], how='inner')
m['cost_region'] = 'Non-ISO BA'
merged.append(m.drop(columns='state_y').rename(columns={'state_x': 'state'}))

mg = pd.concat(merged, ignore_index=True)
mg = mg.drop_duplicates(subset=['q_id_s', 'cost_region'])

rates = pd.DataFrame({
    'Cost records': cost.groupby(cost['region'].replace({'BA': 'Non-ISO BA'})).size(),
    'Matched to queue': mg.groupby('cost_region').size()})
rates['Match rate'] = (rates['Matched to queue'] / rates['Cost records']).round(2)
rates.index.name = 'Territory'
df_to_md(rates, TAB / 'table_match_rates.md', floatfmt='.2f')
print(rates)

lab = {0: 'Active', 1: 'Completed', 2: 'Withdrawn'}
mg['outcome'] = mg['event'].map(lab)
by = mg.groupby('outcome').agg(
    n=('net_kw', 'size'),
    med_net=('net_kw', 'median'), p75_net=('net_kw', lambda s: s.quantile(.75)),
    med_poi=('poi_kw', 'median'), med_tot=('tot_kw', 'median'))
by.columns = ['Projects', 'Median network cost ($/kW)', 'P75 network cost ($/kW)',
              'Median POI cost ($/kW)', 'Median total cost ($/kW)']
df_to_md(by.round(1), TAB / 'table_cost_by_outcome.md', floatfmt='.1f')
print(by.round(1))

s = mg.dropna(subset=['study_date']).copy()
s['entry_t'] = (s['study_date'] - s['q_date']).dt.days / 365.25
s = s[(s['entry_t'] >= 0) & (s['entry_t'] < s['T_years'])]
s['log_net'] = np.log1p(s['net_kw'].clip(lower=0))
s['log_tot'] = np.log1p(s['tot_kw'].clip(lower=0))
q_edges = s['net_kw'].quantile([0, .25, .5, .75, 1.0]).values
q_edges[0] -= 1e-6
s['cost_q'] = pd.cut(s['net_kw'], q_edges, labels=['Q1', 'Q2', 'Q3', 'Q4'])
print(f'\nlandmark sample: {len(s)} projects '
      f'({(s.event == 2).sum()} withdrawn, {(s.event == 1).sum()} completed)')
print('network cost quartile edges (2024 $/kW):', np.round(q_edges[1:], 1))

BASE = ['log_mw', 'q_year']
TECH = pd.get_dummies(s['tech'], prefix='tech', drop_first=True).astype(float)
REG = pd.get_dummies(s['cost_region'], prefix='reg', drop_first=True).astype(float)


def fit_cox(event_code, xtra_cols, frame):
    f = pd.concat([frame[['T_years', 'entry_t'] + BASE], xtra_cols,
                   TECH.loc[frame.index], REG.loc[frame.index]], axis=1)
    f['E'] = (frame['event'] == event_code).astype(int)
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(f, duration_col='T_years', event_col='E', entry_col='entry_t')
    return cph


rows = []
for ev, name in [(2, 'Withdrawal'), (1, 'Completion')]:
    cont = fit_cox(ev, s[['log_net']], s)
    r = cont.summary.loc['log_net']
    rows.append([name, 'log(1+network $/kW)', np.exp(r['coef']),
                 np.exp(r['coef'] - 1.96 * r['se(coef)']),
                 np.exp(r['coef'] + 1.96 * r['se(coef)']), r['p']])
    quart = fit_cox(ev, pd.get_dummies(s['cost_q'], prefix='cq',
                                       drop_first=True).astype(float), s)
    for q in ['cq_Q2', 'cq_Q3', 'cq_Q4']:
        r = quart.summary.loc[q]
        rows.append([name, q.replace('cq_', 'Cost quartile ') + ' vs Q1',
                     np.exp(r['coef']), np.exp(r['coef'] - 1.96 * r['se(coef)']),
                     np.exp(r['coef'] + 1.96 * r['se(coef)']), r['p']])

cox_tbl = pd.DataFrame(rows, columns=['Event', 'Covariate', 'HR', 'CI low',
                                      'CI high', 'p']).set_index(['Event', 'Covariate'])
df_to_md(cox_tbl.round(3), TAB / 'table_cost_cox.md', floatfmt='.3f')
print(cox_tbl.round(3))

s['t_after'] = s['T_years'] - s['entry_t']
H = 3.0
qt = []
for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    g = s[s['cost_q'] == q]
    dec = g[(g['t_after'] <= H) & g['event'].isin([1, 2])]
    atrisk = g[(g['t_after'] > H) | g['event'].isin([1, 2])]
    qt.append([q, len(g), g['net_kw'].median(),
               (dec['event'] == 2).sum() / max(len(atrisk), 1),
               (dec['event'] == 1).sum() / max(len(atrisk), 1)])
qt = pd.DataFrame(qt, columns=['Quartile', 'Projects', 'Median network $/kW',
                               'Withdrawn within 3y of study',
                               'Completed within 3y of study']).set_index('Quartile')
df_to_md(qt.round(3), TAB / 'table_cost_quartiles.md', floatfmt='.3f')
print(qt.round(3))

trn = s[s['study_date'].dt.year <= 2017]
tst = s[s['study_date'].dt.year >= 2018]
from lifelines.utils import concordance_index
pred_rows = []
for label, cols in [('Without cost', s[[]]), ('With cost', s[['log_net']])]:
    f_trn = pd.concat([trn[['T_years', 'entry_t'] + BASE], cols.loc[trn.index],
                       TECH.loc[trn.index], REG.loc[trn.index]], axis=1)
    f_trn['E'] = (trn['event'] == 2).astype(int)
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(f_trn, duration_col='T_years', event_col='E', entry_col='entry_t')
    f_tst = pd.concat([tst[['T_years'] + BASE], cols.loc[tst.index],
                       TECH.loc[tst.index], REG.loc[tst.index]], axis=1)
    ci = concordance_index(tst['T_years'], -cph.predict_partial_hazard(f_tst),
                           (tst['event'] == 2).astype(int))
    pred_rows.append([label, len(f_trn), len(tst), ci])
pred = pd.DataFrame(pred_rows, columns=['Withdrawal model', 'Train projects',
                                        'Test projects', 'Temporal c-index']).set_index('Withdrawal model')
df_to_md(pred.round(3), TAB / 'table_cost_prediction.md', floatfmt='.3f')
print(pred.round(3))

fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.75))

ax = axes[0]
order = ['Completed', 'Active', 'Withdrawn']
colors = {'Completed': C['green'], 'Active': C['silver'], 'Withdrawn': C['blue']}
data = [np.log10(mg.loc[mg['outcome'] == o, 'net_kw'].clip(lower=0.5)) for o in order]
bp = ax.boxplot(data, tick_labels=order, showfliers=False, patch_artist=True,
                widths=0.55, medianprops=dict(color=C['ink'], lw=1.4))
for patch, o in zip(bp['boxes'], order):
    patch.set_facecolor(colors[o]); patch.set_alpha(0.85); patch.set_edgecolor(C['char'])
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(['1', '10', '100', '1,000'])
ax.set_ylabel('Network upgrade cost (2024 $/kW)')
ax.set_title('(a) Allocated cost by outcome', loc='left')
ax.grid(axis='y')

ax = axes[1]
w = cox_tbl.loc['Withdrawal'].iloc[1:]
c_ = cox_tbl.loc['Completion'].iloc[1:]
y = np.arange(3)
ax.errorbar(w['HR'], y + 0.14, xerr=[w['HR'] - w['CI low'], w['CI high'] - w['HR']],
            fmt='o', color=C['blue'], ms=4.5, capsize=2.5, lw=1.2, label='Withdrawal')
ax.errorbar(c_['HR'], y - 0.14, xerr=[c_['HR'] - c_['CI low'], c_['CI high'] - c_['HR']],
            fmt='s', color=C['green'], ms=4.5, capsize=2.5, lw=1.2, label='Completion')
ax.axvline(1.0, color=C['graph'], lw=0.9, ls='--')
ax.set_yticks(y)
ax.set_yticklabels(['Q2 vs Q1', 'Q3 vs Q1', 'Q4 vs Q1'])
ax.set_xlabel('Hazard ratio (95% CI)')
ax.set_title('(b) Exit hazards by cost quartile', loc='left')
ax.legend(frameon=False, fontsize=7, loc='lower right')
ax.grid(axis='x')

ax = axes[2]
sy = mg.dropna(subset=['study_date']).copy()
sy['study_year'] = sy['study_date'].dt.year
med = sy[sy['study_year'].between(2005, 2024)].groupby('study_year')['net_kw'].median()
n = sy[sy['study_year'].between(2005, 2024)].groupby('study_year').size()
ax.plot(med.index, med.values, color=C['char'], lw=1.8, marker='o', ms=3.2, mfc='white')
ax.axvline(2021.0, color=C['blue'], lw=0.9, ls='--')
ax.annotate('2021Q1 break', xy=(2020.6, med.max() * 0.70), fontsize=7,
            color=C['blue'], ha='right')
ax.set_ylabel('Median network cost (2024 $/kW)')
ax.set_xlabel('Study year')
ax.set_title('(c) Cost escalation', loc='left')
ax.grid(axis='y')

plt.tight_layout()
save_figure(fig, 'fig08_cost_mechanism')
plt.close()

mg.drop(columns=['q_id_s', 'state_s'], errors='ignore').to_csv(
    DERIVED / 'cost_merge.csv', index=False)
print('\ndone')
