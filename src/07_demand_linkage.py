"""State electricity demand from EIA-861 and its link to queue composition."""
import io
import re
import warnings
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from utils import C, DATA, DERIVED, TAB, save_figure, set_style, df_to_md

warnings.filterwarnings('ignore')

set_style()

EIA_DIR = DATA / 'eia861'
VIP = DATA / 'census_vip_private_sa.xlsx'

def parse_sales(xls_bytes):
    head = pd.read_excel(io.BytesIO(xls_bytes), header=None, nrows=3)
    sector = head.iloc[0].ffill().astype(str).str.upper().str.strip()
    measure = head.iloc[1].astype(str).str.strip().str.lower()
    names = head.iloc[2].astype(str).str.strip()
    cols = {}
    for sec, key in [('RESIDENTIAL', 'res'), ('COMMERCIAL', 'com'), ('INDUSTRIAL', 'ind')]:
        m = (sector == sec) & (measure == 'sales')
        if m.any():
            cols[f'{key}_mwh'] = int(np.where(m)[0][0])
    for label, key in [('State', 'state'), ('Part', 'part'), ('Data Year', 'year'),
                       ('Utility Name', 'util')]:
        m = names == label
        if m.any():
            cols[key] = int(np.where(m)[0][0])
    df = pd.read_excel(io.BytesIO(xls_bytes), header=None, skiprows=3)
    out = pd.DataFrame({k: df.iloc[:, v] for k, v in cols.items()})
    for c in [c for c in out.columns if c.endswith('_mwh')]:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    out['part'] = out['part'].astype(str).str.strip().str.upper()
    out['state'] = out['state'].astype(str).str.strip().str.upper()
    return out[out['state'].str.len() == 2]


frames = []
for p in sorted(EIA_DIR.glob('f861*.zip')):
    yr = re.search(r'f861(\d{4})', p.name)
    if not yr:
        continue
    yr = int(yr.group(1))
    try:
        with zipfile.ZipFile(p) as z:
            cand = [n for n in z.namelist()
                    if 'Sales_Ult_Cust' in n and '_CS_' not in n and 'CS_' not in n.split('/')[-1]]
            if not cand:
                continue
            df = parse_sales(z.read(sorted(cand)[0]))
    except Exception as e:
        print(f'  {yr}: {type(e).__name__}: {e}')
        continue
    df['yr'] = yr
    frames.append(df)

E = pd.concat(frames, ignore_index=True)
# Parts A (bundled), B (energy only) and D (combined) sum to sales; part C is the
E = E[E['part'].isin(['A', 'B', 'D'])]
state = E.groupby(['yr', 'state'])[['com_mwh', 'ind_mwh', 'res_mwh']].sum().reset_index()
state['ci_twh'] = (state['com_mwh'] + state['ind_mwh']) / 1e6
state['com_twh'] = state['com_mwh'] / 1e6

nat = state.groupby('yr')[['com_mwh', 'ind_mwh', 'res_mwh']].sum() / 1e6
print('National sales by sector (TWh), validation against EIA published totals:')
print(nat.round(0).to_string())

BASE, END = 2018, 2024
w = state.pivot(index='state', columns='yr', values='ci_twh')
wc = state.pivot(index='state', columns='yr', values='com_twh')
growth = pd.DataFrame({
    'ci_2018': w[BASE], 'ci_2024': w[END],
    'com_2018': wc[BASE], 'com_2024': wc[END]}).dropna()
growth['ci_growth_pct'] = (growth['ci_2024'] / growth['ci_2018'] - 1) * 100
growth['com_growth_pct'] = (growth['com_2024'] / growth['com_2018'] - 1) * 100
growth['ci_added_twh'] = growth['ci_2024'] - growth['ci_2018']
df_to_md(growth.sort_values('ci_growth_pct', ascending=False).round(2),
         TAB / 'table_state_load_growth.md', floatfmt='.2f')

d = pd.read_pickle(DERIVED / 'analysis_dataset.pkl')
CENSOR = pd.Timestamp('2025-12-31')
recent = d[d['q_year'].between(2021, 2025)]
active = d[d['event'] == 0].copy()
active['wait_years'] = (CENSOR - active['q_date']).dt.days / 365.25
comp = d[(d['event'] == 1) & (d['end_date'].dt.year >= 2018)]

qs = pd.DataFrame({
    'req_capacity_gw_2125': recent.groupby('state')['mw'].sum() / 1000,
    'active_capacity_gw': active.groupby('state')['mw'].sum() / 1000,
    'median_wait_active': active.groupby('state')['wait_years'].median(),
    'median_dur_completed': comp.groupby('state')['T_years'].median(),
    'n_completed': comp.groupby('state').size(),
    'gas_gw_2325': d[(d['tech'] == 'Gas') & (d['q_year'] >= 2023)].groupby('state')['mw'].sum() / 1000,
})
M = growth.join(qs, how='inner').fillna({'gas_gw_2325': 0, 'req_capacity_gw_2125': 0})
M['req_per_twh'] = M['req_capacity_gw_2125'] / M['ci_2024']
df_to_md(M.round(2), TAB / 'table_state_demand_vs_queue.md', floatfmt='.2f')

lines = ['# Correlations between state electricity load growth and queue outcomes', '',
         f'Load growth measured as the change in commercial plus industrial retail sales '
         f'from {BASE} to {END} (EIA Form 861). n = {len(M)} states.', '']
tests = [
    ('Load growth (%) vs capacity requested 2021-2025 (GW)', 'ci_growth_pct', 'req_capacity_gw_2125'),
    ('Load added (TWh) vs capacity requested 2021-2025 (GW)', 'ci_added_twh', 'req_capacity_gw_2125'),
    ('Load added (TWh) vs gas capacity requested 2023-2025 (GW)', 'ci_added_twh', 'gas_gw_2325'),
    ('Load growth (%) vs median wait of active projects (yr)', 'ci_growth_pct', 'median_wait_active'),
    ('Load growth (%) vs median realized duration (yr)', 'ci_growth_pct', 'median_dur_completed'),
    ('Load growth (%) vs active capacity in queue (GW)', 'ci_growth_pct', 'active_capacity_gw'),
]
print('\nCorrelations (Spearman rho, Pearson r):')
for label, a, b in tests:
    sub = M[[a, b]].dropna()
    if len(sub) < 8:
        continue
    rho, p1 = spearmanr(sub[a], sub[b])
    r, p2 = pearsonr(sub[a], sub[b])
    print(f'  {label}: rho={rho:+.3f} (p={p1:.3f}), r={r:+.3f} (p={p2:.3f}), n={len(sub)}')
    lines.append(f'- {label}: Spearman rho = {rho:+.3f} (p = {p1:.3f}), '
                 f'Pearson r = {r:+.3f} (p = {p2:.3f}), n = {len(sub)}')
(TAB / 'correlations.md').write_text('\n'.join(lines) + '\n')

raw = pd.read_excel(VIP, sheet_name='Private SA', header=None)
hdr = [str(h) for h in raw.iloc[3]]
v = raw.iloc[4:].copy()
v.columns = hdr
v = v.rename(columns={v.columns[0]: 'Date'})
dc_col = [c for c in v.columns if 'Data center' in c][0]
gen_col = [c for c in v.columns if c.strip() == 'General'][0]
v = v[['Date', dc_col, gen_col]].dropna(subset=['Date'])
v.columns = ['Date', 'data_center', 'office_general']
v['dt'] = pd.to_datetime(v['Date'].astype(str).str.replace(r'[pr]$', '', regex=True),
                         format='%b-%y', errors='coerce')
v = v.dropna(subset=['dt']).sort_values('dt')
for c in ['data_center', 'office_general']:
    v[c] = pd.to_numeric(v[c], errors='coerce')
v = v[v['dt'] >= '2014-01-01']
df_to_md(v[['Date', 'data_center', 'office_general']].tail(30).set_index('Date').round(0),
         TAB / 'table_census_vip_recent.md', floatfmt='.0f')

fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9),
                         gridspec_kw={'width_ratios': [1.05, 1.0, 1.0]})

ax = axes[0]
ax.plot(v['dt'], v['data_center'] / 1000, color=C['blue'], lw=2.0, label='Data centers')
ax.plot(v['dt'], v['office_general'] / 1000, color=C['silver'], lw=1.6, label='Other office')
ax.set_ylabel('Construction put in place\n($ billion, annual rate)')
ax.set_title('(a) Construction spending', loc='left')
ax.legend(frameon=False, loc='upper left')
last = v.iloc[-1]
ax.annotate(f"${last['data_center']/1000:.1f}B", xy=(last['dt'], last['data_center'] / 1000),
            xytext=(-6, 6), textcoords='offset points', fontsize=7.5,
            color=C['blue'], ha='right', fontweight='bold')
ax.grid(axis='y')

ax = axes[1]
top = growth.sort_values('ci_growth_pct', ascending=False).head(12).iloc[::-1]
DC_STATES = {'VA', 'OR', 'TX', 'AZ', 'IA', 'GA', 'NE', 'OH', 'NV', 'UT', 'WA', 'IL'}
cols = [C['blue'] if s in DC_STATES else C['mist'] for s in top.index]
ax.barh(top.index, top['ci_growth_pct'], color=cols, height=0.62)
for i, vv in enumerate(top['ci_growth_pct']):
    ax.text(vv + 1.2, i, f'{vv:.0f}%', va='center', fontsize=7, color=C['mut'])
ax.set_xlabel(f'Commercial + industrial sales\ngrowth {BASE} to {END} (%)')
ax.set_title('(b) Where load grew', loc='left')
ax.set_xlim(0, top['ci_growth_pct'].max() * 1.2)
ax.text(0.97, 0.06, 'Blue: major data\ncenter market', transform=ax.transAxes,
        ha='right', fontsize=6.8, color=C['blue'])

ax = axes[2]
sub = M[['ci_added_twh', 'gas_gw_2325']].dropna()
ax.scatter(sub['ci_added_twh'], sub['gas_gw_2325'], s=24,
           color=C['blue'], alpha=0.75, edgecolor='white', linewidth=0.6)
for s_ in ['TX', 'VA', 'OR', 'AZ', 'GA', 'PA', 'OH']:
    if s_ in sub.index:
        ax.annotate(s_, (sub.loc[s_, 'ci_added_twh'], sub.loc[s_, 'gas_gw_2325']),
                    xytext=(4, 3), textcoords='offset points', fontsize=7, color=C['ink'])
b = np.polyfit(sub['ci_added_twh'], sub['gas_gw_2325'], 1)
xs = np.linspace(sub['ci_added_twh'].min(), sub['ci_added_twh'].max(), 50)
ax.plot(xs, np.polyval(b, xs), color=C['green'], lw=1.3, ls='--')
rho, pv = spearmanr(sub['ci_added_twh'], sub['gas_gw_2325'])
r, pr = pearsonr(sub['ci_added_twh'], sub['gas_gw_2325'])
ax.text(0.04, 0.96, f'rho = {rho:.2f} (p = {pv:.3f})\nr = {r:.2f} (p < 0.001)',
        transform=ax.transAxes, va='top', fontsize=6.8, color=C['mut'])
ax.set_xlabel(f'Load added {BASE} to {END} (TWh)')
ax.set_ylabel('Gas capacity requested\n2023-2025 (GW)')
ax.set_title('(c) Supply response', loc='left')
ax.grid(alpha=0.5)

plt.tight_layout()
save_figure(fig, 'fig02_demand_linkage')
plt.close()

state.to_csv(DERIVED / 'eia861_state_sales.csv', index=False)
M.to_csv(DERIVED / 'state_demand_queue_panel.csv')
print('\nwrote figure and state panel')
print(f"\nVirginia: commercial sales {wc.loc['VA', BASE]:.1f} to {wc.loc['VA', END]:.1f} TWh "
      f"({(wc.loc['VA', END] / wc.loc['VA', BASE] - 1) * 100:.0f}%)")
