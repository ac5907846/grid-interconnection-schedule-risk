"""Geography of pending capacity and the data center market county test."""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, mannwhitneyu

from utils import C, DERIVED, TAB, save_figure, set_style, df_to_md

warnings.filterwarnings('ignore')

set_style()
d = pd.read_pickle(DERIVED / 'analysis_dataset.pkl')
CENSOR = pd.Timestamp('2025-12-31')

active = d[d['event'] == 0].copy()
active['wait_years'] = (CENSOR - active['q_date']).dt.days / 365.25
st = active.groupby('state').agg(
    projects=('mw', 'size'), capacity_gw=('mw', lambda s: s.sum() / 1000),
    median_wait=('wait_years', 'median'), share_dc=('dc_market', 'mean')).sort_values(
    'capacity_gw', ascending=False)
df_to_md(st.head(25).round(2), TAB / 'table_active_by_state.md', floatfmt='.2f')
print(st.head(15).round(2).to_string())

comp = d[d['event'] == 1].copy()
a = comp[comp['dc_market'] == 1]['T_years']
b = comp[comp['dc_market'] == 0]['T_years']
u, pu = mannwhitneyu(a, b)
dc_tbl = pd.DataFrame({
    'Data center market counties': [int(d['dc_market'].sum()),
                                    d.loc[d['dc_market'] == 1, 'mw'].sum() / 1000,
                                    float(a.median()),
                                    float(active.loc[active['dc_market'] == 1, 'wait_years'].median()),
                                    float((d.loc[d['dc_market'] == 1, 'event'] == 2).mean())],
    'All other counties': [int((d['dc_market'] == 0).sum()),
                           d.loc[d['dc_market'] == 0, 'mw'].sum() / 1000,
                           float(b.median()),
                           float(active.loc[active['dc_market'] == 0, 'wait_years'].median()),
                           float((d.loc[d['dc_market'] == 0, 'event'] == 2).mean())]},
    index=['Projects', 'Capacity requested (GW)', 'Median request-to-COD (yr)',
           'Median wait of still-active projects (yr)', 'Share withdrawn'])
df_to_md(dc_tbl.round(3), TAB / 'table_dc_market_comparison.md')
(TAB / 'mannwhitney_test.md').write_text(
    '# Mann-Whitney U test on realized durations\n\n'
    'Completed projects in data center market counties versus all other counties.\n\n'
    f'- U = {u:.0f}\n- p = {pu:.4f}\n- n (data center markets) = {len(a):,}; n (other) = {len(b):,}\n'
    f'- Median duration: {a.median():.2f} vs {b.median():.2f} years\n')
print(f'\nMann-Whitney U = {u:.0f}, p = {pu:.4f}; medians {a.median():.2f} vs {b.median():.2f}')

fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.1),
                         gridspec_kw={'width_ratios': [1.15, 1.0, 1.0]})

ax = axes[0]
top = st.head(12).iloc[::-1]
ax.barh(top.index, top['capacity_gw'], color=C['blue'], height=0.62)
for i, (v, w) in enumerate(zip(top['capacity_gw'], top['median_wait'])):
    ax.text(v + 8, i, f'{v:.0f}', va='center', fontsize=7, color=C['mut'])
ax.set_xlabel('Active capacity in queue (GW)')
ax.set_title('(a) Where pending capacity sits', loc='left')
ax.set_xlim(0, top['capacity_gw'].max() * 1.16)

ax = axes[1]
order = active.groupby('region')['wait_years'].median().sort_values().index.tolist()
for i, reg in enumerate(order):
    g = active[active['region'] == reg]['wait_years']
    if len(g) < 30:
        continue
    kde = gaussian_kde(g, bw_method=0.35)
    xs = np.linspace(0, 10, 220)
    ys = kde(xs); ys = ys / ys.max() * 0.85
    ax.fill_between(xs, i, i + ys, color=C['blue'], alpha=0.22, lw=0)
    ax.plot(xs, i + ys, color=C['blue'], lw=1.0)
    ax.plot([g.median()] * 2, [i, i + 0.88], color=C['green'], lw=1.5)
    ax.text(10.2, i + 0.3, f'{reg} ({g.median():.1f})', fontsize=7, va='center', color=C['ink'])
ax.set_yticks([]); ax.set_xlim(0, 10)
ax.spines['left'].set_visible(False)
ax.set_xlabel('Years already waiting')
ax.set_title('(b) Time in queue, active projects', loc='left')

ax = axes[2]
bins = np.arange(0, 12.5, 0.75)
ax.hist(b, bins=bins, density=True, color=C['grey'], alpha=0.85, label='Other counties')
ax.hist(a, bins=bins, density=True, histtype='step', lw=1.8, color=C['blue'],
        label='Data center markets')
ax.axvline(b.median(), color=C['ink'], lw=1.0, ls=':')
ax.axvline(a.median(), color=C['blue'], lw=1.0, ls=':')
ax.set_xlabel('Realized request-to-COD (years)')
ax.set_ylabel('Density')
ax.set_title('(c) Duration by county type', loc='left')
ax.legend(frameon=False, loc='upper right')
ax.text(0.97, 0.47, f'medians {a.median():.1f} vs {b.median():.1f}\nMann-Whitney p = {pu:.3f}',
        transform=ax.transAxes, ha='right', fontsize=6.6, color=C['mut'])
plt.tight_layout()
save_figure(fig, 'fig_extra_geography_panels')
plt.close()
print('\nfigure written.')
