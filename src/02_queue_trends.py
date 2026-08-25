"""Queue entry volumes, realized durations, structural breaks, milestone decomposition."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt

from utils import C, DERIVED, TAB, save_figure, set_style, df_to_md

set_style()
d = pd.read_pickle(DERIVED / 'analysis_dataset.pkl')
comp = d[d['event'] == 1].copy()
comp['cod_year'] = comp['end_date'].dt.year

comp['cod_q'] = comp['end_date'].dt.to_period('Q')
qs = comp.groupby('cod_q')['T_years'].median()
qs = qs[(qs.index >= pd.Period('2005Q1')) & (qs.index <= pd.Period('2025Q4'))]
bkps = rpt.Binseg(model='l2').fit(qs.values).predict(n_bkps=3)
breaks = [str(qs.index[b - 1]) for b in bkps[:-1]]
print('structural breaks:', breaks)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

ax = axes[0]
TECH_ORDER = ['Solar', 'Battery', 'Hybrid', 'Wind', 'Gas', 'Nuclear', 'Other']
TC = {'Solar': C['green'], 'Battery': C['silver'], 'Hybrid': C['mist'], 'Wind': C['purple'],
      'Gas': C['blue'], 'Nuclear': C['char'], 'Other': '#E3E3DE'}
piv = (d[d['q_year'] >= 2010]
       .pivot_table(index='q_year', columns='tech', values='mw', aggfunc='sum')
       .fillna(0) / 1000)
bottom = np.zeros(len(piv))
for t in TECH_ORDER:
    if t in piv:
        ax.bar(piv.index, piv[t], bottom=bottom, color=TC[t], width=0.72,
               label=t, edgecolor='white', linewidth=0.4)
        bottom += piv[t].values
ax.set_ylabel('Capacity entering queues (GW)')
ax.set_title('(a) Interconnection requests by technology', loc='left')
ax.legend(frameon=False, ncol=2, loc='upper left', handlelength=1, handleheight=1)
ax.set_xticks([2010, 2013, 2016, 2019, 2022, 2025])
gas25 = piv.loc[2025, 'Gas'] if 'Gas' in piv else 0
gas_top = bottom[-1] - piv.loc[2025, 'Other'] if 'Other' in piv else bottom[-1]
gas_mid = gas_top - piv.loc[2025, 'Gas'] / 2
ax.annotate(f'Gas\n{gas25:.0f} GW', xy=(2025, gas_mid), xytext=(2021.4, 470),
            fontsize=7.5, color=C['blue'], ha='center', va='center',
            arrowprops=dict(arrowstyle='-', color=C['blue'], lw=0.7,
                            connectionstyle='arc3,rad=-0.15'))
ax.set_ylim(0, 570)

ax = axes[1]
med = comp[comp['cod_year'] >= 2005].groupby('cod_year')['T_years'].median()
ax.plot(med.index, med.values, color=C['char'], lw=1.8, marker='o', ms=3.5,
        mfc='white', mew=1.2)
for b in breaks:
    yr = int(b[:4]) + (int(b[-1]) - 1) / 4
    if 2005 <= yr <= 2025:
        ax.axvline(yr, color=C['blue'], lw=0.9, ls='--', alpha=0.85)
        ax.annotate(b, xy=(yr, 6.05), fontsize=7, color=C['blue'], ha='center')
ax.text(med.index[-1] + 0.2, med.values[-1], f'{med.values[-1]:.1f} yr', fontsize=8,
        color=C['char'], va='center', fontweight='bold')
ax.set_ylabel('Median request-to-COD (years)')
ax.set_title('(b) Realized durations, completed projects', loc='left')
ax.set_ylim(0, 6.6)
ax.grid(axis='y')

plt.tight_layout()
save_figure(fig, 'fig01_queue_trends')
plt.close()

tbl = pd.DataFrame({
    'Median duration (yr)': comp.groupby('cod_year')['T_years'].median(),
    'Mean duration (yr)': comp.groupby('cod_year')['T_years'].mean(),
    'Completions': comp.groupby('cod_year').size(),
    'Capacity completed (GW)': comp.groupby('cod_year')['mw'].sum() / 1000,
}).loc[2005:2025]
df_to_md(tbl.round(2), TAB / 'table_duration_by_cod_year.md', floatfmt='.2f')

entry = pd.DataFrame({
    'Requests': d.groupby('q_year').size(),
    'Capacity (GW)': d.groupby('q_year')['mw'].sum() / 1000,
    'Gas capacity (GW)': d[d['tech'] == 'Gas'].groupby('q_year')['mw'].sum() / 1000,
    'Median project MW': d.groupby('q_year')['mw'].median(),
}).loc[2010:2025].fillna(0)
df_to_md(entry.round(1), TAB / 'table_entry_by_year.md', floatfmt='.1f')
(TAB / 'structural_breaks.md').write_text(
    '# Structural breaks (Binary segmentation, l2 cost, 3 breakpoints)\n\n'
    'Series: quarterly median request-to-COD duration of completed projects, 2005Q1-2025Q4.\n\n'
    + '\n'.join(f'- {b}' for b in breaks) + '\n')
print(tbl.round(2).tail(12).to_string())

m = d[(d['event'] == 1) & d['ia_date'].notna() & d['on_date'].notna()].copy()
m['q_to_ia'] = (m['ia_date'] - m['q_date']).dt.days / 365.25
m['ia_to_cod'] = (m['on_date'] - m['ia_date']).dt.days / 365.25
m = m[(m['q_to_ia'] >= 0) & (m['ia_to_cod'] >= 0)]
cohort_bins = pd.cut(m['q_year'], [1999, 2007, 2013, 2017, 2020, 2025],
                     labels=['2000-07', '2008-13', '2014-17', '2018-20', '2021-25'])
dec = m.groupby(cohort_bins, observed=True).agg(
    n=('q_to_ia', 'size'),
    med_q_to_ia=('q_to_ia', 'median'),
    med_ia_to_cod=('ia_to_cod', 'median'),
    med_total=('T_years', 'median'))
dec.columns = ['Projects', 'Median request to IA (yr)', 'Median IA to COD (yr)',
               'Median request to COD (yr)']
dec.index.name = 'Entry cohort'
df_to_md(dec.round(2), TAB / 'table_milestone_decomposition.md', floatfmt='.2f')
with open(TAB / 'table_milestone_decomposition.md', 'a') as f:
    f.write(f'\nCoverage: {len(m)} of {(d["event"] == 1).sum()} completed projects '
            f'({100 * len(m) / (d["event"] == 1).sum():.0f} percent) report a usable IA date; '
            'coverage varies by ISO, so the split is indicative rather than representative.\n')
print('\nMilestone decomposition:')
print(dec.round(2).to_string())
