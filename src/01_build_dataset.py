"""Builds the analysis dataset from the LBNL interconnection queue workbook."""
import numpy as np
import pandas as pd

from utils import DATA, DERIVED, TAB, df_to_md

SRC = DATA / 'lbnl_queue_thru2025.xlsx'
CENSOR = pd.Timestamp('2025-12-31')

d = pd.read_excel(SRC, sheet_name='03. Complete Queue Data', header=1)
log = [('Raw records in complete queue data sheet', len(d))]

d = d[d['project_type'] == 'Generation']
log.append(('After keeping generation projects', len(d)))
d = d[d['q_date'].notna()]
log.append(('After requiring a queue entry date', len(d)))
d['mw'] = d[['mw_1', 'mw_2', 'mw_3']].sum(axis=1, min_count=1)
d = d[d['mw'].notna() & (d['mw'] > 0)]
log.append(('After requiring positive nameplate capacity', len(d)))
d = d[(d['q_year'] >= 2000) & (d['q_year'] <= 2025)]
log.append(('After restricting entry years to 2000-2025', len(d)))

d = d[d['q_status'] != 'unknown']
d['event'] = 0
d.loc[d['q_status'] == 'operational', 'event'] = 1
d.loc[d['q_status'] == 'withdrawn', 'event'] = 2

d['end_date'] = pd.NaT
d.loc[d['event'] == 1, 'end_date'] = d.loc[d['event'] == 1, 'on_date']
d.loc[d['event'] == 2, 'end_date'] = d.loc[d['event'] == 2, 'wd_date']
d.loc[d['event'] == 0, 'end_date'] = CENSOR
log.append(('Operational records missing a commercial operation date',
            int(((d['event'] == 1) & (d['on_date'].isna())).sum())))
log.append(('Withdrawn records missing a withdrawal date',
            int(((d['event'] == 2) & (d['wd_date'].isna())).sum())))

d = d[d['end_date'].notna()]
log.append(('After requiring a usable event or censoring date', len(d)))
d['T_years'] = (d['end_date'] - d['q_date']).dt.days / 365.25
d = d[(d['T_years'] > 0) & (d['T_years'] <= 30)]
log.append(('Final analysis sample', len(d)))

d['cohort'] = pd.cut(d['q_year'], bins=[1999, 2007, 2013, 2017, 2020, 2022, 2025],
                     labels=['2000-07', '2008-13', '2014-17', '2018-20', '2021-22', '2023-25'])
d['log_mw'] = np.log(d['mw'])
d['size_cat'] = pd.cut(d['mw'], bins=[0, 20, 100, 300, 1000, 1e5],
                       labels=['<20', '20-100', '100-300', '300-1000', '>1000'])
TECH = {'Solar': 'Solar', 'Wind': 'Wind', 'Offshore Wind': 'Wind', 'Battery': 'Battery',
        'Solar+Battery': 'Hybrid', 'Wind+Battery': 'Hybrid', 'Gas+Battery': 'Hybrid',
        'Gas': 'Gas', 'Nuclear': 'Nuclear'}
d['tech'] = d['type_clean'].map(TECH).fillna('Other')

d = d.sort_values('q_date')
parts = []
for _, grp in d.groupby('region'):
    starts, ends = grp['q_date'].values, grp['end_date'].values
    parts.append(pd.Series([((starts < t) & (ends > t)).sum() for t in starts],
                           index=grp.index))
d['backlog'] = pd.concat(parts).sort_index()
d['log_backlog'] = np.log1p(d['backlog'])

# counties of the thirty largest US data center markets
DC_FIPS = {
    51107: 'Northern Virginia', 51153: 'Northern Virginia', 51059: 'Northern Virginia',
    51061: 'Northern Virginia', 4013: 'Phoenix', 4021: 'Phoenix', 48085: 'Dallas-Fort Worth',
    48113: 'Dallas-Fort Worth', 48121: 'Dallas-Fort Worth', 48439: 'Dallas-Fort Worth',
    13067: 'Atlanta', 13121: 'Atlanta', 13135: 'Atlanta', 13223: 'Atlanta', 13097: 'Atlanta',
    41067: 'Portland', 41005: 'Portland', 41047: 'Portland', 41021: 'Portland', 41049: 'Portland',
    17031: 'Chicago', 17043: 'Chicago', 17089: 'Chicago', 39049: 'Columbus', 39041: 'Columbus',
    39089: 'Columbus', 6085: 'Silicon Valley', 6001: 'Silicon Valley', 32003: 'Las Vegas',
    32031: 'Reno', 49035: 'Salt Lake City', 49049: 'Salt Lake City', 19113: 'Central Iowa',
    19153: 'Central Iowa', 19049: 'Central Iowa', 53025: 'Central Washington',
    53037: 'Central Washington', 53017: 'Central Washington', 1089: 'Huntsville',
    37183: 'Raleigh', 37025: 'Charlotte', 45079: 'Columbia', 29189: 'St. Louis',
    47157: 'Memphis', 28033: 'North Mississippi', 22017: 'North Louisiana',
    40109: 'Oklahoma City', 40143: 'Tulsa', 31055: 'Omaha', 31153: 'Omaha',
    20091: 'Kansas City', 20209: 'Kansas City', 55133: 'Milwaukee', 55079: 'Milwaukee',
    26125: 'Detroit', 26163: 'Detroit', 12086: 'Miami', 12011: 'Miami', 35001: 'Albuquerque',
    8005: 'Denver', 8031: 'Denver', 8059: 'Denver', 48029: 'San Antonio', 48453: 'Austin',
    48491: 'Austin',
}
d['fips'] = d['fips_code'].fillna(0).astype(int)
d['dc_market'] = d['fips'].isin(DC_FIPS).astype(int)
d['dc_market_name'] = d['fips'].map(DC_FIPS)

KEEP = ['q_id', 'q_status', 'q_date', 'on_date', 'wd_date', 'ia_date', 'end_date', 'event',
        'T_years', 'county', 'state', 'fips', 'region', 'utility', 'entity', 'service',
        'type_clean', 'tech', 'mw', 'log_mw', 'size_cat', 'q_year', 'cohort',
        'backlog', 'log_backlog', 'dc_market', 'dc_market_name']
out = d[KEEP].copy()
out.to_csv(DERIVED / 'analysis_dataset.csv', index=False)
out.to_pickle(DERIVED / 'analysis_dataset.pkl')

df_to_md(pd.DataFrame(log, columns=['Step', 'Records']),
         TAB / 'table01_sample_construction.md', index=False)

summary = pd.DataFrame({
    'Projects': out.groupby('cohort', observed=True).size(),
    'Capacity (GW)': out.groupby('cohort', observed=True)['mw'].sum() / 1000,
    'Operational': out.groupby('cohort', observed=True)['event'].apply(lambda s: (s == 1).sum()),
    'Withdrawn': out.groupby('cohort', observed=True)['event'].apply(lambda s: (s == 2).sum()),
    'Still active': out.groupby('cohort', observed=True)['event'].apply(lambda s: (s == 0).sum()),
    'Median MW': out.groupby('cohort', observed=True)['mw'].median(),
})
df_to_md(summary.round(1), TAB / 'table_cohort_summary.md', floatfmt='.1f')

print(pd.DataFrame(log, columns=['Step', 'Records']).to_string(index=False))
print(f'wrote {len(out):,} rows')
