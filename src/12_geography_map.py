"""Composite geography figure: facility map, regional waits, county comparison."""
import json

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from utils import C, DATA, DERIVED, SLATE_CMAP, save_figure, set_style

set_style()

gj = json.load(open(DATA / 'us_counties.geojson'))
cty = gpd.GeoDataFrame.from_features(gj['features'])
cty['fips'] = [f['id'] for f in gj['features']]
cty = cty.set_crs('EPSG:4326')
cty['state_fips'] = cty['fips'].str[:2]
conus = cty[~cty['state_fips'].isin(['02', '15', '60', '66', '69', '72', '78'])].to_crs('EPSG:5070')
states = conus.dissolve(by='state_fips', as_index=False)[['state_fips', 'geometry']]

d = pd.read_pickle(DERIVED / 'analysis_dataset.pkl')
ST2FIPS = {'AL': '01', 'AZ': '04', 'AR': '05', 'CA': '06', 'CO': '08', 'CT': '09',
           'DE': '10', 'DC': '11', 'FL': '12', 'GA': '13', 'ID': '16', 'IL': '17',
           'IN': '18', 'IA': '19', 'KS': '20', 'KY': '21', 'LA': '22', 'ME': '23',
           'MD': '24', 'MA': '25', 'MI': '26', 'MN': '27', 'MS': '28', 'MO': '29',
           'MT': '30', 'NE': '31', 'NV': '32', 'NH': '33', 'NJ': '34', 'NM': '35',
           'NY': '36', 'NC': '37', 'ND': '38', 'OH': '39', 'OK': '40', 'OR': '41',
           'PA': '42', 'RI': '44', 'SC': '45', 'SD': '46', 'TN': '47', 'TX': '48',
           'UT': '49', 'VT': '50', 'VA': '51', 'WA': '53', 'WV': '54', 'WI': '55',
           'WY': '56'}

act = d[d['event'] == 0]
wait = act.groupby('state')['T_years'].median()
wait = wait[act.groupby('state').size() >= 30]
states['wait'] = states['state_fips'].map(
    {ST2FIPS[s]: v for s, v in wait.items() if s in ST2FIPS})

atlas = pd.read_csv(DATA / 'im3_data_center_atlas.csv')
atlas = atlas[atlas['state_abb'].isin(ST2FIPS)]
pts = gpd.GeoDataFrame(atlas, geometry=gpd.points_from_xy(atlas['lon'], atlas['lat']),
                       crs='EPSG:4326').to_crs('EPSG:5070')
sq = pts['sqft'].fillna(pts['sqft'].median()).clip(2e4, 4e6)
size = 3 + 46 * (sq - sq.min()) / (sq.max() - sq.min())

fig = plt.figure(figsize=(7.3, 6.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1.55, 1.0], hspace=0.16, wspace=0.24,
                      left=0.07, right=0.97, top=0.95, bottom=0.08)

ax = fig.add_subplot(gs[0, :])
states.plot(ax=ax, column='wait', cmap=SLATE_CMAP, edgecolor='white', linewidth=0.6,
            missing_kwds={'color': '#EDEDEA'}, vmin=1.5, vmax=5.2)
pts.plot(ax=ax, markersize=size, color=C['green'], alpha=0.6,
         edgecolor='#2E4D2A', linewidth=0.25)
sm = plt.cm.ScalarMappable(cmap=SLATE_CMAP, norm=plt.Normalize(1.5, 5.2))
cb = fig.colorbar(sm, ax=ax, shrink=0.62, pad=0.005)
cb.set_label('Median years waiting, active projects', fontsize=7.5)
cb.ax.tick_params(labelsize=7)
ax.set_axis_off()
ax.set_title('(a) Data center facilities and median queue wait by state',
             loc='left', fontsize=9)
h = [Line2D([0], [0], marker='o', ls='', mfc=C['green'], mec='#2E4D2A', ms=6,
            alpha=0.75, label='Data center facility (IM3 atlas), sized by floor area')]
ax.legend(handles=h, loc='lower left', frameon=False, fontsize=7)

ax = fig.add_subplot(gs[1, 0])
reg_order = act.groupby('region')['T_years'].median().sort_values().index.tolist()
data = [act.loc[act['region'] == r, 'T_years'].clip(upper=12).values for r in reg_order]
vp = ax.violinplot(data, vert=False, showmedians=True, showextrema=False, widths=0.8)
for body in vp['bodies']:
    body.set_facecolor(C['silver'])
    body.set_alpha(0.75)
    body.set_edgecolor(C['graph'])
    body.set_linewidth(0.5)
vp['cmedians'].set_color(C['blue'])
vp['cmedians'].set_linewidth(1.6)
ax.set_yticks(range(1, len(reg_order) + 1))
ax.set_yticklabels(reg_order, fontsize=7)
ax.set_xlabel('Years already waiting, active projects')
ax.set_title('(b) Elapsed wait of active projects by region', loc='left', fontsize=9)
ax.grid(axis='x')

ax = fig.add_subplot(gs[1, 1])
comp = d[d['event'] == 1]
a = comp.loc[comp['dc_market'] == 1, 'T_years'].clip(upper=12)
b = comp.loc[comp['dc_market'] == 0, 'T_years'].clip(upper=12)
bins = np.linspace(0, 12, 36)
ax.hist(b, bins=bins, density=True, color=C['mist'], alpha=0.9, label='Other counties')
ax.hist(a, bins=bins, density=True, histtype='step', color=C['red'], lw=1.7,
        label='Data center market counties')
ax.axvline(b.median(), color=C['graph'], lw=1.0, ls='--')
ax.axvline(a.median(), color=C['red'], lw=1.0, ls='--')
ax.text(0.98, 0.72, f'medians {a.median():.1f} vs {b.median():.1f} yr\nMann-Whitney p = 0.456',
        transform=ax.transAxes, ha='right', fontsize=7, color=C['char'])
ax.set_xlabel('Realized request-to-COD (years)')
ax.set_ylabel('Density')
ax.set_title('(c) Realized durations by county type', loc='left', fontsize=9)
ax.legend(frameon=False, fontsize=7, loc='upper right')
ax.grid(axis='y')

save_figure(fig, 'fig07_geography')
plt.close()
print(len(pts), 'facilities,', states['wait'].notna().sum(), 'states shaded')
