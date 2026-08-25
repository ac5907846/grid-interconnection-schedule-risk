from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
DERIVED = ROOT / 'output' / 'derived'
FIG = ROOT / 'output' / 'figures'
TAB = ROOT / 'output' / 'tables'
for _p in (DERIVED, FIG, TAB):
    _p.mkdir(parents=True, exist_ok=True)

C = {'blue': '#31597E', 'green': '#57904F', 'red': '#7A2E37', 'purple': '#7D4A6E',
     'ink': '#2B2B2B', 'char': '#3A3A3A', 'graph': '#6B6B66',
     'silver': '#9C9C96', 'mist': '#C9C9C4', 'mut': '#6B6B66', 'grey': '#C9C9C4'}

COHORT_COLORS = {'2000-07': '#DBDBD6', '2008-13': '#C9C9C4', '2014-17': '#9C9C96',
                 '2018-20': '#3A3A3A', '2021-22': '#31597E', '2023-25': '#57904F'}

SLATE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'slate', ['#F2F2EF', '#C3CDD8', '#7E96AC', '#31597E', '#1E3A56'])


def set_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.titlesize': 9.5,
        'axes.labelsize': 9, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 0.6, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'legend.fontsize': 7.5, 'grid.color': '#e6e6e3', 'grid.linewidth': 0.5,
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'axes.axisbelow': True})


def save_figure(fig, name):
    fig.savefig(FIG / f'{name}.png', dpi=300)
    fig.savefig(FIG / f'{name}.svg')
    fig.savefig(FIG / f'{name}.pdf', dpi=300)
    print(f'wrote {name}')


def df_to_md(df, path, floatfmt='.3f', index=True):
    try:
        md = df.to_markdown(floatfmt=floatfmt, index=index)
    except Exception:
        md = df.to_string()
    Path(path).write_text(md + '\n')
