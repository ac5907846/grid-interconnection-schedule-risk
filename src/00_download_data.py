"""Downloads the public source files into the data folder. Copies shipped with
the repository are used as-is when a download is skipped or fails."""
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / 'data'
(DATA / 'eia861').mkdir(parents=True, exist_ok=True)
(DATA / 'costs').mkdir(parents=True, exist_ok=True)
HEADERS = {'User-Agent': 'Mozilla/5.0 (research reproduction)'}

FILES = {
    DATA / 'lbnl_queue_thru2025.xlsx':
        'https://eta-publications.lbl.gov/sites/default/files/2026-05/lbnl_ix_queue_data_file_thru2025.xlsx',
    DATA / 'us_counties.geojson':
        'https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json',
    DATA / 'costs' / 'pjm_costs_2022_clean_data.xlsx':
        'https://eta-publications.lbl.gov/sites/default/files/pjm_costs_2022_clean_data.xlsx',
    DATA / 'costs' / 'miso_costs_2021_clean_data.xlsx':
        'https://eta-publications.lbl.gov/sites/default/files/miso_costs_2021_clean_data.xlsx',
    DATA / 'costs' / 'spp_costs_2023_clean_data.xlsx':
        'https://eta-publications.lbl.gov/sites/default/files/spp_costs_2023_clean_data.xlsx',
    DATA / 'costs' / 'nyiso_costs_2022_clean_data.xlsx':
        'https://eta-publications.lbl.gov/sites/default/files/nyiso_2022_final_data_cleaned_publication_vfinal.xlsx',
    DATA / 'costs' / 'isone_costs_2021_clean_data.xlsx':
        'https://eta-publications.lbl.gov/sites/default/files/isone_interconnection_cost_data_publication_vfinal.xlsx',
    DATA / 'costs' / 'ba_costs_2024_clean_data.xlsx':
        'https://eta-publications.lbl.gov/sites/default/files/2026-02/ba_costs_2024_clean_data.xlsx',
}
for year in range(2012, 2025):
    FILES[DATA / 'eia861' / f'f861{year}.zip'] = \
        f'https://www.eia.gov/electricity/data/eia861/zip/f861{year}.zip'


def fetch(dest, url):
    if dest.exists() and dest.stat().st_size > 10_000:
        print(f'  have {dest.name}')
        return
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=180) as r:
            dest.write_bytes(r.read())
        print(f'  fetched {dest.name} ({dest.stat().st_size:,} bytes)')
    except Exception as e:
        print(f'  FAILED {dest.name}: {e}')


if __name__ == '__main__':
    for dest, url in FILES.items():
        fetch(dest, url)
    vip = DATA / 'census_vip_private_sa.xlsx'
    im3 = DATA / 'im3_data_center_atlas.csv'
    print(f'  {"have" if vip.exists() else "MISSING"} {vip.name} (June 2026 vintage, shipped)')
    print(f'  {"have" if im3.exists() else "MISSING"} {im3.name} (v2026.02.09, shipped; '
          'source: MSD-LIVE record 65g71-a4731)')
