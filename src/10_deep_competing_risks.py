"""Deep competing-risks model with strictly temporal validation."""
import json
import os
import tempfile
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from lifelines import CoxPHFitter
from pycox.evaluation import EvalSurv
from pycox.models import DeepHit
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv

from utils import C, DERIVED, TAB, save_figure, set_style, df_to_md

warnings.filterwarnings('ignore')

set_style()
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {DEVICE}')

d = pd.read_pickle(DERIVED / 'analysis_dataset.pkl')

X = pd.get_dummies(d[['log_mw', 'q_year', 'log_backlog', 'dc_market', 'tech', 'region']],
                   columns=['tech', 'region']).astype('float32')
FEATS = list(X.columns)
tr_mask = (d['q_year'] <= 2017).values
te_mask = ~tr_mask
scale_cols = ['log_mw', 'q_year', 'log_backlog']
mu, sd = X.loc[tr_mask, scale_cols].mean(), X.loc[tr_mask, scale_cols].std()
X[scale_cols] = (X[scale_cols] - mu) / sd

durations = d['T_years'].values.astype('float32')
events = d['event'].values.astype('int64')

NUM_DUR = 40
class LabTransform(LabTransDiscreteTime):
    def transform(self, durations, events):
        durations, is_event = super().transform(durations, events > 0)
        events[is_event == 0] = 0
        return durations, events.astype('int64')

labtrans = LabTransform(NUM_DUR)
y_tr = labtrans.fit_transform(durations[tr_mask], events[tr_mask].copy())
y_te_dur, y_te_ev = labtrans.transform(durations[te_mask], events[te_mask].copy())

class CauseSpecificNet(torch.nn.Module):
    """Shared trunk with one subnetwork per competing risk (DeepHit architecture)."""
    def __init__(self, in_features, num_risks, num_durations,
                 shared=(128, 128), specific=(64, 64), dropout=0.2):
        super().__init__()
        def mlp(sizes_in, sizes):
            layers, prev = [], sizes_in
            for h in sizes:
                layers += [torch.nn.Linear(prev, h), torch.nn.ReLU(),
                           torch.nn.BatchNorm1d(h), torch.nn.Dropout(dropout)]
                prev = h
            return torch.nn.Sequential(*layers), prev
        self.shared, w = mlp(in_features, shared)
        self.risk_nets = torch.nn.ModuleList()
        for _ in range(num_risks):
            net, w2 = mlp(w, specific)
            self.risk_nets.append(torch.nn.Sequential(net, torch.nn.Linear(w2, num_durations)))
        self.num_risks, self.num_durations = num_risks, num_durations

    def forward(self, x):
        z = self.shared(x)
        out = [net(z) for net in self.risk_nets]
        return torch.stack(out, dim=1)              # (batch, risks, durations)

net = CauseSpecificNet(len(FEATS), num_risks=2, num_durations=NUM_DUR)
model = DeepHit(net, tt.optim.Adam(0.001), alpha=0.2, sigma=0.1,
                duration_index=labtrans.cuts, device=DEVICE)

Xtr = X.values[tr_mask]; Xte = X.values[te_mask]
val_idx = np.random.RandomState(SEED).rand(Xtr.shape[0]) < 0.15
val = (Xtr[val_idx], (y_tr[0][val_idx], y_tr[1][val_idx]))
trn = (Xtr[~val_idx], (y_tr[0][~val_idx], y_tr[1][~val_idx]))

print('training DeepHit ...')
log = model.fit(trn[0], trn[1], batch_size=256, epochs=200, verbose=False,
                val_data=val, callbacks=[tt.callbacks.EarlyStopping(
                    patience=15,
                    file_path=str(Path(tempfile.gettempdir()) / f'ckpt_{os.getpid()}.pt'))])
print(f'  stopped after {len(log.to_pandas())} epochs')

cif = model.predict_cif(Xte)                        # (risks, durations, n_test)
surv_c1 = pd.DataFrame(1 - cif[0], index=labtrans.cuts)
surv_c2 = pd.DataFrame(1 - cif[1], index=labtrans.cuts)
ctd_c1 = EvalSurv(surv_c1, durations[te_mask], (events[te_mask] == 1).astype(int),
                  censor_surv='km').concordance_td('antolini')
ctd_c2 = EvalSurv(surv_c2, durations[te_mask], (events[te_mask] == 2).astype(int),
                  censor_surv='km').concordance_td('antolini')
print(f'DeepHit temporal-split Ctd: completion {ctd_c1:.3f}, withdrawal {ctd_c2:.3f}')

dtr = d[tr_mask].copy(); dte = d[te_mask].copy()
cox_X = X.copy(); cox_X['T'] = durations; cox_X['E'] = (events == 1).astype(int)
cph = CoxPHFitter(penalizer=0.01).fit(cox_X[tr_mask], duration_col='T', event_col='E')
from lifelines.utils import concordance_index
cox_c = concordance_index(durations[te_mask], -cph.predict_partial_hazard(cox_X[te_mask]),
                          (events[te_mask] == 1).astype(int))
gb = GradientBoostingSurvivalAnalysis(n_estimators=300, learning_rate=0.05,
                                      max_depth=3, subsample=0.8, random_state=SEED)
gb.fit(X.values[tr_mask], Surv.from_arrays((events[tr_mask] == 1), durations[tr_mask]))
gb_c = gb.score(X.values[te_mask], Surv.from_arrays((events[te_mask] == 1), durations[te_mask]))

comp = pd.DataFrame([
    {'Model': 'Cause-specific Cox (A04 specification)', 'Temporal C-index': cox_c,
     'Handles withdrawal jointly': 'no', 'Assumption-free': 'no'},
    {'Model': 'Random survival forest (A05/A09)', 'Temporal C-index': 0.733,
     'Handles withdrawal jointly': 'no', 'Assumption-free': 'yes'},
    {'Model': 'Gradient-boosted survival', 'Temporal C-index': gb_c,
     'Handles withdrawal jointly': 'no', 'Assumption-free': 'yes'},
    {'Model': 'DeepHit (this analysis), completion', 'Temporal C-index': ctd_c1,
     'Handles withdrawal jointly': 'yes', 'Assumption-free': 'yes'},
    {'Model': 'DeepHit (this analysis), withdrawal', 'Temporal C-index': ctd_c2,
     'Handles withdrawal jointly': 'yes', 'Assumption-free': 'yes'},
]).set_index('Model')
df_to_md(comp.round(3), TAB / 'table_model_comparison.md')
print(comp.round(3).to_string())

print('permutation importance ...')
rng = np.random.RandomState(0)
groups = {'Project size': ['log_mw'], 'Entry year': ['q_year'],
          'Queue backlog': ['log_backlog'], 'DC market county': ['dc_market'],
          'Technology': [c for c in FEATS if c.startswith('tech_')],
          'Region / ISO': [c for c in FEATS if c.startswith('region_')]}
imp_rows = []
sub = np.random.RandomState(1).choice(np.where(te_mask)[0], size=min(4000, te_mask.sum()),
                                      replace=False)
Xs = X.values[sub]; ds_ = durations[sub]; es_ = events[sub]
base_cif = model.predict_cif(Xs)
base_ctd = EvalSurv(pd.DataFrame(1 - base_cif[0], index=labtrans.cuts), ds_,
                    (es_ == 1).astype(int), censor_surv='km').concordance_td('antolini')
for gname, cols in groups.items():
    idx = [FEATS.index(c) for c in cols]
    drops = []
    for _ in range(3):
        Xp = Xs.copy()
        Xp[:, idx] = Xp[rng.permutation(len(Xp))][:, idx]
        pc = model.predict_cif(Xp)
        c = EvalSurv(pd.DataFrame(1 - pc[0], index=labtrans.cuts), ds_,
                     (es_ == 1).astype(int), censor_surv='km').concordance_td('antolini')
        drops.append(base_ctd - c)
    imp_rows.append({'Predictor': gname, 'Importance': np.mean(drops), 'SD': np.std(drops)})
    print(f'  {gname}: {np.mean(drops):.4f}')
imp = pd.DataFrame(imp_rows).sort_values('Importance', ascending=False)
df_to_md(imp.round(4), TAB / 'table_deephit_importance.md', index=False)

print('refitting final model on all data ...')
y_all = labtrans.transform(durations, events.copy())
net2 = CauseSpecificNet(len(FEATS), num_risks=2, num_durations=NUM_DUR)
model_f = DeepHit(net2, tt.optim.Adam(0.001), alpha=0.2, sigma=0.1,
                  duration_index=labtrans.cuts, device=DEVICE)
val2_idx = np.random.RandomState(7).rand(len(X)) < 0.15
model_f.fit(X.values[~val2_idx], (y_all[0][~val2_idx], y_all[1][~val2_idx]),
            batch_size=256, epochs=200, verbose=False,
            val_data=(X.values[val2_idx], (y_all[0][val2_idx], y_all[1][val2_idx])),
            callbacks=[tt.callbacks.EarlyStopping(
                    patience=15,
                    file_path=str(Path(tempfile.gettempdir()) / f'ckpt_{os.getpid()}.pt'))])

def make_row(mw, tech, region, year):
    r = {c: 0.0 for c in FEATS}
    r['log_mw'] = (np.log(mw) - mu['log_mw']) / sd['log_mw']
    r['q_year'] = (year - mu['q_year']) / sd['q_year']
    med_bl = d['log_backlog'].median()
    r['log_backlog'] = (med_bl - mu['log_backlog']) / sd['log_backlog']
    r['dc_market'] = 0.0
    r[f'tech_{tech}'] = 1.0
    r[f'region_{region}'] = 1.0
    return [r[c] for c in FEATS]

ARCH = [
    ('300 MW gas, ERCOT',       300, 'Gas',    'ERCOT', 2023),
    ('300 MW solar, ERCOT',     300, 'Solar',  'ERCOT', 2023),
    ('300 MW gas, PJM',         300, 'Gas',    'PJM',   2023),
    ('200 MW solar, PJM',       200, 'Solar',  'PJM',   2023),
    ('300 MW hybrid, CAISO',    300, 'Hybrid', 'CAISO', 2023),
    ('400 MW solar, ISO-NE',    400, 'Solar',  'ISO-NE', 2023),
]
Xa = np.array([make_row(*a[1:]) for a in ARCH], dtype='float32')
cif_a = model_f.predict_cif(Xa)                        # (2, durations, n_arch)
arch_tbl = pd.DataFrame({
    'P(COD) 3y': [np.interp(3, labtrans.cuts, cif_a[0][:, i]) for i in range(len(ARCH))],
    'P(COD) 5y': [np.interp(5, labtrans.cuts, cif_a[0][:, i]) for i in range(len(ARCH))],
    'P(COD) 7y': [np.interp(7, labtrans.cuts, cif_a[0][:, i]) for i in range(len(ARCH))],
    'P(withdrawn) 5y': [np.interp(5, labtrans.cuts, cif_a[1][:, i]) for i in range(len(ARCH))],
}, index=[a[0] for a in ARCH])
df_to_md(arch_tbl.round(3), TAB / 'table_archetype_predictions.md')
print(arch_tbl.round(3).to_string())

fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), gridspec_kw={'width_ratios': [1.0, 1.35, 1.0]})

ax = axes[0]
rows = [('Cox', cox_c, C['silver']), ('RSF', 0.733, C['silver']),
        ('Boosting', gb_c, C['silver']), ('DeepHit', ctd_c1, C['blue'])]
ax.barh([r[0] for r in rows][::-1], [r[1] for r in rows][::-1],
        color=[r[2] for r in rows][::-1], height=0.55)
for i, (nme, v, _) in enumerate(rows[::-1]):
    ax.text(v + 0.006, i, f'{v:.3f}', va='center', fontsize=7.2, color=C['mut'])
ax.set_xlim(0.5, 0.85)
ax.set_xlabel('Temporal-split concordance\n(completion)')
ax.set_title('(a) Out-of-time discrimination', loc='left')

ax = axes[1]
acol = [C['blue'], C['green'], C['purple'], C['red'], C['graph'], C['silver']]
astyle = ['-', '-', '-', '-', '--', '--']
order = np.argsort(-cif_a[0][-1, :])
for rank, i in enumerate(order):
    name = ARCH[i][0]
    ax.plot(labtrans.cuts, cif_a[0][:, i], color=acol[rank], lw=1.9, ls=astyle[rank],
            label=f'{name}')
ax.set_xlim(0, 10)
ax.set_ylim(0, max(0.30, float(cif_a[0].max()) * 1.15))
ax.set_xlabel('Years since interconnection request')
ax.set_ylabel('P(commercial operation)')
ax.set_title('(b) Predicted delivery, 2023 entrants', loc='left')
ax.legend(frameon=False, fontsize=6.2, loc='upper left', handlelength=1.6)
ax.grid(axis='y')
plt.setp(ax, xticks=[0, 2, 4, 6, 8, 10])

ax = axes[2]
imp_s = imp.sort_values('Importance')
ax.barh(imp_s['Predictor'], imp_s['Importance'], color=C['blue'], height=0.55)
ax.errorbar(imp_s['Importance'], range(len(imp_s)), xerr=imp_s['SD'], fmt='none',
            ecolor=C['ink'], elinewidth=0.8, capsize=2)
ax.set_xlabel('Permutation importance\n(loss in Ctd)')
ax.set_title('(c) What the network uses', loc='left')

plt.tight_layout()
save_figure(fig, 'fig06_deep_competing_risks')
plt.close()

json.dump({'device': DEVICE, 'epochs': len(log.to_pandas()),
           'ctd_completion': float(ctd_c1), 'ctd_withdrawal': float(ctd_c2),
           'cox_temporal': float(cox_c), 'gb_temporal': float(gb_c),
           'n_train': int(tr_mask.sum()), 'n_test': int(te_mask.sum())},
          open(TAB / 'deephit_fit.json', 'w'), indent=1)
print('\nfigure written.')
