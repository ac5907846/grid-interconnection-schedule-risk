"""Hyperparameter search, seed ensemble, and planning grid for the deep model."""
import argparse
import json
import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from pycox.evaluation import EvalSurv
from pycox.models import DeepHit
from pycox.preprocessing.label_transforms import LabTransDiscreteTime

from utils import C, DERIVED, TAB, save_figure, set_style

warnings.filterwarnings('ignore')

OUT = TAB.parent / 'ensemble'
OUT.mkdir(exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_DUR = 40

parser = argparse.ArgumentParser()
parser.add_argument('--trials', type=int, default=300)
parser.add_argument('--ensemble', type=int, default=10)
parser.add_argument('--bootstrap', type=int, default=200)
args = parser.parse_args()
print(f'device: {DEVICE} | trials: {args.trials} | ensemble: {args.ensemble}')
if DEVICE == 'cpu':
    print('WARNING: no CUDA detected; this will be slow. It still runs correctly.')

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


class LabTransform(LabTransDiscreteTime):
    def transform(self, durations, events):
        durations, is_event = super().transform(durations, events > 0)
        events[is_event == 0] = 0
        return durations, events.astype('int64')


labtrans = LabTransform(NUM_DUR)
y_tr_all = labtrans.fit_transform(durations[tr_mask], events[tr_mask].copy())
Xtr_all, Xte = X.values[tr_mask], X.values[te_mask]
rng = np.random.RandomState(42)
val_idx = rng.rand(Xtr_all.shape[0]) < 0.15
TRN = (Xtr_all[~val_idx], (y_tr_all[0][~val_idx], y_tr_all[1][~val_idx]))
VAL = (Xtr_all[val_idx], (y_tr_all[0][val_idx], y_tr_all[1][val_idx]))


class CauseSpecificNet(torch.nn.Module):
    def __init__(self, in_features, shared, specific, dropout,
                 num_risks=2, num_durations=NUM_DUR):
        super().__init__()
        def mlp(n_in, sizes):
            layers, prev = [], n_in
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

    def forward(self, x):
        z = self.shared(x)
        return torch.stack([net(z) for net in self.risk_nets], dim=1)


def fit_one(params, seed=0, train=TRN, val=VAL, epochs=300, patience=20):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = CauseSpecificNet(len(FEATS), params['shared'], params['specific'],
                           params['dropout'])
    model = DeepHit(net, tt.optim.Adam(params['lr']), alpha=params['alpha'],
                    sigma=params['sigma'], duration_index=labtrans.cuts, device=DEVICE)
    model.fit(train[0], train[1], batch_size=params['batch'], epochs=epochs,
              verbose=False, val_data=val,
              callbacks=[tt.callbacks.EarlyStopping(
                  patience=patience,
                  file_path=str(Path(tempfile.gettempdir()) /
                                f'ckpt_{os.getpid()}_{seed}.pt'))])
    return model


def ctd_completion(model, Xs, dur, ev):
    cif = model.predict_cif(Xs)
    surv = pd.DataFrame(1 - cif[0], index=labtrans.cuts)
    return EvalSurv(surv, dur, (ev == 1).astype(int),
                    censor_surv='km').concordance_td('antolini')


import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
storage = f'sqlite:///{OUT / "optuna_study.db"}'
study = optuna.create_study(direction='maximize', study_name='deephit_queue',
                            storage=storage, load_if_exists=True,
                            sampler=optuna.samplers.TPESampler(seed=42))

WIDTHS = {'narrow': (64, 64), 'medium': (128, 128), 'wide': (256, 256),
          'deep': (128, 128, 128), 'vwide': (512, 256)}
SPEC = {'small': (32, 32), 'medium': (64, 64), 'large': (128, 64)}


def objective(trial):
    params = {
        'shared': WIDTHS[trial.suggest_categorical('shared', list(WIDTHS))],
        'specific': SPEC[trial.suggest_categorical('specific', list(SPEC))],
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'alpha': trial.suggest_float('alpha', 0.05, 0.95),
        'sigma': trial.suggest_float('sigma', 0.05, 1.0, log=True),
        'batch': trial.suggest_categorical('batch', [128, 256, 512, 1024]),
    }
    model = fit_one(params, seed=trial.number)
    return ctd_completion(model, VAL[0], durations[tr_mask][val_idx],
                          events[tr_mask][val_idx])


done = len([t for t in study.trials if t.state.name == 'COMPLETE'])
if done < args.trials:
    print(f'searching: {done} trials done, running to {args.trials} ...')
    study.optimize(objective, n_trials=args.trials - done, show_progress_bar=True)
best_raw = study.best_params
BEST = {'shared': WIDTHS[best_raw['shared']], 'specific': SPEC[best_raw['specific']],
        'dropout': best_raw['dropout'], 'lr': best_raw['lr'],
        'alpha': best_raw['alpha'], 'sigma': best_raw['sigma'],
        'batch': best_raw['batch']}
json.dump({'raw': best_raw, 'value': study.best_value}, open(OUT / 'best_params.json', 'w'),
          indent=1)
study.trials_dataframe().to_csv(OUT / 'trials.csv', index=False)
print(f'best validation Ctd: {study.best_value:.4f} | {best_raw}')

print(f'fitting seed ensemble of {args.ensemble} on the temporal training set ...')
models = [fit_one(BEST, seed=s) for s in range(args.ensemble)]
ctds = np.array([ctd_completion(m, Xte, durations[te_mask], events[te_mask])
                 for m in models])
print(f'ensemble temporal Ctd (completion): mean {ctds.mean():.4f}, '
      f'range {ctds.min():.4f} to {ctds.max():.4f}')

cifs_te = np.stack([m.predict_cif(Xte) for m in models])       # (S, 2, K, n)
mean_surv1 = pd.DataFrame(1 - cifs_te[:, 0].mean(0), index=labtrans.cuts)
boot = []
n_te = Xte.shape[0]
brng = np.random.RandomState(0)
for b in range(args.bootstrap):
    idx = brng.randint(0, n_te, n_te)
    ev = EvalSurv(mean_surv1.iloc[:, idx], durations[te_mask][idx],
                  (events[te_mask][idx] == 1).astype(int), censor_surv='km')
    boot.append(ev.concordance_td('antolini'))
boot = np.array(boot)
json.dump({'ensemble_ctds': ctds.tolist(),
           'ensemble_mean_ctd': float(ctds.mean()),
           'bootstrap_ci90': [float(np.percentile(boot, 5)),
                              float(np.percentile(boot, 95))]},
          open(OUT / 'ensemble_ctd.json', 'w'), indent=1)
print(f'bootstrap 90pct CI on ensemble-mean Ctd: '
      f'[{np.percentile(boot, 5):.4f}, {np.percentile(boot, 95):.4f}]')

print('refitting final ensemble on all entries for the planning table ...')
y_all = labtrans.transform(durations, events.copy())
val2 = np.random.RandomState(7).rand(len(X)) < 0.15
TRN2 = (X.values[~val2], (y_all[0][~val2], y_all[1][~val2]))
VAL2 = (X.values[val2], (y_all[0][val2], y_all[1][val2]))
finals = [fit_one(BEST, seed=100 + s, train=TRN2, val=VAL2)
          for s in range(args.ensemble)]

REGIONS = sorted(d['region'].unique())
TECHS = ['Solar', 'Wind', 'Battery', 'Hybrid', 'Gas']
SIZES = [50, 100, 200, 400, 800]
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
med_bl = d['log_backlog'].median()

rows, batch = [], []
for reg in REGIONS:
    for tech in TECHS:
        for mw in SIZES:
            for yr in YEARS:
                r = {c: 0.0 for c in FEATS}
                r['log_mw'] = (np.log(mw) - mu['log_mw']) / sd['log_mw']
                r['q_year'] = (yr - mu['q_year']) / sd['q_year']
                r['log_backlog'] = (med_bl - mu['log_backlog']) / sd['log_backlog']
                if f'tech_{tech}' in r: r[f'tech_{tech}'] = 1.0
                if f'region_{reg}' in r: r[f'region_{reg}'] = 1.0
                batch.append([r[c] for c in FEATS])
                rows.append((reg, tech, mw, yr))
Xg = np.array(batch, dtype='float32')
cifs = np.stack([m.predict_cif(Xg) for m in finals])           # (S, 2, K, n)
cuts = labtrans.cuts


def at(hz, arr):                                               # arr (S, K, n) -> (S, n)
    return np.stack([np.interp(hz, cuts, arr[s]) for s in range(arr.shape[0])])


tbl = pd.DataFrame(rows, columns=['region', 'tech', 'mw', 'entry_year'])
for hz in (3, 5, 7):
    comp = np.stack([np.stack([np.interp(hz, cuts, cifs[s, 0, :, i])
                               for i in range(len(rows))]) for s in range(len(finals))])
    tbl[f'p_cod_{hz}y_mean'] = comp.mean(0)
    tbl[f'p_cod_{hz}y_lo90'] = np.percentile(comp, 5, axis=0)
    tbl[f'p_cod_{hz}y_hi90'] = np.percentile(comp, 95, axis=0)
wd5 = np.stack([np.stack([np.interp(5, cuts, cifs[s, 1, :, i])
                          for i in range(len(rows))]) for s in range(len(finals))])
tbl['p_withdrawn_5y_mean'] = wd5.mean(0)
tbl.round(4).to_csv(OUT / 'planning_table.csv', index=False)
print(f'planning_table.csv: {len(tbl):,} rows '
      f'({len(REGIONS)} regions x {len(TECHS)} tech x {len(SIZES)} sizes x {len(YEARS)} years)')

set_style()
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

ax = axes[0]
ARCH = [('300 MW gas, ERCOT', 'Gas', 'ERCOT'), ('300 MW gas, PJM', 'Gas', 'PJM'),
        ('200 MW solar, PJM', 'Solar', 'PJM'), ('400 MW solar, ISO-NE', 'Solar', 'ISO-NE')]
cols = [C['blue'], C['green'], C['graph'], C['silver']]
for (name, tech, reg), col in zip(ARCH, cols):
    r = {c: 0.0 for c in FEATS}
    r['log_mw'] = (np.log(300 if 'gas' in name else (200 if 'PJM' in name else 400))
                   - mu['log_mw']) / sd['log_mw']
    r['q_year'] = (2023 - mu['q_year']) / sd['q_year']
    r['log_backlog'] = (med_bl - mu['log_backlog']) / sd['log_backlog']
    r[f'tech_{tech}'] = 1.0
    r[f'region_{reg}'] = 1.0
    xa = np.array([[r[c] for c in FEATS]], dtype='float32')
    cc = np.stack([m.predict_cif(xa)[0][:, 0] for m in finals])
    ax.plot(cuts, cc.mean(0), color=col, lw=1.9, label=name)
    ax.fill_between(cuts, np.percentile(cc, 5, 0), np.percentile(cc, 95, 0),
                    color=col, alpha=0.15, lw=0)
ax.set_xlim(0, 10); ax.set_xlabel('Years since interconnection request')
ax.set_ylabel('P(commercial operation)')
ax.set_title('(a) Ensemble delivery curves, 90% bands', loc='left')
ax.legend(frameon=False, fontsize=6.4, loc='upper left')
ax.grid(axis='y')

ax = axes[1]
ax.hist(boot, bins=30, color=C['silver'], edgecolor='white')
ax.axvline(ctds.mean(), color=C['blue'], lw=1.6)
ax.set_xlabel('Bootstrap temporal Ctd (completion)')
ax.set_title('(b) Uncertainty of out-of-time accuracy', loc='left')

plt.tight_layout()
save_figure(fig, OUT, 'figR_deep_ensemble')
print('done. All outputs in', OUT)
