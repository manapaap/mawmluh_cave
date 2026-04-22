# -*- coding: utf-8 -*-
"""
Gaussian Process regression to estimate lags between records
Part 2: Load models, generate ensembles, compute lag distributions
"""

import numpy as np
import pandas as pd
from os import chdir, makedirs
from os.path import join, exists
from itertools import combinations
import torch
import gpytorch
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


chdir('C:/Users/aakas/Documents/Oster_lab/programs')
from shared_funcs import combine_mawmluh, load_data, plot_psd, d_o_dates
chdir('C:/Users/aakas/Documents/Oster_lab/')


# ── Configuration ─────────────────────────────────────────────────────────────

RECORDS_TO_FIT = {
    # 'maw_3_clean': 'd18O',
    'maw_comb':    'd18O',
    'ngrip':       'd18O',
    'wais':        'd18O',
    'arabia':      'refl',
    'hulu':        'd18O',
    'sofular':     'd13C'
}

RECORD_LABELS = {
    # 'maw_3_clean': 'MAW_raw',
    'maw_comb':    'MAW',
    'ngrip':       'NGRIP',
    'wais':        'WAIS',
    'arabia':      'OMZ',
    'hulu':        'Hulu',
    'sofular':     'SOF'
}

MODEL_DIR    = 'gp_models'
RESULTS_DIR  = 'gp_results'
MIN_ICE_DATE = 27000
MAX_ICE_DATE = 46000
N_GRID       = 950
N_SAMPLES    = 1000
MAX_LAG_YRS  = 500
DROP_BOUNDARY_LAGS = True 

# ── GPU configuration ─────────────────────────────────────────────────────────
USE_GPU = True   # ← flip to False to force CPU

if USE_GPU and torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    # Check available VRAM and set threshold accordingly
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    # Kernel matrix is n² float32 = n² * 4 bytes
    # Leave headroom for gradients (~3-4x the kernel matrix)
    # So max n ≈ sqrt(vram_bytes / 4 / 4)
    torch.cuda.set_per_process_memory_fraction(0.85)  # use max 85% = ~10.2 GB
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU enabled: {torch.cuda.get_device_name(0)}")
    print(f"Physical VRAM: {vram_gb:.1f} GB")
    safe_n = int((vram_gb * 1e9 / 4 / 4) ** 0.5)
    EXACT_GP_MAX_POINTS = min(safe_n, 10000)
    print(f"GPU enabled: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {vram_gb:.1f} GB  →  exact GP threshold: {EXACT_GP_MAX_POINTS} points")
else:
    DEVICE = torch.device('cpu')
    EXACT_GP_MAX_POINTS = 3000
    if USE_GPU and not torch.cuda.is_available():
        print("Warning: USE_GPU=True but no CUDA device found — falling back to CPU")
    else:
        print("Running on CPU")


# ── Model classes ─────────────────────────────────────────────────────────────

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SparseGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ── GP helpers ────────────────────────────────────────────────────────────────

def fit_gp(train_x, train_y, n_iter=500, lr=0.5,
           ls_prior_mean=None, ls_prior_std=0.5):
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE)
    model = ExactGPModel(train_x, train_y, likelihood).to(DEVICE)

    if ls_prior_mean is None:
        median_spacing = torch.diff(train_x.sort().values).median().item()
        ls_prior_mean = 20 * median_spacing

    ls_prior = gpytorch.priors.LogNormalPrior(
        loc=np.log(ls_prior_mean), scale=ls_prior_std
    )
    model.covar_module.base_kernel.register_prior(
        'lengthscale_prior', ls_prior, 'lengthscale'
    )

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(n_iter):
        optimizer.zero_grad()
        with gpytorch.settings.max_cg_iterations(100):
            loss = -mll(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            ls    = model.covar_module.base_kernel.lengthscale.item()
            noise = likelihood.noise.item()
            print(f"  Iter {i+1}/{n_iter}  Loss:{loss.item():.4f}  "
                  f"LS:{ls:.4f}  Noise:{noise:.4f}")
    return model, likelihood


def fit_sparse_gp(train_x, train_y, n_inducing=500, n_iter=500, lr=0.05,
                  ls_prior_mean=None, ls_prior_std=0.5):
    indices = torch.linspace(0, len(train_x) - 1, n_inducing).long()
    inducing_points = train_x[indices].unsqueeze(-1).to(DEVICE)
    model      = SparseGPModel(inducing_points).to(DEVICE)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE)

    if ls_prior_mean is None:
        median_spacing = torch.diff(train_x.sort().values).median().item()
        ls_prior_mean = 20 * median_spacing

    ls_prior = gpytorch.priors.LogNormalPrior(
        loc=np.log(ls_prior_mean), scale=ls_prior_std
    )
    model.covar_module.base_kernel.register_prior(
        'lengthscale_prior', ls_prior, 'lengthscale'
    )

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(likelihood.parameters()), lr=lr
    )
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(train_y))
    for i in range(n_iter):
        optimizer.zero_grad()
        loss = -mll(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            ls    = model.covar_module.base_kernel.lengthscale.item()
            noise = likelihood.noise.item()
            print(f"  Iter {i+1}/{n_iter}  Loss:{loss.item():.4f}  "
                  f"LS:{ls:.4f}  Noise:{noise:.4f}")
    return model, likelihood


def save_gp(path, model, likelihood, t_mean, t_std, y_mean, y_std, **meta):
    # Move to CPU before saving so checkpoints are device-agnostic
    torch.save({
        'model_state_dict':      {k: v.cpu() for k, v in model.state_dict().items()},
        'likelihood_state_dict': {k: v.cpu() for k, v in likelihood.state_dict().items()},
        't_mean': t_mean, 't_std': t_std,
        'y_mean': y_mean, 'y_std': y_std,
        **meta
    }, path)


def load_gp(path, train_x, train_y):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE)

    if checkpoint.get('is_sparse', False):
        n_inducing = checkpoint.get('n_inducing', 500)
        indices = torch.linspace(0, len(train_x) - 1, n_inducing).long()
        inducing_points = train_x[indices].unsqueeze(-1).to(DEVICE)
        model = SparseGPModel(inducing_points).to(DEVICE)
    else:
        model = ExactGPModel(train_x, train_y, likelihood).to(DEVICE)

    # Re-register prior so state_dict keys match — values are overwritten on load
    dummy_prior = gpytorch.priors.LogNormalPrior(loc=0.0, scale=1.0)
    model.covar_module.base_kernel.register_prior(
        'lengthscale_prior', dummy_prior, 'lengthscale'
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
    model.eval()
    likelihood.eval()
    return model, likelihood, checkpoint


def fit_and_save_record(key, record, y_col, model_dir, max_ice_date,
                        n_iter=500, lr=0.5):
    print(f"\n{'='*50}\nFitting GP for: {key}  (N={len(record)})  device={DEVICE}")
    t = record['age_BP'].values.astype(np.float64)
    y = record[y_col].values.astype(np.float64)
    t_mean, t_std = t.mean(), t.std()
    y_mean, y_std = y.mean(), y.std()

    # Move training tensors to device
    train_x = torch.tensor((t - t_mean) / t_std, dtype=torch.float32).to(DEVICE)
    train_y = torch.tensor((y - y_mean) / y_std, dtype=torch.float32).to(DEVICE)

    if len(train_x) > EXACT_GP_MAX_POINTS:
        print(f"  Sparse GP (n_inducing=500)")
        model, likelihood = fit_sparse_gp(train_x, train_y, n_iter=n_iter, lr=0.05)
        is_sparse = True
    else:
        model, likelihood = fit_gp(train_x, train_y, n_iter=n_iter, lr=lr)
        is_sparse = False

    ls_years = model.covar_module.base_kernel.lengthscale.item() * t_std
    noise    = likelihood.noise.item()
    print(f"  LS={ls_years:.1f} yr  Noise={noise:.4f}  sparse={is_sparse}")

    model.eval()
    likelihood.eval()
    t_grid_norm = torch.linspace(
        train_x.min(), train_x.max(), 2000
    ).to(DEVICE)

    with torch.no_grad():
        if is_sparse:
            preds = likelihood(model(t_grid_norm))
        else:
            with gpytorch.settings.fast_pred_var():
                preds = likelihood(model(t_grid_norm))
        mu           = preds.mean.cpu().numpy()
        lower, upper = preds.confidence_region()
        lower        = lower.cpu().numpy()
        upper        = upper.cpu().numpy()

    t_grid       = t_grid_norm.cpu().numpy() * t_std + t_mean
    mu_denorm    = mu    * y_std + y_mean
    lower_denorm = lower * y_std + y_mean
    upper_denorm = upper * y_std + y_mean

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(t, y, s=1, color='orange', alpha=0.5, label='Data')
    ax.plot(t_grid, mu_denorm, color='blue', lw=1.5, label='GP mean')
    ax.fill_between(t_grid, lower_denorm, upper_denorm, alpha=0.2, label='±2σ')
    ax.invert_xaxis()
    ax.set_xlabel('Age (yr BP)')
    ax.set_ylabel(y_col)
    ax.set_title(f'{key}  LS={ls_years:.0f} yr  sparse={is_sparse}  device={DEVICE}')
    ax.legend()
    plt.tight_layout()
    fig.savefig(join(model_dir, f'{key}_gp_fit.png'), dpi=150)
    plt.close(fig)

    model_path = join(model_dir, f'{key}.pth')
    save_gp(model_path, model, likelihood, t_mean, t_std, y_mean, y_std,
            record=key, y_col=y_col,
            age_range=(MIN_ICE_DATE, max_ice_date),
            kernel='Matern_nu2.5', n_iter=n_iter,
            lengthscale_years=ls_years, noise=noise,
            is_sparse=is_sparse, n_inducing=500 if is_sparse else None)
    print(f"  Saved → {model_path}")
    return model, likelihood


# ── Lag calculation ───────────────────────────────────────────────────────────

def find_nearest(array, value):
    return (np.abs(array - value)).argmin()


def lag_finder(y1, y2, period):
    y1 = signal.detrend(np.asarray(y1, dtype=float))
    y2 = signal.detrend(np.asarray(y2, dtype=float))
    n    = y1.size
    corr = signal.correlate(y1, y2, mode='same')
    norm = np.sqrt(
        signal.correlate(y1, y1, mode='same')[n // 2] *
        signal.correlate(y2, y2, mode='same')[n // 2]
    )
    corr /= norm
    lags     = period * signal.correlation_lags(n, n, mode='same')
    low_idx  = find_nearest(lags, -MAX_LAG_YRS)
    high_idx = find_nearest(lags,  MAX_LAG_YRS)

    # Use absolute value — handles inverted records (e.g. NGRIP polarity)
    corr_abs  = np.abs(corr)
    corr_work = corr_abs.copy()
    max_lag   = corr_work.argmax()
    while max_lag < low_idx or max_lag > high_idx:
        corr_work[max_lag] = -1e9
        max_lag = corr_work.argmax()

    return lags[max_lag]


def sample_gp_ensemble(model, likelihood, checkpoint, t_grid_norm, n_samples):
    model.eval()
    likelihood.eval()
    t_grid_norm = t_grid_norm.to(DEVICE)
    with torch.no_grad():
        posterior = model(t_grid_norm)
        samples   = posterior.sample(sample_shape=torch.Size([n_samples]))
    return samples.cpu().numpy()


def summarize_lag(lag_dist, ci=95):
    """
    Returns mean and CI width from empirical lag distribution.
    CI is the full interval width (hi - lo) at the requested confidence level.
    """
    alpha = (100 - ci) / 2
    mean  = np.nanmean(lag_dist)
    lo    = np.nanpercentile(lag_dist, alpha)
    hi    = np.nanpercentile(lag_dist, 100 - alpha)
    ci_width = hi - lo
    return mean, ci_width


def lowpass_filter(y, cutoff_years, sample_spacing_years, order=4):
    """
    Butterworth low-pass filter. Removes variability faster than cutoff_years.
    """
    nyq  = 0.5 / sample_spacing_years   # Nyquist frequency in 1/yr
    freq = 1.0 / cutoff_years           # cutoff frequency in 1/yr
    if freq >= nyq:
        return y  # cutoff above Nyquist — no filtering possible
    b, a = butter(order, freq / nyq, btype='low')
    return filtfilt(b, a, y)


def compute_lag_distribution(samples_a, samples_b, t_grid,
                              filter_cutoff_years=500, filt=False):
    period  = np.abs(np.median(np.diff(t_grid)))
    lags    = []

    for k in range(len(samples_a)):
        if filt:
            sa = lowpass_filter(samples_a[k], filter_cutoff_years, period)
            sb = lowpass_filter(samples_b[k], filter_cutoff_years, period)
        else:
            sa = samples_a[k]
            sb = samples_b[k]
        lags.append(lag_finder(sa, sb, period))

    lags = np.array(lags)

    if DROP_BOUNDARY_LAGS:
        n_total   = len(lags)
        lags      = lags[np.abs(lags) < MAX_LAG_YRS]
        n_dropped = n_total - len(lags)
        if n_dropped > 0:
            print(f"    Dropped {n_dropped}/{n_total} boundary lags "
                  f"({100*n_dropped/n_total:.0f}%)")
        if len(lags) == 0:
            print(f"    Warning: all lags were boundary values — returning NaN")
            return np.array([np.nan])

    return lags


def plot_samples(model, likelihood, t_grid_norm, t_grid_real, n_samples=5,
                 filter_cutoff=500, key=''):
    """
    Plot a few raw and filtered posterior samples to visually inspect
    whether they carry D-O structure before cross-correlation.
    """
    model.eval()
    likelihood.eval()
    t_grid_norm = t_grid_norm.to(DEVICE)

    with torch.no_grad():
        observed_pred = likelihood(model(t_grid_norm))
        samples = observed_pred.sample(
            sample_shape=torch.Size([n_samples])
        ).cpu().numpy()
        mu = observed_pred.mean.cpu().numpy()

    period = np.abs(np.median(np.diff(t_grid_real)))

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    axes[1].set_xlim(MIN_ICE_DATE, MAX_ICE_DATE)

    # Raw samples
    for k in range(n_samples):
        axes[0].plot(t_grid_real, samples[k], alpha=0.4, lw=0.8)
    axes[0].plot(t_grid_real, mu, 'k-', lw=1.5, label='posterior mean')
    axes[0].set_title(f'{key} — raw posterior samples')
    axes[0].legend()

    # Filtered samples
    nyq  = 0.5 / period
    freq = 1.0 / filter_cutoff
    if freq < nyq:
        b, a = butter(4, freq / nyq, btype='low')
        for k in range(n_samples):
            axes[1].plot(t_grid_real,
                         filtfilt(b, a, samples[k]), alpha=0.4, lw=0.8)
        axes[1].plot(t_grid_real, filtfilt(b, a, mu),
                     'k-', lw=1.5, label='filtered mean')
    axes[1].set_title(f'{key} — filtered samples (cutoff={filter_cutoff} yr)')
    axes[1].legend()

    for ax in axes:
        ax.invert_xaxis()
        ax.set_xlabel('Age (yr BP)')

    plt.tight_layout()

# Plotting information

def plot_colored_table(df):
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis('off')

    # Create table data with row labels as first column
    table_data = []
    # Add header row with "Record" in top-left
    header_row = ["Record"] + list(df.columns)
    table_data.append(header_row)
    
    # Add data rows with index names
    for i, row_name in enumerate(df.index):
        table_data.append([row_name] + list(df.iloc[i, :]))

    # Create the table
    table = ax.table(
        cellText=table_data,
        cellLoc='center',
        loc='center'
    )

    # Adjust font size and scaling
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.2)

    # Apply colors
    for i in range(len(table_data)):       # rows
        for j in range(len(table_data[0])): # columns
            cell = table[i, j]
            
            # Color all header cells (first row and first column)
            if i == 0 and j == 0:
                cell.set_facecolor("cornflowerblue")
                cell.set_text_props(weight='bold', color='black')
                continue
            elif i == 0 or j == 0:
                cell.set_facecolor("cornflowerblue")
                cell.set_text_props(weight='bold', color='black')
                continue
            
            # Parse and color data cells
            cell_value = table_data[i][j]
            lag, error = parse_lag_error(cell_value)

            if lag is None or error is None:    
                cell.set_facecolor("white")
            elif lag == 0 and error == 0:
                cell.set_facecolor("lightblue")  # diagonal elements
            elif abs(lag) > error:
                cell.set_facecolor("lightcoral")  # green = not significant
            else:
                cell.set_facecolor("palegreen")  # red = significant

    plt.tight_layout()
    plt.show()
    

def parse_lag_error(cell_value):
    """Parse a 'lag ± error' string into floats."""
    try:
        parts = str(cell_value).replace(" ", "").split("±")
        lag = float(parts[0])
        error = float(parts[1])
        return lag, error
    except Exception:
        return None, None
    
    
def plot_diag_table(df):
    df = df[::-1]
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis('off')

    # Create table data with row labels as first column
    table_data = []
    # Add header row with "Record" in top-left
    header_row = ["Record"] + list(df.columns)
    table_data.append(header_row)
    
    # Add data rows with index names
    for i, row_name in enumerate(df.index):
        table_data.append([row_name] + list(df.iloc[i, :len(df) - i]))
    # reformat the table to make square
    max_len = len(table_data) - 1
    # remove last column
    table_data[0].pop()
    table_data[1].pop()
    # remove last row
    table_data.pop()
    for n, row in enumerate(table_data):
        row_len = len(row)
        if row_len < max_len:
            table_data[n] += [np.nan] * (max_len - row_len) 
        for m, entry in enumerate(row):
            if entry == '0 ± 0':
                table_data[n][m] = None

    # Create the table
    table = ax.table(
        cellText=table_data,
        cellLoc='center',
        loc='center'
    )

    # Adjust font size and scaling
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.2)

    # Apply colors
    for i in range(len(table_data)):       # rows
        for j in range(len(table_data[0])): # columns
            cell = table[i, j]
            
            # Color all header cells (first row and first column)
            if i == 0 and j == 0:
                cell.set_facecolor("white")
                cell.set_text_props(weight='bold', color='black')
                cell.set_linewidth(1.5)
                continue
            elif i == 0 or j == 0:
                cell.set_facecolor("white")
                cell.set_text_props(weight='bold', color='black')
                cell.set_linewidth(1.5)
                continue
            
            # Parse and color data cells
            cell_value = table_data[i][j]
            lag, error = parse_lag_error(cell_value)

            if lag is None or error is None:    
                cell.set_facecolor("white")
                cell.set_text_props(color='white')
                cell.set_linewidth(1.5)
            elif abs(lag) > error and lag < 0:
                cell.set_facecolor("violet")  # purple = lags
                cell.set_linewidth(1.5)
            elif abs(lag) > error:
                cell.set_facecolor("lightcoral")  # tomato = leads 
                cell.set_linewidth(1.5)
            else:
                cell.set_facecolor("khaki")  # = no lag
                cell.set_linewidth(1.5)
                #cell.get_text().set_text(str(int(abs(lag))) + ' ± ' +\
                #                         str(int(error)))

    plt.tight_layout()
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global records, gp_models

    makedirs(MODEL_DIR,   exist_ok=True)
    makedirs(RESULTS_DIR, exist_ok=True)

    records = load_data(filter_year='46000')
    records['maw_comb'] = combine_mawmluh(records, cutoff=39500)
    
    for key in records:
        records[key] = records[key].query(
            f'{MIN_ICE_DATE} <= age_BP <= {MAX_ICE_DATE}'
        )

    # ── 1. Train or load models ───────────────────────────────────────────────
    gp_models = {}
    for key, y_col in RECORDS_TO_FIT.items():
        if key not in records:
            print(f"Warning: '{key}' not in records, skipping.")
            continue

        model_path = join(MODEL_DIR, f'{key}.pth')
        record = records[key]
        t = record['age_BP'].values.astype(np.float64)
        y = record[y_col].values.astype(np.float64)

        if exists(model_path):
            print(f"Loading saved model for '{key}'")
            checkpoint = torch.load(model_path, map_location=DEVICE)
            t_mean, t_std = checkpoint['t_mean'], checkpoint['t_std']
            y_mean, y_std = checkpoint['y_mean'], checkpoint['y_std']
            train_x = torch.tensor(
                (t - t_mean) / t_std, dtype=torch.float32
            ).to(DEVICE)
            train_y = torch.tensor(
                (y - y_mean) / y_std, dtype=torch.float32
            ).to(DEVICE)
            model, likelihood, checkpoint = load_gp(model_path, train_x, train_y)
        else:
            lr = 0.05 if key == 'ngrip' else 0.5
            model, likelihood = fit_and_save_record(
                key, record, y_col, MODEL_DIR, MAX_ICE_DATE, n_iter=500, lr=lr
            )
            checkpoint = torch.load(model_path, map_location=DEVICE)
            t_mean, t_std = checkpoint['t_mean'], checkpoint['t_std']
            y_mean, y_std = checkpoint['y_mean'], checkpoint['y_std']
            train_x = torch.tensor(
                (t - t_mean) / t_std, dtype=torch.float32
            ).to(DEVICE)
            train_y = torch.tensor(
                (y - y_mean) / y_std, dtype=torch.float32
            ).to(DEVICE)

        gp_models[key] = {
            'model':      model,
            'likelihood': likelihood,
            'checkpoint': checkpoint,
            't_mean': t_mean, 't_std': t_std,
            'y_mean': y_mean, 'y_std': y_std,
            'train_x': train_x,
        }
    # -- 1.5: Plot the ensembles to see if they're reasonable --
    if False:
        for key, info in gp_models.items():
            t_grid_real = np.linspace(
                info['t_mean'] + info['train_x'].min().item() * info['t_std'],
                info['t_mean'] + info['train_x'].max().item() * info['t_std'],
                N_GRID
            )
            t_grid_norm = torch.tensor(
                (t_grid_real - info['t_mean']) / info['t_std'],
                dtype=torch.float32
            )
            plot_samples(info['model'], info['likelihood'],
                         t_grid_norm, t_grid_real,
                         n_samples=5, filter_cutoff=500, key=key)


    # ── 2 & 3. Ensemble sampling + lag distributions ──────────────────────────
    keys   = list(gp_models.keys())
    labels = [RECORD_LABELS.get(k, k) for k in keys]
    n      = len(keys)

    mean_arr = np.full((n, n), np.nan)
    ci_arr   = np.full((n, n), np.nan)
    np.fill_diagonal(mean_arr, 0.0)
    np.fill_diagonal(ci_arr,   0.0)

    for i, j in combinations(range(n), 2):
        key_a, key_b   = keys[i], keys[j]
        info_a, info_b = gp_models[key_a], gp_models[key_b]

        print(f"\nComputing lag: {labels[i]} vs {labels[j]}")

        t_min_norm_a = info_a['train_x'].min().item()
        t_max_norm_a = info_a['train_x'].max().item()
        t_min_norm_b = info_b['train_x'].min().item()
        t_max_norm_b = info_b['train_x'].max().item()

        t_min_real = max(info_a['t_mean'] + t_min_norm_a * info_a['t_std'],
                         info_b['t_mean'] + t_min_norm_b * info_b['t_std'])
        t_max_real = min(info_a['t_mean'] + t_max_norm_a * info_a['t_std'],
                         info_b['t_mean'] + t_max_norm_b * info_b['t_std'])

        if t_max_real <= t_min_real:
            print(f"  No temporal overlap — skipping.")
            continue

        t_grid_real = np.linspace(t_min_real, t_max_real, N_GRID)

        t_grid_a = torch.tensor(
            (t_grid_real - info_a['t_mean']) / info_a['t_std'],
            dtype=torch.float32
        )
        t_grid_b = torch.tensor(
            (t_grid_real - info_b['t_mean']) / info_b['t_std'],
            dtype=torch.float32
        )

        samples_a = sample_gp_ensemble(
            info_a['model'], info_a['likelihood'],
            info_a['checkpoint'], t_grid_a, N_SAMPLES
        )
        samples_b = sample_gp_ensemble(
            info_b['model'], info_b['likelihood'],
            info_b['checkpoint'], t_grid_b, N_SAMPLES
        )

        lag_dist = compute_lag_distribution(samples_a, samples_b, t_grid_real)
        mean, ci_width = summarize_lag(lag_dist)
        print(f"  Mean lag: {np.nanmean(lag_dist):.1f} yr  "
              f"95% CI width: {ci_width:.1f} yr")

        mean_arr[i, j] =  -mean
        ci_arr[i, j]   =  ci_width
        mean_arr[j, i] = mean
        ci_arr[j, i]   =  ci_width

    # ── 4. Compile into dataframes ────────────────────────────────────────────
    df_mean = pd.DataFrame(mean_arr, index=labels, columns=labels)
    df_ci   = pd.DataFrame(ci_arr,   index=labels, columns=labels)

    df_combined = df_mean.copy().astype(object)
    for i in range(n):
        for j in range(n):
            m  = mean_arr[i, j]
            ci = ci_arr[i, j]
            if i == j:
                df_combined.iloc[i, j] = '0'
            elif np.isnan(m):
                df_combined.iloc[i, j] = 'N/A'
            else:
                df_combined.iloc[i, j] = f'{m:.0f} ± {ci:.0f}'

    print("\n\nLag table (positive = row leads column):")
    print(df_combined.to_string())

    df_mean.to_csv(join(RESULTS_DIR, 'lag_means.csv'))
    df_ci.to_csv(join(RESULTS_DIR,   'lag_ci.csv'))
    df_combined.to_csv(join(RESULTS_DIR, 'lag_table.csv'))
    print(f"\nResults saved to '{RESULTS_DIR}/'")


    df_combined.rename_axis('Record')
    plot_colored_table(df_combined)
    plot_diag_table(df_combined)
    return df_mean, df_ci, df_combined

    # Plot final results!!


if __name__ == "__main__":
    df_mean, df_std, df_combined = main()
    
