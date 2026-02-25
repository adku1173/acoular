# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
# %%
"""
Airfoil in open jet -- Covariance matrix fitting (CMF).
=======================================================

Demonstrates the inverse CMF method with different solvers.
Uses measured data in file example_data.h5, calibration in file example_calib.xml,
microphone geometry in array_56.xml (part of Acoular).
"""

from pathlib import Path

import acoular as ac
ac.config.global_caching = 'none'  # for testing purposes, to ensure that all calculations are performed
from acoular.tools.helpers import get_data_file
import sys 
sys.path.append('/home/kujawski/Documents/Code/SBL/SBL_MF_Python')  # noqa: E402
from sbl import SBL, Options
import numpy as np

# %%
# The 4 kHz third-octave band is used for the example.

cfreq = 4000
num = 3


# %%
# Obtain necessary data

time_data_file = get_data_file('example_data.h5')
calib_file = get_data_file('example_calib.xml')

# %%
# Setting up the processing chain for :class:`~acoular.fbeamform.BeamformerCMF` methods.
#
# .. hint::
#    A step-by-step explanation for setting up the processing chain is given in the example
#    :doc:`example_airfoil_in_open_jet_steering_vectors`.

ts = ac.MaskedTimeSamples(
    file=time_data_file,
    invalid_channels=[1, 7],
    start=0,
    stop=16000,
)
calib = ac.Calib(source=ts, file=calib_file, invalid_channels=[1, 7])
mics = ac.MicGeom(file=Path(ac.__file__).parent / 'xml' / 'array_56.xml', invalid_channels=[1, 7])
grid = ac.RectGrid(x_min=-0.6, x_max=-0.0, y_min=-0.3, y_max=0.3, z=-0.68, increment=0.05)
env = ac.Environment(c=346.04)
st = ac.SteeringVector(grid=grid, mics=mics, env=env)
f = ac.PowerSpectra(source=calib, window='Hanning', overlap='50%', block_size=128)
b = ac.BeamformerDamas(freq_data=f, steer=st)

# sbl
fft = ac.RFFT(source=calib, block_size=128, window='Rectangular')
mf = ac.MaskedFreqOut(source=fft, freqs=[cfreq])
sbl = ac.BeamformerSBL(source=mf, steer=st, num_snapshots=-1)


# %%
# Plot result maps for CMF with different solvers from `SciPy <https://scipy.org/>`_ and
# `scikit-learn <https://scikit-learn.org/stable/>`_, including:
#
# * LassoLars
# * LassoLarsBIC
# * OMPCV
# * NNLS
# * fmin_l_bfgs_b

import matplotlib.pyplot as plt


methods = ['FISTA', 'NNLS', 'GaussSeidel', 'SBL1']
methods = ['NNLS']

s_metrics = {}
r_metrics = {}
s1_metrics = {}

plt.figure(1, (10, 4))
i1 = 1
for method in methods:
    print(f"Processing {method}...")
    if method in ['SBL', 'M-SBL','SBL1']:
        # opt = Options(10 ** (-4), 10 ** (-4), 5000, 1, 1, 2, 32, 1)
        # spectra = next(mf.result(num=mf.num_samples))
        # spectra = spectra.reshape(-1, mf.num_freqs, mf.num_channels)
        # spectra = spectra[:, 0, :].T[..., np.newaxis]
        # sm = SBL(A=st.transfer(mf.freqs[0]).T[..., np.newaxis], Y=spectra, options=opt)
        # map = sm[0].reshape(grid.shape)

        sbl.solver.method = method
        sbl.solver.options['niter'] = 5000
        sbl.solver.options['nsources'] = 25
        sm = next(sbl.result(num=1))
        sm.reshape(-1, sbl.num_freqs, st.grid.size)
        map = sm[0].reshape(grid.shape)   

    elif method in ['FISTA','ISTA']:
        b.solver = ac.ISTACV(method=method, num_grid=10, cv=5, seed=42, warm_start=True,
                             options={'niter': 5000}, 
                             cv_options={'niter': 5000}
                             )
        map = b.synthetic(cfreq, 0)
    elif method == 'NNLS':
        b.solver = ac.NNLSProjLandweber(options={'niter': 10000})
        map = b.synthetic(cfreq, 0)
    else:
        b.solver = ac.GaussSeidelSolver(options={'niter': 5000})
        map = b.synthetic(cfreq, 0)
    
    plt.subplot(1, len(methods), i1)
    i1 += 1
    mx = ac.L_p(map.max())
    plt.imshow(ac.L_p(map.T), vmax=mx, vmin=mx - 15, origin='lower', interpolation='nearest', extent=grid.extent)
    plt.colorbar(shrink=0.5)
    plt.title(method)

    if method in ['FISTA','ISTA', 'NNLS', 'GaussSeidel']:
        ind = list(b.solver.output.keys())[0]
        s_metrics[method] = b.solver.output[ind].get('s_cost')
        r_metrics[method] = b.solver.output[ind].get('r_cost')
        s1_metrics[method] = b.solver.output[ind].get('s1_cost')
    else: 
        ind = list(sbl.solver.output.keys())[0]
        s_metrics[method] = sbl.solver.output[ind].get('s_cost')
        r_metrics[method] = sbl.solver.output[ind].get('r_cost')
        s1_metrics[method] = sbl.solver.output[ind].get('s1_cost')

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharex=False)
for method in methods:
    if s_metrics[method] is not None:
        axes[0].plot(s_metrics[method], label=method)
    if r_metrics[method] is not None:
        axes[1].plot(r_metrics[method], label=method)
    if s1_metrics[method] is not None:
        axes[2].plot(s1_metrics[method], label=method)

axes[0].set_title("eps_s")
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("eps_s")
axes[0].set_yscale('log')
axes[0].legend()

axes[1].set_title("eps_r")
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("eps_r")
axes[1].set_yscale('log')
axes[1].legend()

axes[2].set_title("eps_s1")
axes[2].set_xlabel("Iteration")
axes[2].set_ylabel("eps_s1")
axes[2].set_yscale('log')
axes[2].legend()

fig.tight_layout()
plt.show()


# %%
