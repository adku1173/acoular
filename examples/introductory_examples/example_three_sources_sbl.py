# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
# %%
"""
Three sources -- Generate synthetic microphone array data.
==========================================================

Generates a test data set for three sources.

The simulation generates the sound pressure at 64 microphones that are
arrangend in the 'array64' geometry which is part of the package. The sound
pressure signals are sampled at 51200 Hz for a duration of 1 second.
The simulated signals are stored in a HDF5 file named 'three_sources.h5'.

Source location (relative to array center) and levels:

====== =============== ======
Source Location        Level
====== =============== ======
1      (-0.1,-0.1,0.3) 1 Pa
2      (0.15,0,0.3)    0.7 Pa
3      (0,0.1,0.3)     0.5 Pa
====== =============== ======


"""
import sys 
sys.path.append('/home/kujawski/Documents/Code/SBL/SBL_MF_Python')  # noqa: E402
from sbl import SBL, Options
from pathlib import Path

import acoular as ac
import numpy as np
import matplotlib.pyplot as plt

sfreq = 51200
block_size = 512
niter = 100
n_snap = -1
dyn= 20
duration = 5.
num_samples = duration * sfreq
micgeofile = Path(ac.__file__).parent / 'xml' / 'array_64.xml'
h5savefile = Path('three_sources.h5')

m = ac.MicGeom(file=micgeofile)
n1 = ac.WNoiseGenerator(sample_freq=sfreq, num_samples=num_samples, seed=1)
n2 = ac.WNoiseGenerator(sample_freq=sfreq, num_samples=num_samples, seed=2, rms=0.7)
n3 = ac.WNoiseGenerator(sample_freq=sfreq, num_samples=num_samples, seed=3, rms=0.5)
p1 = ac.PointSource(signal=n1, mics=m, loc=(-0.1, -0.1, -0.3))
p2 = ac.PointSource(signal=n2, mics=m, loc=(0.15, 0, -0.3))
p3 = ac.PointSource(signal=n3, mics=m, loc=(0, 0.1, -0.3))
p = ac.Mixer(source=p1, sources=[p2, p3])
fft = ac.RFFT(source=p, block_size=block_size, window='Rectangular')
rg = ac.RectGrid(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z=-0.3, increment=0.01)
st = ac.SteeringVector(grid=rg, mics=m)

mf = ac.MaskedFreqOut(source=fft)
f = 8000

print(f"helmholz number ", m.aperture*f/343)
mf.freqs = [f]

sbl = ac.BeamformerSBL(
    source=mf,
    steer=st,
    num_snapshots=n_snap,
)
sbl.solver.method = 'SBL1'
sbl.solver.options['nsources'] = 10
sbl.solver.options['sig_sq'] = None

if True:
    sm = next(sbl.result(num=1))
    sm.reshape(-1, sbl.num_freqs, st.grid.size)
    sm = sm[0].reshape(rg.shape)   
    Lm = ac.L_p(sm)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(Lm.T, vmax=Lm.max(), vmin=Lm.max()-dyn, origin='lower', extent=(rg.x_min, rg.x_max, rg.y_min, rg.y_max))
    plt.colorbar(label='L_p (dB)')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Sparse Bayesian Learning Result')

    opt = Options(10 ** (-4), 10 ** (-4), niter, 1, 1, 2, 3, 1)
    spectra = next(mf.result(num=mf.num_samples))
    spectra = spectra.reshape(-1, mf.num_freqs, mf.num_channels)
    spectra = spectra[:, 0, :].T[..., np.newaxis]
    sm = SBL(A=st.transfer(mf.freqs[0]).T[..., np.newaxis], Y=spectra, options=opt)
    sm = sm[0].reshape(rg.shape)
    Lm = ac.L_p(sm)

    plt.subplot(1, 2, 2)
    plt.imshow(Lm.T, vmax=Lm.max(), vmin=Lm.max()-dyn, origin='lower', extent=(rg.x_min, rg.x_max, rg.y_min, rg.y_max))
    plt.colorbar(label='L_p (dB)')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Sparse Bayesian Learning Result')
    plt.show()


# %%
# .. seealso::
#    :doc:`example_basic_beamforming` for an example on how to load and analyze the generated data
#    with Beamforming.
