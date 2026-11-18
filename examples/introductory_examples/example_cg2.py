# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
# %%
"""
CG2 and subspace reduction: a two-source comparison
==================================================

Reconstruct the four panels of Fig. 3 in W. Ma and X. Liu, *DAMAS with compression computational
grid for acoustic source mapping*, Journal of Sound and Vibration 410 (2017), 473--484,
https://doi.org/10.1016/j.jsv.2017.03.027.

The paper specifies a 30-microphone array of diameter 0.35 m, a 50-by-50 grid at z = 0.5 m,
a 60-degree opening angle, c = 340 m/s, and f = 8 kHz. Two incoherent monopoles occupy the
one-based grid locations (25, 25) and (31, 25), with powers 1 and 0.1 Pa squared. Both DAMAS
calculations use 1000 iterations and a zero initial solution. CG2 retains positive values of
the diagonal-removed delay-and-sum map. All panels share a -30 to 0 dB scale relative to
1 Pa squared.

Eight additional panels compare DR-AE and DR-PE from Feng et al. (2026),
https://doi.org/10.1121/10.0046883, using the same CSM, original grid, and DAMAS solver.
Both methods are evaluated with ``n = 2`` (the true source count) and ``n = num_mics // 2 = 15``
(half the microphones). The simulation always contains two physical sources. At ``n = M/2``
the signal and noise subspaces have equal dimensions, so DR-PE does not truncate either subspace
and is identical to DR-AE. Equal masks and reconstructions are expected only for that setting.
Panel titles report the number of retained points relative to the original grid size.

**Reproduction limits:** the paper does not supply microphone coordinates. The positions below
are approximate digitizations of its Fig. 2, not the authors' original geometry. The circle in
the 138-by-138-pixel figure is calibrated to diameter 0.35 m, with centre (69, 70) and radius
66 pixels. Image y coordinates are inverted to obtain physical y coordinates.

The paper also does not specify the snapshot count or random seed for its 15 dB Gaussian noise.
Here, 1024 independent complex Gaussian frequency-domain snapshots and seed 2017 are used.
SNR is defined from the population signal power averaged over microphones. Thus the figure is
a reproducible reconstruction of the scenario, not an exact numerical replication of Table 1.
The reported 898 retained points, compression ratio 2.8, and historical runtimes are not targets.

A :class:`~acoular.spectra.PowerSpectraImport` supplies the sample CSM. The preliminary map and
PSFs use the ``inverse`` steering formulation, matching the distance weighting in Ma's Eq. (4).
The DAMAS kernel is called explicitly because :class:`~acoular.fbeamform.BeamformerDamas`
initializes from the dirty map, whereas the paper initializes from zero. Its full-diagonal PSF
is used, as in the paper's Eq. (12). CG2 needs no changes to the reconstruction algorithm.

Run this file directly to display the comparison. No measurement files or original-paper assets
are needed at runtime. Persistent caching is disabled during calculation, and JIT kernels are
warmed up before the timed comparisons.
"""

from time import perf_counter

import acoular as ac
from acoular.fastFuncs import damasSolverGaussSeidel

import matplotlib.pyplot as plt
import numpy as np

FREQUENCY = 8000.0
N_ITER = 1000

# Approximate marker centres digitized from Fig. 2, in image pixels (x right, y down).
MICROPHONE_PIXELS = np.array(
    [
        (74.0, 5.2),
        (89.8, 14.7),
        (97.3, 26.0),
        (99.6, 39.8),
        (95.7, 52.8),
        (85.9, 63.8),
        (130.3, 55.1),
        (127.0, 73.1),
        (118.8, 83.7),
        (106.8, 89.8),
        (92.6, 90.2),
        (79.2, 84.4),
        (103.5, 125.0),
        (83.7, 126.4),
        (70.9, 121.6),
        (61.0, 112.1),
        (56.4, 99.1),
        (58.0, 84.3),
        (26.1, 117.7),
        (19.2, 101.9),
        (19.7, 87.6),
        (25.9, 75.7),
        (36.9, 67.0),
        (51.4, 63.9),
        (9.7, 44.4),
        (22.9, 32.6),
        (36.4, 28.9),
        (49.7, 31.0),
        (61.2, 38.7),
        (68.9, 51.4),
    ],
)


# %%
# Build the original grid, digitized array, and a reproducible sample CSM.
def create_setup(seed=2017, num_snapshots=1024):
    """Create the Fig. 3 scenario using the documented geometry and sampling approximations."""
    xy = (MICROPHONE_PIXELS - np.array([69, 70])) * (0.35 / (2 * 66))
    xy[:, 1] *= -1
    mics = ac.MicGeom(pos_total=np.vstack([xy.T, np.zeros(30)]))

    length = 2 * 0.5 * np.tan(np.deg2rad(60 / 2))
    grid = ac.RectGrid(
        x_min=-length / 2,
        x_max=length / 2,
        y_min=-length / 2,
        y_max=length / 2,
        z=0.5,
        increment=length / 49,
    )
    steer = ac.SteeringVector(grid=grid, mics=mics, env=ac.Environment(c=340), steer_type='inverse')
    # Convert the paper's one-based (x, y) grid indices to Acoular's flat ordering.
    source_indices = np.ravel_multi_index(([24, 30], [24, 24]), grid.shape)
    source_powers = np.array([1.0, 0.1])
    transfer = steer.transfer(FREQUENCY, source_indices).T

    rng = np.random.default_rng(seed)
    sources = (rng.normal(size=(2, num_snapshots)) + 1j * rng.normal(size=(2, num_snapshots))) / np.sqrt(2)
    pressure = transfer @ (np.sqrt(source_powers)[:, np.newaxis] * sources)
    mean_signal_power = np.sum(np.abs(transfer) ** 2 * source_powers[np.newaxis, :]) / mics.num_mics
    noise_variance = mean_signal_power / 10 ** (15 / 10)
    noise = (rng.normal(size=pressure.shape) + 1j * rng.normal(size=pressure.shape)) * np.sqrt(noise_variance / 2)
    pressure += noise
    csm = pressure @ pressure.conj().T / num_snapshots
    spectra = ac.PowerSpectraImport(csm=csm[np.newaxis], frequencies=FREQUENCY)
    return steer, spectra, source_indices, source_powers


# %%
# Compare three selection rules without changing the two physical sources.
def create_compressions(steer, spectra):
    """Configure CG2 and each dictionary reduction with true and half-microphone source counts."""
    reductions = {'CG2': ac.CG2(steer=steer, freq_data=spectra, freq=FREQUENCY, threshold=0)}
    for n in (2, steer.mics.num_mics // 2):
        for name, method in [('DR-AE', ac.DRAE), ('DR-PE', ac.DRPE)]:
            reductions[f'{name} (n={n})'] = method(steer=steer, freq_data=spectra, freq=FREQUENCY, n=n)
    return reductions


# %%
# Use Acoular's PSF and DAMAS kernel, but start from zero as specified in the paper.
def solve_damas(steer, dirty_map, n_iter=N_ITER):
    """Return the solution, PSF construction time, and iteration time for the supplied grid."""
    start = perf_counter()
    psf = ac.PointSpreadFunction(steer=steer, freq=FREQUENCY, calcmode='full', cached=False).psf
    psf_time = perf_counter() - start
    solution = np.zeros(steer.grid.size)
    start = perf_counter()
    damasSolverGaussSeidel(psf, np.ascontiguousarray(dirty_map), n_iter, 1.0, solution)
    return solution, psf_time, perf_counter() - start


# %%
# Plot all reconstructions on the original grid. Only the solve uses a smaller grid.
def plot_maps(grid, source_positions, dirty_map, full_map, compressed, reduced_map, *, additional_reductions=()):
    """Draw the original four panels and optional reduction rows on a common colour scale."""
    step = grid.increment
    x_edges = np.linspace(grid.x_min - step / 2, grid.x_max + step / 2, grid.nxsteps + 1)
    y_edges = np.linspace(grid.y_min - step / 2, grid.y_max + step / 2, grid.nysteps + 1)
    cmap = plt.get_cmap('jet').with_extremes(under='white')
    panels = [
        (f'(a) Delay-and-sum map\n{grid.size} points', dirty_map, None),
        (f'(b) DAMAS, original grid\n{grid.size} points; {N_ITER} iterations', full_map, None),
        (f'(c) CG2\n{compressed.size}/{grid.size} points retained', dirty_map, compressed),
        (f'(d) DAMAS, CG2 grid\n{compressed.size}/{grid.size} points; {N_ITER} iterations', reduced_map, None),
    ]
    for name, subset, reconstruction in additional_reductions:
        label = chr(ord('a') + len(panels))
        panels.append((f'({label}) {name}\n{subset.size}/{grid.size} points retained', dirty_map, subset))
        label = chr(ord('a') + len(panels))
        panels.append(
            (f'({label}) DAMAS, {name}\n{subset.size}/{grid.size} points; {N_ITER} iterations', reconstruction, None)
        )
    rows = len(panels) // 2
    figure, axes = plt.subplots(rows, 2, figsize=(10, 4.5 * rows), sharex=True, sharey=True, layout='constrained')
    for axis, (title, values, subset) in zip(axes.flat, panels, strict=True):
        levels = 10 * np.log10(np.maximum(values.reshape(grid.shape).T, 1e-30))
        image = axis.pcolormesh(
            x_edges,
            y_edges,
            levels,
            cmap=cmap,
            vmin=-30,
            vmax=0,
            shading='flat',
            edgecolors='0.6',
            linewidth=0.15,
        )
        axis.plot(source_positions[0], source_positions[1], '+', color='white', markersize=7, markeredgewidth=1.5)
        axis.set(title=title, xlabel='x (m)', ylabel='y (m)', aspect='equal')
        axis.set_xticks(np.arange(-0.2, 0.21, 0.1))
        axis.set_yticks(np.arange(-0.2, 0.21, 0.1))
        if subset is not None:
            axis.scatter(
                subset.pos[0],
                subset.pos[1],
                s=8,
                facecolors='none',
                edgecolors='royalblue',
                linewidths=0.4,
            )
    figure.colorbar(image, ax=axes, label=r'dB re 1 Pa$^2$', ticks=np.arange(-30, 1, 3), shrink=0.85)
    title = 'Ma and Liu (2017), Fig. 3 scenario'
    if additional_reductions:
        title += ' with Feng dictionary reductions'
    figure.suptitle(title + '\nApproximate digitized array; seeded finite-snapshot CSM')
    return figure


# %%
# Compare full-grid DAMAS against exactly the same solve on each compressed subset.
def main():
    """Run the comparison, print diagnostics, and return its figure."""
    previous_caching = ac.config.global_caching
    ac.config.global_caching = 'none'
    try:
        steer, spectra, source_indices, source_powers = create_setup()
        grid = steer.grid
        source_positions = grid.pos[:, source_indices]
        beamformer = ac.BeamformerBase(freq_data=spectra, steer=steer, r_diag=True, cached=False)
        dirty_map = beamformer.synthetic(FREQUENCY).ravel()

        # Warm up the same PSF/solver kernels on a two-point grid before measuring runtimes.
        warmup_steer = ac.SteeringVector(
            grid=ac.ImportGrid(pos=source_positions), mics=steer.mics, env=steer.env, steer_type=steer.steer_type
        )
        solve_damas(warmup_steer, dirty_map[source_indices], n_iter=1)

        full_map, full_psf_time, full_iteration_time = solve_damas(steer, dirty_map)

        # Measure the shared eigendecomposition separately, avoiding an unfair first-method cost.
        start = perf_counter()
        _ = spectra.eigvecs
        evd_time = perf_counter() - start
        reductions = create_compressions(steer, spectra)
        reconstructions = {}
        timings = {}
        for name, compressed in reductions.items():
            start = perf_counter()
            indices = compressed.indices
            selection_time = perf_counter() - start
            reduced_steer = ac.SteeringVector(
                grid=compressed, mics=steer.mics, env=steer.env, steer_type=steer.steer_type
            )
            values, psf_time, iteration_time = solve_damas(reduced_steer, dirty_map[indices])
            reconstruction = np.zeros(grid.size)
            reconstruction[indices] = values
            reconstructions[name] = reconstruction
            timings[name] = (selection_time, psf_time, iteration_time)
    finally:
        ac.config.global_caching = previous_caching

    print('Approximate reconstruction of Ma and Liu (2017), Fig. 3')
    print('Digitized microphone geometry; 1024 complex snapshots; seed 2017; SNR 15 dB')
    print(f'Grid: {grid.shape}; source separation: {np.linalg.norm(np.diff(source_positions, axis=1)):.5f} m')
    half_mics = steer.mics.num_mics // 2
    print(f'DR-AE/DR-PE compare n=2 (true source count) with n={half_mics} (half the microphones).')
    print('At n=M/2, DR-PE equals DR-AE: neither subspace is truncated.')
    balanced_ae = reductions[f'DR-AE (n={half_mics})']
    balanced_pe = reductions[f'DR-PE (n={half_mics})']
    print(f'Identical DR-AE/DR-PE masks at n={half_mics}: {np.array_equal(balanced_ae.mask, balanced_pe.mask)}')
    for name, compressed in reductions.items():
        print(
            f'{name} retains {compressed.size}/{grid.size} points; compression ratio: {grid.size / compressed.size:.2f}'
        )
        print(f'  Both true source points retained: {compressed.mask[source_indices].all()}')
    # Integrate over the two nearest-source regions; their union is the entire original grid.
    nearest = np.linalg.norm(grid.pos[:, :, np.newaxis] - source_positions[:, np.newaxis, :], axis=0).argmin(axis=1)
    print(f'True source powers: {source_powers} Pa^2')
    maps = {'DAMAS': full_map}
    maps.update({f'DAMAS-{name}': values for name, values in reconstructions.items()})
    for name, values in maps.items():
        powers = np.bincount(nearest, weights=values, minlength=2)
        error = abs(values.sum() - source_powers.sum()) / source_powers.sum()
        print(f'{name}: region powers {powers} Pa^2; total {values.sum():.4f} Pa^2; total-power error {error:.2%}')
    print(f'Original grid: PSF {full_psf_time:.3f} s; iterations {full_iteration_time:.3f} s')
    print(f'Shared eigendecomposition: {evd_time:.3f} s (charged to each DR method in the speed-up comparison)')
    full_time = full_psf_time + full_iteration_time
    for name, (selection_time, psf_time, iteration_time) in timings.items():
        method_time = selection_time + psf_time + iteration_time
        if name != 'CG2':
            method_time += evd_time
        print(
            f'{name}: selection {selection_time:.3f} s; PSF {psf_time:.3f} s; iterations {iteration_time:.3f} s; '
            f'speed-up {full_time / method_time:.2f}x'
        )
    print('Speed-ups exclude the common DAS map and JIT warm-up; persistent caching is disabled.')
    print('Paper CG2 reference: 898 retained points, ratio 2.8; differences are expected from our approximations.')
    return plot_maps(
        grid,
        source_positions,
        dirty_map,
        full_map,
        reductions['CG2'],
        reconstructions['CG2'],
        additional_reductions=[
            (name, subset, reconstructions[name]) for name, subset in reductions.items() if name != 'CG2'
        ],
    )


if __name__ == '__main__':
    main()
    plt.show()
