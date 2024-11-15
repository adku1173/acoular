=======================
Key Concepts
=======================



Object Oriented Programming (OOP)
=================================
Acoular is structured around an object-oriented programming paradigm. 
Therefore, most of the Acoular source code is organized into `classes <https://docs.python.org/3/tutorial/classes.html>`_ that represent objects (or concepts) with a specific functionality.
Each class exposes a set of methods and attributes that can be used to interact with the object.
Through the use of inheritance (sub-classing), new classes can leverage existing code for shared functionalities, facilitating the addition of new processing capabilities. 
Currently, Acoular comprises |numclasses| classes across |nummodules| modules, including:

- **Data Acquisition and Import**: Handling of both streaming and file-based measurement data.
- **Signal Processing**: Linear and nonlinear filtering in the time and frequency domain.
- **Beamforming and Advanced Microphone Array Methods**: Spatial filtering, noise source characterization and spectrum estimation techniques in time and frequency domain.
- **Environment and Motion Modeling**: Different sound propagation models and support for moving sources.
- **Simulation and Synthesis**: Tools for artificial data generation, including the simulation of moving sources in complex environments.


The concept of inheritance can be understood visually from the inheritance diagram of Acoular's signal generator classes:

.. inheritance-diagram:: acoular.signals
    :top-classes: acoular.signals.SignalGenerator
    :parts: 1



Lazy Evaluation
===============

Acoular employs lazy evaluation, where computational steps are set up initially but only executed when results are explicitly needed. This minimizes unnecessary computations, making the package efficient for applications with high computational demands. Consider the following example, where a white noise signal with a length of one second is created. The signal generation is only performed when the :meth:`signal` method is explicitly called:

.. code-block:: python

    import acoular as ac

    wn = ac.WNoiseGenerator(sample_freq=51200, numsamples=51200, seed=1) # create a white noise signal generator (no computation at all)
    signal = wn.signal() # compute the white noise signal




Generators and Generator Pipelines
===================================

Many classes in Acoular are subclasses of the :class:`~acoular.base.Generator` class which implements the :meth:`result` generator method. Python generators allow iterating over a sequence of items without storing the whole sequence in memory. This of particular importance when dealing with long measurements or streaming applications.
Acoular also integrates **generator pipelines**, a sequential architecture where each generator’s output is the source to the next generator in the chain. For instance, consider a pipeline for computing an averaged auto-power spectrum from a time-domain signal:

.. code-block:: python

    import acoular as ac

    signal = ac.WNoiseGenerator(sample_freq=51200, numsamples=51200).signal().reshape(-1, 1)
    ts = ac.TimeSamples(data=signal, sample_freq=51200)
    fft = ac.RFFT(source=ts) # perform a FFT of the time signal
    power = ac.AutoPowerSpectra(source=fft) # calculate the auto-power spectrum
    avg = ac.Average(source=power, naverage=10) # average the auto-power spectrum over 10 snapshots

    # iterate over the generator pipeline
    for p in avg.result():
        print(p)

Caching
=======

Acoular implements efficient, intelligent transparent and persistent caching of intermediate and final results. Two different caching mechanisms are available.

1. **Cached Properties**: Utilizes in-memory storage of class properties (specific attributes that call a function to determine its value) for quick access during runtime but does not persist across program executions.
2. **Persistent File Cache**: Uses HDF5 files to store results persistently, allowing for reuse across sessions.



Cached Properties
------------------

Acoular leverages the `Traits <https://docs.enthought.com/traits/>`_ library, specifically the :func:`cached_property` decorator, to cache results from computationally intensive tasks. The result remains in memory, accessible on subsequent calls, and is automatically cleared if any dependent attribute or object changes.

Acoular provides a unique hashing mechanism for each object, allowing dependent objects and users to trace the exact configuration of processing instances. By calling the :attr:`digest` property, one can access a unique hash that represents the current object state. Any modification, such as altering microphone positions, triggers a new hash value, thereby capturing the updated object state:

Acoular classes have a hashing mechanism that allows to uniquely identify the state of an object. The current hash / state can be obtained by calling the `digest` property. For example, let's create a microphone geometry instance and calculate its hash:

.. code-block:: python

    from acoular import MicGeom
    m = MicGeom()
    print(m.digest)

If we add a microphone to the geometry, the hash will change:

.. code-block:: python

    import numpy as np
    m.mpos_tot = np.array([[0],[0],[0]])
    print(m.digest)


File Cache
----------

# hdf5 files are used for caching
# currently, file caching can be used at any point of a generator pipeline and is for example implemented in all Beamformer classes.


Acoular provides global settings to control the caching behavior. The following caching modes are available:

.. list-table:: Caching Modes
   :header-rows: 1

   * - Cache Mode
     - Description
   * - None
     - No caching
   * - Overwrite
     - Overwrites cached results
   * - Read-Only
     - Uses cached results without modification
   * - All
     - Enables both memory and file caching
   * - Individual
     - Customizable caching based on specific needs



Parallelization
===============


Numba


