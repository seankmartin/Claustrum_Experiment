# A set of files for behavioural testing while recording in the Claustrum

## Maintainers

Originally created and currently maintained by Sean Martin and Gao Xiang Ham.

## Documentation

Full API documentation for bvmpc is available at https://seankmartin.github.io/Claustrum_Experiment/html/bvmpc/index.html which was created using pdoc3 from Python docstrings.

## Contents

### Top Level

Various batch scripts and helper files. Of real note, the control scripts `bv_control.py` and `bv_ctl_lfp.py` should be considered the main entry point.

### Axona

Basic style scripts to run with DACQUSB as control scripts

### MEDPC

Files in the .MPC language and our own pmpc custom language which is parsed into .MPC using parse_mpc.py

### docs

html documentation on the bvmpc module.

### configs

Configuration files for the control scripts.

### bvmpc

A python module to analyse the behavioural task we designed using Axona and MedPC ouptuts. The module should be generic enough to provide a basis for working with this kind of task in the future. Can be installed by typing pip install . in this directory.

bvmpc contains a number of modules, which are roughly described here:

- bv_analyse.py - An assorted set of behavioural analyses.
- bv_array_methods.py - Helper functions for performing simple routines on lists and NumPy arrays.
- bv_axona.py - Read Axona .inp files and Axona .set files.
- bv_batch.py - Holds batch related functions. For example, plotting the performance of an animal across multiple days.
- bv_file.py - File reading and conversion functions for behavioural data.
- bv_mne.py - Interfacing with the MNE python library.
- bv_nc.py - Interfacing with NeuroChaT and phy and spikeinterface.
- bv_plot.py - Plotting utilities and also some behavioural plots.
- bv_session.py - Holds all of the routines related to single behavioural sessions from MEDPC, Axona, HDF5, and neo.
- bv_session_extractor.py - Holds all of the routines related to parsing multiple sessions out of a single MedPC file into a list.
- bv_session_info.py - This is used to hold metadata and setup information about the MEDPC and Axona behavioural sessions. For example, it denotes which I/O pins are used in an experiment and what they represent.
- bv_utils.py - An assorted set of utility functions for simple but widely used functionality.
- compare_lfp.py - Compare lfp signals to check their similarity.
- lfp_coherence.py - Coherence, Wavelet coherence, mean vector length, and other similar LFP calculations.
- lfp_odict.py - Store Neurochat LFP signals in a dictionary to ease multi-channel analysis.
- lfp_plot.py - Plot things such as coherence and power spectrum.
