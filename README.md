# A set of files for behavioural testing while recording in the Claustrum

## Maintainers

Originally created and currently maintained by Sean Martin and Gao Xiang Ham.

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
- bv_axona.py -
