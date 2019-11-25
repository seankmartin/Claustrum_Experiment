# A set of files for behavioural testing while recording in the Claustrum

## Maintainers

Originally created and currently maintained by Sean Martin and Gao Xiang Ham.

## Contents

- Top Level: Various batch scripts and helper files. Of real note, is bv_control.py, which is the control script for the bvmpc module.
- Axona: Basic style scripts to run with DACQUSB as control scripts
- MEDPC: Files in the .MPC language and our own pmpc custom language which is parsed into .MPC using parse_mpc.py
- bvmpc: A python module to analyse the behavioural task we designed using Axona and MedPC ouptuts. The module should be generic enough to provide a basis for working with this kind of task in the future. Can be installed by typing pip install . in this directory.
- docs: html documentation on the bvmpc module.
