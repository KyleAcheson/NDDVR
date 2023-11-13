# NDDVR

A python module for solving the TISE using discrete-variable representation.

- Solve up to 3D problems using Colbert-Miller DVR
- Solve up to 2D problems for DVR type algorithms learned from program synthesis (`synthesised_algorithms.py`)

![Calculated Eigenstate of 2D-Harmonic Oscillator](figures/2D_HO_neig16.png)


### TBC:
- Extend program synthesis DVR algorithms to ND - utilising sparse matrices
- Expand `potential_functions.py` to include more than Harmonic oscillators
- Test PS algorithms on 2D and 3D molecular systems
- Currently some issues with evaluating expectation values - likely FFT related
