# Training Dynamics and Hyperparameter Sensitivity

## Training Dynamics

The mask network is optimized during the first 20 epochs and frozen afterward.
Over 100 epochs, balanced accuracy stabilizes near 0.973, representation
dispersion approaches 0.042, PUE stabilizes near 0.965, and cumulative privacy
loss reaches the configured budget of \(\varepsilon=0.114\). Mask adaptation is
part of private training and is included in the accountant.

## Hyperparameter Sensitivity

The manuscript varies one parameter at a time while holding the others fixed.

- Data minimization: balanced accuracy remains above 96% for
  \(\lambda_{sparse}\in\{10^{-5},10^{-4},10^{-3}\}\) and reaches 97.3% at
  the selected value \(10^{-4}\). The two outer tested settings reported in the
  figure obtain 93.2% and 94.5%.
- Structural privacy design: balanced accuracy remains above 96% for
  \(\lambda_{design}\in\{10^{-4},10^{-3},10^{-2}\}\) and reaches 97.3% at
  the selected value \(10^{-3}\).
- Mask adaptation: balanced accuracy increases from 94.1% at
  \(T_{warm}=0\) to 97.3% at 20 epochs, then remains between 97.0% and 97.2%
  through 40 epochs.

Only values stated numerically in the manuscript are recorded here; unreported
figure coordinates are not inferred.
