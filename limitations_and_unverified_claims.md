# Limitations and claims that remain unverified

## Limitations of the experiment
1. One-capacitance thermal model; every parameter (including the COP slope) is assumed.
2. No calibration against, or comparison with, a real building; no measured data; no IoT or hardware-in-the-loop.
3. One weather shape, one occupancy schedule, four disturbances, 20 seeds.
4. Hyper-parameters tuned on development seeds; selected trigger factor (1.5) is the smallest grid value, so the true optimum may lie outside the grid. The always-on arm received an equal-sized grid.
5. Thermostat baseline is coarse (two stages, 15-min sampling) and has less information than the AI controllers. The energy gap to it must not be generalised. No tuned MPC was compared.
6. Absolute effects are small (<= ~0.01 degC overall MAE) relative to 0.1 degC sensor noise.
7. 152 paired tests were computed; they are exploratory (Bonferroni markers are given in the paper).
8. Cooling only; no heating, humidity, or ventilation control.
9. Controller timings are machine-dependent.

## Claims not verified
* Full texts of the nine sources were not reviewed (abstract/summary level only); [1]'s findings are unknown.
* Page range of [3] and article numbers for [5], [6], [8] are unconfirmed; [5] and [6] were verified only as arXiv preprints (journal versions not checked).
* "No prior study isolates gated vs. always-on vs. frozen updating in a single-zone HVAC simulation" is based on a limited search and must be re-checked in Google Scholar / IEEE Xplore before submission.
* The explanation that a falling COP drives the fixed predictor's error is a hypothesis; it was not tested by a targeted sweep.
* Cause of the larger parameter drift in always-on RLS was not isolated.
* Only [9] is a standard; no IEEE Xplore papers were included. Add 4-6 IEEE-indexed papers through institutional access.
* AI-disclosure wording and plagiarism/similarity checking are the authors' responsibility; no similarity or AI-detector score was measured or promised.
