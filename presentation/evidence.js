// Reviewer presentation data. Seconds are one-second EEG patches; MF1 is
// expressed in percent, and gains and differences in percentage points.
// A classic script keeps the presentation usable directly from file://.
(() => {
  'use strict';

  const sources = {
    dataset: 'Supplement §II, Table II (pp. 2–3)',
    system: 'Manuscript §V-B, Fig. 2 (p. 7) and Table II (p. 8)',
    systemGain: 'Manuscript Fig. 2 (p. 7); matched-budget definition: Supplement §III-E (p. 5)',
    dreamsGain: 'Supplement §III-E (p. 5) gives +13.73 pp; Manuscript Fig. 2 rounds to +13.7 pp',
    baselineMf1: 'Manuscript §V-B / Fig. 2; scores recomputed at the reported peaks from the local 20-fold pooled evaluation counts',
    systemStages: 'Supplement §VI-A, Fig. 3 and Table XXII (p. 14)',
    matched: 'Manuscript §V-E, Table III(c) (p. 9) and Fig. 4 (p. 11); Supplement §VI-B, Table XXIV (p. 16) and Fig. 4 (p. 18)',
    history: 'Manuscript §V-D (p. 9), Fig. 3 (p. 10); Supplement §X, Figs. 10–11 (pp. 27–30)',
    atTenSeconds: 'Supplement §VIII, Fig. 7 (pp. 24, 26); interpolation at 10 s, 32-epoch inference window',
    risk: 'Supplement Fig. 6 (p. 25) and §VII-C (pp. 23–24)',
  };

  const datasets = [
    {
      id: 'dreams', name: 'DREAMS-SUB', epochs: 20242,
      system: {
        baseline: { name: 'EARLIEST', seconds: 29.76, mf1: 70.94 },
        firstExceeds: { seconds: 4.94, mf1: 75.02, eta: 0.06667 },
        gain: 13.73, gainPrecision: 2,
        byStage: { W: 6.29, N1: 19.50, N2: 5.56, N3: 3.01, REM: 18.86 },
      },
      matched: {
        gain: 2.21, peakGain: 0.71,
        byStage: { W: -0.16, N1: 3.34, N2: -0.45, N3: 1.20, REM: 1.71 },
      },
      atTenSeconds: 79.13,
      sources: { ...sources, systemGain: sources.dreamsGain },
    },
    {
      id: 'edf20', name: 'Sleep-EDF-20', epochs: 40293,
      system: {
        baseline: { name: 'FIRMBOUND', seconds: 23.67, mf1: 70.67 },
        firstExceeds: { seconds: 3.95, mf1: 73.84, eta: 0.06667 },
        gain: 12.8, gainPrecision: 1,
        byStage: { W: 9.70, N1: 13.30, N2: 3.79, N3: 5.43, REM: 16.96 },
      },
      matched: {
        gain: 1.78, peakGain: 0.89,
        // N3 uses the supplement's Table XXIV (main Table III(c) prints 2.92).
        byStage: { W: -0.22, N1: 3.57, N2: -0.45, N3: 2.93, REM: 0.70 },
      },
      atTenSeconds: 78.12,
      sources,
    },
    {
      id: 'edf78', name: 'Sleep-EDF-78', epochs: 162456,
      system: {
        baseline: { name: 'EARLIEST', seconds: 29.38, mf1: 66.22 },
        firstExceeds: { seconds: 4.30, mf1: 68.91, eta: 0.06667 },
        gain: 13.2, gainPrecision: 1,
        byStage: { W: 8.33, N1: 13.55, N2: 2.60, N3: 7.92, REM: 20.36 },
      },
      matched: {
        gain: 3.55, peakGain: 1.72,
        byStage: { W: -0.26, N1: 6.20, N2: -0.27, N3: 9.84, REM: 1.89 },
      },
      atTenSeconds: 73.28,
      sources,
    },
    {
      id: 'shhs', name: 'SHHS', epochs: 324148,
      system: {
        baseline: { name: 'FIRMBOUND', seconds: 24.75, mf1: 67.02 },
        firstExceeds: { seconds: 3.62, mf1: 67.93, eta: 0.13333 },
        gain: 11.4, gainPrecision: 1,
        byStage: { W: 3.65, N1: 18.25, N2: 3.61, N3: 3.92, REM: 13.20 },
      },
      matched: {
        gain: 3.92, peakGain: 2.68,
        byStage: { W: -0.18, N1: 15.11, N2: -0.37, N3: 5.28, REM: 1.44 },
      },
      atTenSeconds: 73.94,
      sources,
    },
  ];

  const history = {
    source: sources.history,
    ranges: {
      fullOnlyLoss: [0.89, 12.02],
      reduction: [0.27, 2.10],
      advantageOnPolicy: [0.42, 2.91],
      advantageOnComplete: [0.15, 0.96],
      oneSecondAdvantageAtTightestCost: [2.18, 3.90],
    },
    costs: [0.0005, 0.006667, 0.02, 0.03333, 0.06667, 0.13333, 0.23333, 0.5],
    // Each row is I = (A - B) - (C - D) in manuscript Fig. 3(a).
    reductionByDataset: {
      dreams: [0.94, 0.87, 1.12, 0.70, 1.20, 1.94, 1.95, 1.93],
      edf20: [0.53, 0.37, 0.58, 0.85, 0.75, 1.16, 1.65, 1.41],
      edf78: [0.70, 1.03, 1.23, 1.33, 1.27, 0.87, 0.29, 0.27],
      shhs: [0.43, 0.83, 1.35, 1.60, 2.10, 2.04, 1.36, 0.27],
    },
    // Fig. 3(b), η=0.5: Paired minus Full-only at the 1 s prefix,
    // comparing the two backbones with the same policy-generated history.
    oneSecondByDataset: { dreams: 3.90, edf20: 3.59, edf78: 2.34, shhs: 2.18 },
    examples: [
      { id: 'shhs', eta: 0.06667, reduction: 2.10, advantageOnPolicy: 2.25 },
      { id: 'shhs', eta: 0.5, reduction: 0.27, advantageOnPolicy: 0.42, oneSecondAdvantageOnPolicy: 2.18 },
      { id: 'dreams', eta: 0.5, reduction: 1.93, oneSecondAdvantageOnPolicy: 3.90 },
    ],
  };

  const risk = {
    image: 'risk-trajectories.png',
    source: sources.risk,
    eta: 0.02,
    dataset: 'DREAMS-SUB',
    stages: ['W', 'N1', 'N2', 'N3', 'REM'],
    caption: 'Illustrative DREAMS-SUB risk trajectories at η = 0.02. Panel titles are reference stages. Orange solid: stop risk. Green dashed: fitted continuation risk. The dot marks the first prefix where stop risk is no greater than continuation risk. The plotted suffix after each dot is an offline counterfactual and is unavailable to the online policy.',
  };

  window.PAUSE_EVIDENCE = { datasets, history, sources, risk };
})();
