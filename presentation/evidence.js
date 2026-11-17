// Reviewer presentation data. Seconds are one-second EEG patches; MF1 is
// expressed in percent, and gains and differences in percentage points.
// A classic script keeps the presentation usable directly from file://.
(() => {
  'use strict';

  const sources = {
    dataset: 'Supplement §II, Table II (pp. 2–3)',
    system: 'Manuscript §V-B, Fig. 2 (p. 7); PAUSE operating points also appear in Table II (p. 8) and low-budget baseline comparison in Table I (p. 8)',
    systemGain: 'Manuscript Fig. 2 (p. 7); matched-budget definition: Supplement §III-E (p. 5)',
    dreamsGain: 'Supplement §III-E (p. 5) gives +13.73 pp; Manuscript Fig. 2 rounds to +13.7 pp',
    baselineMf1: 'Manuscript §V-B / Fig. 2; scores recomputed at the reported peaks from the local 20-fold pooled evaluation counts',
    systemStages: 'Supplement §VI-A, Fig. 3 and Table XXII (p. 14)',
    matched: 'Manuscript §V-E, Table III(c) (p. 9) and Fig. 4 (p. 11); Supplement §VI-B, Table XXIV (p. 16) and Fig. 4 (p. 18) (Sleep-EDF-20 N3: Main Table III(c) prints 2.92; Supplement Table XXIV prints 2.93)',
    history: 'Manuscript §V-D (p. 9), Fig. 3 (p. 10); Supplement §X, Figs. 10–11 (pp. 27–30)',
    atTenSeconds: 'Supplement §VIII, Fig. 7 (pp. 24, 26); interpolation at 10 s, 32-epoch inference window',
    risk: 'Supplement Fig. 6 (p. 25) and §VII-C (pp. 23–24)',
    ablationSweep: 'Manuscript §V-C, Table II (p. 8)',
    ablationStages: 'Manuscript §V-C, Table III(a)–(b) (p. 9); Supplement §VI-C, Table XXVI (p. 17)',
    fullObservation: 'Manuscript §V-E1, Table IV (p. 10); Supplement §VI-D, Table XXVII (p. 19)',
    classWeight: 'Manuscript §V-E3, Table V (p. 10); Supplement §IV-C2, Table VIII (p. 11)',
    spectralChange: 'Manuscript §V-F, Table VI (p. 11); Supplement §IX, Table XXXIV (p. 28)',
    onlineMass: 'Supplement §VII-A, Tables XXVIII–XXIX (pp. 20–21)',
    deployment: 'Manuscript §V-G, Tables VII–VIII (p. 12)',
  };

  const ABBREVIATED_COSTS = [0.5, 0.067, 0.0005];
  const makeAblationSweep = rows => rows.map(([eta, pM, pT, wpM, wpT, weM, weT]) => ({
    eta,
    pause: { mf1: pM, seconds: pT },
    woPairedHistory: { mf1: wpM, seconds: wpT },
    woEndpointMemory: { mf1: weM, seconds: weT },
  }));
  const selectKeyTable2Costs = sweep => sweep.filter(entry => ABBREVIATED_COSTS.includes(entry.eta));

  const makeDatasetAblation = (sweepRows, vsWoEndpointMemory, vsWoPairedHistory) => {
    const table2Full = makeAblationSweep(sweepRows);
    return {
      source: sources.ablationSweep,
      stagesSource: sources.ablationStages,
      costs: ABBREVIATED_COSTS.slice(),
      table2: selectKeyTable2Costs(table2Full),
      table2Full,
      // Manuscript Table III(a)-(b) matched-budget Pareto gains (pp).
      paretoGains: { vsWoEndpointMemory, vsWoPairedHistory },
    };
  };

  const datasets = [
    {
      id: 'dreams',
      name: 'DREAMS-SUB',
      epochs: 20242,
      recordings: 20,
      subjects: 20,
      channel: 'Cz-A1',
      stageShares: { W: 17.58, N1: 7.31, N2: 40.76, N3: 19.43, REM: 14.91 },
      system: {
        baseline: { name: 'EARLIEST', seconds: 29.76, mf1: 70.94 },
        firstExceeds: { seconds: 4.94, mf1: 75.02, eta: 0.06667 },
        gain: 13.73,
        gainPrecision: 2,
        byStage: { W: 6.29, N1: 19.50, N2: 5.56, N3: 3.01, REM: 18.86 },
      },
      matched: {
        gain: 2.21,
        peakGain: 0.71,
        byStage: { W: -0.16, N1: 3.34, N2: -0.45, N3: 1.20, REM: 1.71 },
      },
      atTenSeconds: 79.13,
      ablation: makeDatasetAblation(
        [
          [0.5, 60.34, 1.1, 50.65, 1.1, 55.85, 2.4],
          [0.233, 65.14, 1.7, 55.29, 1.4, 64.01, 5.4],
          [0.133, 70.05, 2.7, 60.30, 2.2, 69.75, 9.4],
          [0.067, 75.02, 4.9, 67.33, 5.0, 73.00, 14.8],
          [0.033, 77.27, 7.7, 72.44, 10.6, 74.15, 18.2],
          [0.02, 79.11, 9.9, 75.84, 13.7, 74.34, 19.8],
          [0.0067, 80.12, 13.8, 74.58, 17.6, 74.60, 21.0],
          [0.0005, 80.63, 17.0, 75.38, 19.7, 74.68, 21.6],
        ],
        { mf1: 9.53, byStage: { W: 4.11, N1: 19.57, N2: 6.36, N3: 1.70, REM: 15.27 } },
        { mf1: 6.58, byStage: { W: 3.09, N1: 15.40, N2: 2.85, N3: 6.66, REM: 7.11 } }
      ),
      sources: { ...sources, systemGain: sources.dreamsGain },
    },
    {
      id: 'edf20',
      name: 'Sleep-EDF-20',
      epochs: 40293,
      recordings: 39,
      subjects: 20,
      channel: 'Fpz-Cz',
      stageShares: { W: 16.28, N1: 6.86, N2: 43.75, N3: 13.97, REM: 19.14 },
      system: {
        baseline: { name: 'FIRMBOUND', seconds: 23.67, mf1: 70.67 },
        firstExceeds: { seconds: 3.95, mf1: 73.84, eta: 0.06667 },
        gain: 12.8,
        gainPrecision: 1,
        byStage: { W: 9.70, N1: 13.30, N2: 3.79, N3: 5.43, REM: 16.96 },
      },
      matched: {
        gain: 1.78,
        peakGain: 0.89,
        // Main Table III(c) prints 2.92; Supplement Table XXIV prints 2.93.
        n3SupplementTable24: 2.93,
        byStage: { W: -0.22, N1: 3.57, N2: -0.45, N3: 2.92, REM: 0.70 },
      },
      atTenSeconds: 78.12,
      ablation: makeDatasetAblation(
        [
          [0.5, 63.06, 1.1, 57.69, 1.1, 58.99, 2.5],
          [0.233, 66.72, 1.5, 61.54, 1.4, 65.42, 4.7],
          [0.133, 69.88, 2.2, 63.69, 1.9, 69.28, 7.5],
          [0.067, 73.84, 3.9, 69.14, 3.6, 72.27, 11.6],
          [0.033, 76.62, 6.4, 73.24, 6.9, 73.52, 14.9],
          [0.02, 77.63, 8.2, 75.42, 9.8, 73.85, 16.7],
          [0.0067, 78.40, 10.8, 76.41, 14.1, 74.12, 19.1],
          [0.0005, 79.56, 15.9, 77.41, 17.2, 74.21, 20.6],
        ],
        { mf1: 7.72, byStage: { W: 8.64, N1: 10.80, N2: 3.64, N3: 4.16, REM: 11.61 } },
        { mf1: 3.47, byStage: { W: 2.58, N1: 3.83, N2: 2.17, N3: 5.18, REM: 3.94 } }
      ),
      sources,
    },
    {
      id: 'edf78',
      name: 'Sleep-EDF-78',
      epochs: 162456,
      recordings: 153,
      subjects: 78,
      channel: 'Fpz-Cz',
      stageShares: { W: 22.03, N1: 12.52, N2: 41.65, N3: 7.94, REM: 15.86 },
      system: {
        baseline: { name: 'EARLIEST', seconds: 29.38, mf1: 66.22 },
        firstExceeds: { seconds: 4.30, mf1: 68.91, eta: 0.06667 },
        gain: 13.2,
        gainPrecision: 1,
        byStage: { W: 8.33, N1: 13.55, N2: 2.60, N3: 7.92, REM: 20.36 },
      },
      matched: {
        gain: 3.55,
        peakGain: 1.72,
        byStage: { W: -0.26, N1: 6.20, N2: -0.27, N3: 9.84, REM: 1.89 },
      },
      atTenSeconds: 73.28,
      ablation: makeDatasetAblation(
        [
          [0.5, 54.18, 1.1, 50.92, 1.1, 50.77, 1.9],
          [0.233, 60.32, 1.6, 58.28, 1.7, 57.28, 3.9],
          [0.133, 64.44, 2.4, 62.18, 2.4, 62.02, 6.7],
          [0.067, 68.91, 4.3, 66.41, 4.3, 66.30, 11.5],
          [0.033, 71.97, 7.1, 69.50, 7.2, 68.16, 15.6],
          [0.02, 73.10, 9.2, 70.79, 9.3, 68.73, 17.8],
          [0.0067, 74.15, 13.5, 72.18, 14.2, 69.08, 21.2],
          [0.0005, 74.74, 17.7, 72.81, 17.3, 69.15, 22.7],
        ],
        { mf1: 8.55, byStage: { W: 5.60, N1: 14.82, N2: 3.31, N3: 3.81, REM: 16.79 } },
        { mf1: 2.31, byStage: { W: 1.41, N1: 2.84, N2: 1.53, N3: 2.23, REM: 3.85 } }
      ),
      sources,
    },
    {
      id: 'shhs',
      name: 'SHHS',
      epochs: 324148,
      recordings: 329,
      subjects: 329,
      channel: 'C4-A1',
      stageShares: { W: 14.07, N1: 3.18, N2: 43.85, N3: 16.87, REM: 22.03 },
      system: {
        baseline: { name: 'FIRMBOUND', seconds: 24.75, mf1: 67.02 },
        firstExceeds: { seconds: 3.62, mf1: 67.93, eta: 0.13333 },
        gain: 11.4,
        gainPrecision: 1,
        byStage: { W: 3.65, N1: 18.25, N2: 3.61, N3: 3.92, REM: 13.20 },
      },
      matched: {
        gain: 3.92,
        peakGain: 2.68,
        byStage: { W: -0.18, N1: 15.11, N2: -0.37, N3: 5.28, REM: 1.44 },
      },
      atTenSeconds: 73.94,
      ablation: makeDatasetAblation(
        [
          [0.5, 54.90, 1.2, 50.23, 1.2, 51.24, 1.9],
          [0.233, 64.01, 2.3, 59.68, 2.1, 58.40, 4.5],
          [0.133, 67.93, 3.6, 64.95, 3.5, 62.88, 7.5],
          [0.067, 71.55, 5.9, 69.40, 6.1, 66.62, 12.4],
          [0.033, 73.58, 9.0, 71.64, 9.2, 68.05, 16.4],
          [0.02, 74.59, 11.7, 72.81, 11.6, 68.57, 19.0],
          [0.0067, 75.73, 17.1, 74.08, 15.9, 68.86, 22.2],
          [0.0005, 76.16, 21.1, 74.83, 19.8, 68.94, 24.3],
        ],
        { mf1: 9.00, byStage: { W: 3.04, N1: 20.09, N2: 5.77, N3: 3.98, REM: 10.12 } },
        { mf1: 1.97, byStage: { W: 0.72, N1: 6.41, N2: 1.11, N3: 0.99, REM: 1.27 } }
      ),
      sources,
    },
  ];

  const ablationSummary = {
    source: `${sources.ablationSweep}; ${sources.ablationStages}`,
    vsWoEndpointMemory: {
      mf1Range: [7.72, 9.53],
      n1Range: [10.80, 20.09],
      remRange: [10.12, 16.79],
      looseCostGapRange: [5.35, 7.22],
      tightCostGapRange: [3.41, 4.49],
    },
    vsWoPairedHistory: {
      mf1Range: [1.97, 6.58],
      looseCostGapRange: [1.33, 5.25],
      tightCostGapRange: [3.26, 9.69],
    },
  };

  const windowSensitivity = {
    source: sources.atTenSeconds,
    budgetSeconds: 10,
    lengths: [16, 32, 64],
    atL32: { dreams: 79.13, edf20: 78.12, edf78: 73.28, shhs: 73.94 },
    dreamsByLength: { l16: 77.65, l32: 79.13, l64: 75.95 },
    l16DropRangePp: [1.46, 1.64],
    l64DropRangePp: [1.90, 3.92],
    l16GainOverEctsRangePp: [9.66, 12.34],
    l64GainOverEctsRangePp: [7.28, 10.64],
    note: 'Same L = 32 trained weights evaluated without retraining at L ∈ {16, 32, 64} under a 10 s/epoch budget.',
  };

  const policyComparisonMeta = {
    source: sources.matched,
    comparators: ['Fixed', 'Threshold', 'AAGC', 'GP (α = 0)'],
    note: 'All policies share the frozen PAUSE backbone and calibration; Table III(c) reports matched-budget F1 gains (pp) over their upper envelope.',
    n3MainVsSuppNote: 'Sleep-EDF-20 N3 prints 2.92 in Manuscript Table III(c) and 2.93 in Supplement Table XXIV.',
  };

  const history = {
    source: sources.history,
    conditions: {
      A: 'Full-only training · Complete 30 s history',
      B: 'Full-only training · Policy-truncated history',
      C: 'Paired-history training · Complete 30 s history',
      D: 'Paired-history training · Policy-truncated history',
    },
    protocolNote: 'Frozen Full-only and Paired backbones evaluated under shared endpoint positions from the frozen PAUSE policy; scores use uncalibrated ML (argmax_c s_{n,k}(c)) averaged over all 30 prefixes and 20 folds (excluding window-start epoch n = 1).',
    gapNote: 'Full-only drop Δ_F = A − B (0.89–12.02 pp across all 8 costs, mean E[τ] ≈ 1.1–21 s) combines unavoidable missing-EEG loss with shift; I = (A − B) − (C − D) isolates the net reduction in history-switch sensitivity.',
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

  const fullObservation = {
    source: sources.fullObservation,
    note: 'Full 30-patch observation (stopper disabled). Equal-weight means ± SD (ddof=0) over 20 held-out folds.',
    caveatNote: 'PAUSE leads mean MF1 on 3/4 datasets (trails AttnSleep by 0.38 pp on Sleep-EDF-78) and mean ACC on 4/4, while trailing AttnSleep/LGSleepNet on MGm on Sleep-EDF-20/78. Between-fold SDs are sizable and N3 remains the backbone weakness (Supplement §VI-D, Table XXVII).',
    methods: ['PAUSE', 'AttnSleep', 'LGSleepNet', 'MSA-CNN', 'ULW-SleepNet'],
    byDataset: {
      dreams: [
        { method: 'PAUSE', acc: 84.79, accSd: 3.65, mf1: 78.18, mf1Sd: 6.34, mgm: 84.64, mgmSd: 5.17 },
        { method: 'AttnSleep', acc: 82.39, accSd: 3.70, mf1: 74.48, mf1Sd: 7.04, mgm: 82.86, mgmSd: 6.68 },
        { method: 'LGSleepNet', acc: 82.47, accSd: 4.13, mf1: 74.31, mf1Sd: 6.64, mgm: 82.63, mgmSd: 5.70 },
        { method: 'MSA-CNN', acc: 81.50, accSd: 6.22, mf1: 69.62, mf1Sd: 10.49, mgm: 76.93, mgmSd: 11.78 },
        { method: 'ULW-SleepNet', acc: 82.87, accSd: 3.28, mf1: 71.53, mf1Sd: 4.19, mgm: 79.05, mgmSd: 4.49 },
      ],
      edf20: [
        { method: 'PAUSE', acc: 85.58, accSd: 3.33, mf1: 78.84, mf1Sd: 3.85, mgm: 85.49, mgmSd: 3.24 },
        { method: 'AttnSleep', acc: 84.21, accSd: 4.64, mf1: 77.92, mf1Sd: 4.99, mgm: 86.20, mgmSd: 3.31 },
        { method: 'LGSleepNet', acc: 83.91, accSd: 4.92, mf1: 77.36, mf1Sd: 5.59, mgm: 85.59, mgmSd: 3.78 },
        { method: 'MSA-CNN', acc: 82.48, accSd: 5.82, mf1: 68.91, mf1Sd: 7.88, mgm: 75.37, mgmSd: 8.27 },
        { method: 'ULW-SleepNet', acc: 84.12, accSd: 3.37, mf1: 74.45, mf1Sd: 4.17, mgm: 81.74, mgmSd: 3.59 },
      ],
      edf78: [
        { method: 'PAUSE', acc: 79.73, accSd: 4.06, mf1: 73.52, mf1Sd: 5.07, mgm: 81.62, mgmSd: 3.95 },
        { method: 'AttnSleep', acc: 78.47, accSd: 3.90, mf1: 73.90, mf1Sd: 3.64, mgm: 82.77, mgmSd: 2.76 },
        { method: 'LGSleepNet', acc: 78.03, accSd: 4.06, mf1: 73.58, mf1Sd: 3.72, mgm: 82.82, mgmSd: 2.64 },
        { method: 'MSA-CNN', acc: 78.23, accSd: 4.24, mf1: 71.82, mf1Sd: 4.03, mgm: 80.63, mgmSd: 3.05 },
        { method: 'ULW-SleepNet', acc: 76.81, accSd: 4.50, mf1: 70.07, mf1Sd: 4.74, mgm: 79.22, mgmSd: 3.73 },
      ],
      shhs: [
        { method: 'PAUSE', acc: 83.61, accSd: 1.79, mf1: 75.88, mf1Sd: 2.09, mgm: 83.30, mgmSd: 1.41 },
        { method: 'AttnSleep', acc: 81.73, accSd: 1.92, mf1: 72.94, mf1Sd: 2.18, mgm: 82.87, mgmSd: 1.49 },
        { method: 'LGSleepNet', acc: 82.53, accSd: 1.94, mf1: 73.20, mf1Sd: 2.49, mgm: 82.60, mgmSd: 2.17 },
        { method: 'MSA-CNN', acc: 82.72, accSd: 2.26, mf1: 66.78, mf1Sd: 2.00, mgm: 71.50, mgmSd: 1.35 },
        { method: 'ULW-SleepNet', acc: 78.10, accSd: 2.58, mf1: 63.21, mf1Sd: 2.51, mgm: 69.33, mgmSd: 2.22 },
      ],
    },
  };

  const classWeight = {
    source: sources.classWeight,
    dataset: 'DREAMS-SUB',
    datasetId: 'dreams',
    patchRange: [7.00, 16.63],
    defaultAlpha: 0.5,
    gainOverZero: 2.09,
    costFormulaNote: 'Misclassification cost Λ_{c,a} = ρ w_c I[c ≠ a] with ρ = 10; α = 0 yields calibrated MAP and α = 0.5 applies tempered inverse-prevalence weighting.',
    values: [
      { alpha: 0, mf1: 77.15 },
      { alpha: 0.25, mf1: 78.74 },
      { alpha: 0.5, mf1: 79.24 },
      { alpha: 0.75, mf1: 79.14 },
      { alpha: 1, mf1: 78.43 },
    ],
  };

  const RETAIN_HIGH_POSITIVE_COSTS = '7 of 8 acquisition costs on both Sleep-EDF-78 and SHHS';

  const spectralChange = {
    source: sources.spectralChange,
    eta: 0.06667,
    metric: 'Next-epoch 30-prefix mean accuracy gain (pp)',
    changeScoreFormula: 'D_{m,j} = ||v_{m,j} − v_{m,j−1}||_2 / sqrt(128) on training-normalized 128-bin log-PSD patches',
    retainDefinition: 'τ_m − 1 → τ_m (effect of retaining the last observed second)',
    completeDefinition: 'τ_m → 30 (effect of restoring unacquired suffix when D_{m,τ_m+1} ≥ q90)',
    unroundedNote: 'High − Low contrasts use unrounded estimates (Manuscript Table VI).',
    thresholds: { high: 'D >= q90', low: 'D <= q50' },
    retainLastObservedSecond: [
      { id: 'edf78', dataset: 'Sleep-EDF-78', matchedPairs: 3870, high: 0.55, low: 0.12, highMinusLow: 0.42 },
      { id: 'shhs', dataset: 'SHHS', matchedPairs: 8598, high: 0.95, low: 0.25, highMinusLow: 0.70 },
    ],
    completePrecedingEpoch: [
      { id: 'edf78', dataset: 'Sleep-EDF-78', matchedPairs: 4771, high: 1.94, low: 0.78, highMinusLow: 1.15 },
      { id: 'shhs', dataset: 'SHHS', matchedPairs: 9480, high: 1.40, low: 0.78, highMinusLow: 0.61 },
    ],
    summary: {
      retainPositiveCosts: RETAIN_HIGH_POSITIVE_COSTS,
      retainHighPositiveCosts: RETAIN_HIGH_POSITIVE_COSTS,
      completeHighMinusLowPositiveCosts: '8 of 8 acquisition costs on both Sleep-EDF-78 and SHHS',
    },
  };

  const onlineMass = {
    source: sources.onlineMass,
    massMeanDelaySeconds: 495,
    massDelayRangeSeconds: [30, 960],
    ceilings: [3.0, 11.72, 21.0],
    delayRatioRange: [23.6, 165.0],
    moderateToHighDelayRatioRange: [23.6, 42.2],
    moderateToHighDeltaWithinPp: 0.70,
    moderateToHighMeanDiffPp: 0.35,
    moderateToHighDelayRangeSeconds: [11.72, 21.00],
    lowBudgetDeltaRangePp: [-5.77, -4.13],
    effectiveCeilingNote: 'Budget ceilings (3.0 / 11.72 / 21.0 s) correspond to 10% / 40% / 72% nominal MASS integrity after integer epoch/patch masking over 32×30 tokens (Supplement Table XXVIII).',
    byDataset: {
      dreams: [
        { ceiling: 3.0, nominalIntegrityPct: 10, mass: { mf1: 75.58, delay: 495 }, pause: { mf1: 70.68, delay: 3.00 }, deltaMf1: -4.90, delayRatio: 165.0 },
        { ceiling: 11.72, nominalIntegrityPct: 40, mass: { mf1: 80.02, delay: 495 }, pause: { mf1: 79.58, delay: 11.72 }, deltaMf1: -0.44, delayRatio: 42.2 },
        { ceiling: 21.0, nominalIntegrityPct: 72, mass: { mf1: 80.71, delay: 495 }, pause: { mf1: 80.63, delay: 17.01 }, deltaMf1: -0.08, delayRatio: 29.1 },
      ],
      edf20: [
        { ceiling: 3.0, nominalIntegrityPct: 10, mass: { mf1: 76.62, delay: 495 }, pause: { mf1: 71.73, delay: 3.00 }, deltaMf1: -4.89, delayRatio: 165.0 },
        { ceiling: 11.72, nominalIntegrityPct: 40, mass: { mf1: 79.21, delay: 495 }, pause: { mf1: 78.60, delay: 11.72 }, deltaMf1: -0.61, delayRatio: 42.2 },
        { ceiling: 21.0, nominalIntegrityPct: 72, mass: { mf1: 79.26, delay: 495 }, pause: { mf1: 79.56, delay: 15.92 }, deltaMf1: 0.30, delayRatio: 31.1 },
      ],
      edf78: [
        { ceiling: 3.0, nominalIntegrityPct: 10, mass: { mf1: 71.61, delay: 495 }, pause: { mf1: 65.84, delay: 3.00 }, deltaMf1: -5.77, delayRatio: 165.0 },
        { ceiling: 11.72, nominalIntegrityPct: 40, mass: { mf1: 74.42, delay: 495 }, pause: { mf1: 73.72, delay: 11.72 }, deltaMf1: -0.70, delayRatio: 42.2 },
        { ceiling: 21.0, nominalIntegrityPct: 72, mass: { mf1: 74.95, delay: 495 }, pause: { mf1: 74.74, delay: 17.68 }, deltaMf1: -0.21, delayRatio: 28.0 },
      ],
      shhs: [
        { ceiling: 3.0, nominalIntegrityPct: 10, mass: { mf1: 70.25, delay: 495 }, pause: { mf1: 66.12, delay: 3.00 }, deltaMf1: -4.13, delayRatio: 165.0 },
        { ceiling: 11.72, nominalIntegrityPct: 40, mass: { mf1: 75.18, delay: 495 }, pause: { mf1: 74.59, delay: 11.72 }, deltaMf1: -0.59, delayRatio: 42.2 },
        { ceiling: 21.0, nominalIntegrityPct: 72, mass: { mf1: 76.58, delay: 495 }, pause: { mf1: 76.15, delay: 21.00 }, deltaMf1: -0.43, delayRatio: 23.6 },
      ],
    },
  };

  const deployment = {
    source: sources.deployment,
    afePower: {
      overallSavingsPctRange: [31.0, 73.4],
      standbyShareOfPausePowerPctRange: [54.5, 84.5],
      wakeRecoveryOverheadMwRange: [0.002, 0.11],
      amplifiers: [
        {
          id: 'I',
          name: 'ADS1299-4',
          activeMw: 22.0,
          standbyMw: 5.1,
          savingsPctRange: [45.2, 67.5],
          pauseMwByDataset: { dreams: 7.88, edf20: 7.32, edf78: 7.52, shhs: 7.14 },
        },
        {
          id: 'II',
          name: 'ADS131A04',
          activeMw: 15.8,
          standbyMw: 2.6,
          savingsPctRange: [52.3, 73.4],
          pauseMwByDataset: { dreams: 4.77, edf20: 4.34, edf78: 4.49, shhs: 4.19 },
        },
        {
          id: 'III',
          name: 'ADS1294',
          activeMw: 10.1,
          standbyMw: 4.0,
          savingsPctRange: [31.0, 53.1],
          pauseMwByDataset: { dreams: 5.00, edf20: 4.80, edf78: 4.87, shhs: 4.74 },
        },
      ],
      // Estimated AFE power (mW) at each system's Table VII comparison point.
      table7ByDataset: {
        dreams: { EARLIEST: [21.87, 15.70, 10.05], 'ECONOMY-γ-Max': [21.33, 15.28, 9.86], EarlyMix: [17.91, 12.60, 8.62], FIRMBOUND: [18.51, 13.08, 8.84], PAUSE: [7.88, 4.77, 5.00] },
        edf20: { EARLIEST: [21.98, 15.79, 10.09], 'ECONOMY-γ-Max': [21.71, 15.58, 10.00], EarlyMix: [15.12, 10.42, 7.61], FIRMBOUND: [18.44, 13.02, 8.81], PAUSE: [7.32, 4.34, 4.80] },
        edf78: { EARLIEST: [21.65, 15.53, 9.97], 'ECONOMY-γ-Max': [21.83, 15.66, 10.04], EarlyMix: [16.82, 11.76, 8.23], FIRMBOUND: [18.96, 13.43, 9.00], PAUSE: [7.52, 4.49, 4.87] },
        shhs: { EARLIEST: [21.50, 15.41, 9.92], 'ECONOMY-γ-Max': [21.97, 15.78, 10.09], EarlyMix: [13.04, 8.80, 6.87], FIRMBOUND: [19.04, 13.49, 9.03], PAUSE: [7.14, 4.19, 4.74] },
      },
    },
    footprint: {
      paramsMillions: 2.0,
      weightsMiB: 7.6,
      kvCacheKiB: 434,
      macsPerPatchM: 2.09,
      flopsPerPatchM: 4.19,
      gpStopper: {
        regressorsCount: 29,
        kMacsPerDecision: 4.42,
        backboneMacSharePct: 0.21,
        memoryMB: 0.28,
        backToBackLatencyOverheadPctRange: [1.1, 1.3],
      },
    },
    latency: {
      host: 'MediaTek Dimensity 9500',
      runtime: 'ONNX Runtime 1.28, single CPU thread, FP32, thermal steady state',
      backToBack: [
        { core: 'C1-Ultra', ghz: 2.75, medianMs: 2.14, p90Ms: 2.40, p99Ms: 6.96, stopperMs: 0.028 },
        { core: 'C1-Premium', ghz: 2.60, medianMs: 2.46, p90Ms: 2.83, p99Ms: 3.39, stopperMs: null },
        { core: 'C1-Pro', ghz: 2.30, medianMs: 3.32, p90Ms: 3.99, p99Ms: 6.32, stopperMs: 0.036 },
      ],
      oneHzCadence: [
        { core: 'Scheduler', ghz: 1.56, medianMs: 9.44, p90Ms: 12.56, p99Ms: 18.80, stopperMs: null },
        { core: 'C1-Pro', ghz: 1.72, medianMs: 8.97, p90Ms: 12.43, p99Ms: 13.66, stopperMs: null },
      ],
      schedulerMedianMs: 9.44,
      schedulerP99Ms: 18.80,
      medianSpeedupVsOneSecondInterval: 105,
      dutyCycle: {
        patchesPerEpoch: 8,
        computeMsPerEpoch: 76,
        epochSeconds: 30,
        dutyCyclePct: 0.25,
      },
    },
  };

  const risk = {
    image: 'risk-trajectories.png',
    source: sources.risk,
    eta: 0.02,
    dataset: 'DREAMS-SUB',
    stages: ['W', 'N1', 'N2', 'N3', 'REM'],
    caption: 'Illustrative DREAMS-SUB risk trajectories at η = 0.02. Panel titles are reference stages. Orange solid: stop risk. Green dashed: fitted continuation risk. The dot marks the first prefix where stop risk is no greater than continuation risk. The plotted suffix after each dot is an offline counterfactual and is unavailable to the online policy.',
  };

  window.PAUSE_EVIDENCE = {
    datasets,
    history,
    sources,
    risk,
    ablationSummary,
    windowSensitivity,
    policyComparisonMeta,
    fullObservation,
    classWeight,
    spectralChange,
    onlineMass,
    deployment,
  };
})();
