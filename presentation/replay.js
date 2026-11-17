'use strict';

(() => {
  const get = id => document.getElementById(id);
  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const formatSigned = (value, digits = 2) => `${value > 0 ? '+' : ''}${value.toFixed(digits)}`;
  const round6 = value => Math.round(value * 1e6) / 1e6;

  // ---------------------------------------------------------------------------
  // 1. Opening 4-Page Story Deck (#background)
  // ---------------------------------------------------------------------------
  const background = get('background');
  const deckTabsWrap = get('deck-tabs');
  const deckPrev = get('deck-prev');
  const deckNext = get('deck-next');
  const deckStatus = get('deck-status');

  if (background && deckTabsWrap) {
    const deckTabs = Array.from(deckTabsWrap.querySelectorAll('[data-deck-page]'));
    const deckPages = Array.from(background.querySelectorAll('.deck-page'));
    const totalPages = deckPages.length || 4;
    let activeDeckPage = 1;

    const renderDeck = () => {
      deckTabs.forEach(tab => {
        const pageNum = Number(tab.dataset.deckPage);
        const selected = pageNum === activeDeckPage;
        tab.setAttribute('aria-selected', String(selected));
        tab.tabIndex = selected ? 0 : -1;
      });
      deckPages.forEach(page => {
        const pageNum = Number(page.dataset.page);
        page.hidden = pageNum !== activeDeckPage;
      });
      if (deckPrev) deckPrev.disabled = activeDeckPage <= 1;
      if (deckNext) deckNext.disabled = activeDeckPage >= totalPages;
      if (deckStatus) deckStatus.textContent = `Page ${activeDeckPage} of ${totalPages}`;
    };

    const setDeckPage = pageNum => {
      activeDeckPage = Math.min(totalPages, Math.max(1, pageNum));
      renderDeck();
    };

    deckTabs.forEach((tab, index) => {
      tab.addEventListener('click', () => {
        setDeckPage(Number(tab.dataset.deckPage));
      });
      tab.addEventListener('keydown', event => {
        let nextIndex = null;
        if (event.key === 'ArrowRight') nextIndex = (index + 1) % deckTabs.length;
        else if (event.key === 'ArrowLeft') nextIndex = (index - 1 + deckTabs.length) % deckTabs.length;
        else if (event.key === 'Home') nextIndex = 0;
        else if (event.key === 'End') nextIndex = deckTabs.length - 1;
        if (nextIndex === null) return;
        event.preventDefault();
        deckTabs[nextIndex].focus();
        setDeckPage(Number(deckTabs[nextIndex].dataset.deckPage));
      });
    });

    if (deckPrev) {
      deckPrev.addEventListener('click', () => {
        if (activeDeckPage > 1) setDeckPage(activeDeckPage - 1);
      });
    }
    if (deckNext) {
      deckNext.addEventListener('click', () => {
        if (activeDeckPage < totalPages) setDeckPage(activeDeckPage + 1);
      });
    }
    renderDeck();
  }

  // ---------------------------------------------------------------------------
  // 2. P1 Upgraded 2-Epoch Interactive Replay (#replay)
  //    Illustrative schematic dynamics (NOT real PSG recording values).
  // ---------------------------------------------------------------------------
  const horizon = 30;
  const traces = {
    higher: {
      cost: 0.5,
      'with-memory': {
        epochs: [
          {
            label: 'N2',
            stop: [6.20, 4.95, 3.65, 2.75],
            next: [3.10, 2.95, 2.75, 2.55],
            pred: ['W', 'N2', 'N2', 'N2'],
          },
          {
            label: 'N1',
            stop: [5.45, 4.80, 4.35, 3.90, 3.55, 3.10],
            next: [3.25, 3.15, 3.05, 2.95, 2.85, 2.78],
            pred: ['N2', 'N1', 'N1', 'N1', 'N1', 'N1'],
          },
        ],
      },
      'no-memory': {
        epochs: [
          {
            label: 'N2',
            stop: [6.20, 4.95, 3.65, 2.75],
            next: [3.10, 2.95, 2.75, 2.55],
            pred: ['W', 'N2', 'N2', 'N2'],
          },
          {
            label: 'N1',
            stop: [6.35, 6.20, 5.95, 5.75, 5.50, 5.20, 4.95, 4.70, 4.45, 4.25, 4.10, 3.95, 3.88, 3.80],
            next: [3.85, 3.80, 3.75, 3.70, 3.64, 3.58, 3.52, 3.48, 3.44, 3.40, 3.36, 3.34, 3.32, 3.35],
            pred: ['W', 'W', 'REM', 'W', 'REM', 'REM', 'W', 'REM', 'N1', 'REM', 'W', 'REM', 'N1', 'N1'],
          },
        ],
      },
    },
    lower: {
      cost: 0.02,
      'with-memory': {
        epochs: [
          {
            label: 'N2',
            stop: [6.20, 4.95, 3.65, 2.75, 2.40, 2.15, 1.95, 1.65],
            next: [2.18, 2.25, 2.16, 2.34, 2.12, 1.96, 1.84, 1.78],
            pred: ['W', 'N2', 'N2', 'N2', 'N2', 'N2', 'N2', 'N2'],
          },
          {
            label: 'N1',
            stop: [5.25, 4.65, 4.15, 3.70, 3.30, 2.95, 2.78, 2.45],
            next: [2.40, 2.48, 2.55, 2.60, 2.64, 2.68, 2.66, 2.58],
            pred: ['N2', 'N1', 'N1', 'N1', 'N1', 'N1', 'N1', 'N1'],
          },
        ],
      },
      'no-memory': {
        epochs: [
          {
            label: 'N2',
            stop: [6.20, 4.95, 3.65, 2.75, 2.40, 2.15, 1.95, 1.65],
            next: [2.18, 2.25, 2.16, 2.34, 2.12, 1.96, 1.84, 1.78],
            pred: ['W', 'N2', 'N2', 'N2', 'N2', 'N2', 'N2', 'N2'],
          },
          {
            label: 'N1',
            stop: [6.35, 6.20, 5.95, 5.75, 5.50, 5.20, 4.95, 4.70, 4.45, 4.25, 4.10, 3.95, 3.88, 3.80, 3.74, 3.68, 3.62, 3.56, 3.48, 3.32],
            next: [2.28, 2.42, 2.56, 2.70, 2.82, 2.90, 2.95, 3.00, 3.06, 3.12, 3.18, 3.24, 3.30, 3.42, 3.48, 3.50, 3.46, 3.42, 3.40, 3.36],
            pred: ['W', 'W', 'REM', 'W', 'REM', 'REM', 'W', 'REM', 'N1', 'REM', 'W', 'REM', 'N1', 'REM', 'N1', 'N1', 'N1', 'N1', 'N1', 'N1'],
          },
        ],
      },
    },
  };

  const replay = get('replay');
  const scenario = get('scenario');
  const historyModeSelect = get('history-mode');
  const chart = get('risk-chart');
  const stepButton = get('replay-step');
  const backButton = get('replay-back');
  const playButton = get('replay-play');
  const resetButton = get('replay-reset');
  const epochList = get('epochs');
  const decisionEl = get('decision');
  const cursorLabel = get('cursor-label');
  const epochTabAuto = get('epoch-tab-auto');
  const epochTab1 = get('epoch-tab-1');
  const epochTab2 = get('epoch-tab-2');
  const svgNs = 'http://www.w3.org/2000/svg';

  if (replay && scenario && chart && stepButton && playButton && resetButton && epochList) {
    let rows = [];
    let undoStack = [];
    let timer = null;
    let active = 0;
    let count = 0;
    let inspectMode = 'auto';

    const getMode = () => (historyModeSelect ? historyModeSelect.value : 'with-memory');
    const getViewIndex = () => (inspectMode === 'auto' ? active : inspectMode);

    const findStopPrefix = (trace, cost) => {
      for (let i = 0; i < trace.stop.length; i++) {
        if (round6(trace.stop[i]) <= round6(trace.next[i] + cost)) return i + 1;
      }
      return horizon;
    };

    const setState = state => {
      const focused = document.activeElement;
      replay.dataset.state = state;
      playButton.textContent = state === 'playing' ? 'Pause' : state === 'paused' ? 'Resume' : 'Play';
      playButton.disabled = state === 'complete';
      stepButton.disabled = state === 'complete' || state === 'playing';
      if (backButton) backButton.disabled = count === 0 || state === 'playing';
      if (state === 'complete' && (focused === playButton || focused === stepButton)) resetButton.focus();
    };

    const svgNode = (tag, attrs, content) => {
      const element = document.createElementNS(svgNs, tag);
      for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
      if (content !== undefined) element.textContent = content;
      chart.append(element);
      return element;
    };

    const renderChart = (row, cost) => {
      chart.replaceChildren();
      const x = k => 39 + (k - 1) * 12.45;
      const y = value => 133 - value * 15.5;
      for (const value of [0, 2, 4, 6]) {
        svgNode('line', { x1: 39, x2: 400, y1: y(value), y2: y(value), stroke: 'var(--plot-grid)' });
        svgNode('text', { x: 30, y: y(value) + 4, 'text-anchor': 'end', fill: 'var(--dim)', 'font-size': 11 }, String(value));
      }
      for (const k of [1, 5, 10, 15, 20, 25, 30]) {
        svgNode('text', { x: x(k), y: 151, 'text-anchor': 'middle', fill: 'var(--dim)', 'font-size': 11 }, String(k));
      }
      svgNode('text', { x: 400, y: 164, 'text-anchor': 'end', fill: 'var(--dim)', 'font-size': 11 }, 'ACQUIRED PREFIX (s)');

      if (!row.acquired) {
        svgNode('text', {
          x: 220,
          y: 84,
          'text-anchor': 'middle',
          fill: 'var(--dim)',
          'font-size': 12,
          class: 'chart-prompt',
        }, 'Acquire 1 s or Play to trace risk');
        return;
      }

      const stoppedEarly = row.done && row.acquired < horizon;
      if (stoppedEarly) {
        const stopX = x(row.acquired);
        svgNode('rect', {
          x: stopX + 2,
          y: 22,
          width: Math.max(0, 398 - stopX),
          height: 111,
          fill: 'var(--plot-suffix)',
        });
        if (400 - stopX >= 96) {
          svgNode('text', {
            x: 396,
            y: 37,
            'text-anchor': 'end',
            fill: 'var(--dim)',
            'font-size': 10.5,
          }, 'SUFFIX UNOBSERVED');
        }
        svgNode('line', {
          x1: stopX,
          x2: stopX,
          y1: 22,
          y2: 133,
          stroke: 'var(--endpoint)',
          'stroke-width': 1.5,
          'stroke-dasharray': '3 3',
        });
      }

      const series = [
        [row.trace.next, cost, 'var(--continue)', '5 4'],
        [row.trace.stop, 0, 'var(--endpoint)', 'none'],
      ];
      for (const [values, add, stroke, dash] of series) {
        const points = values.slice(0, row.acquired).map((value, index) => `${x(index + 1)},${y(value + add)}`);
        svgNode('polyline', {
          points: points.join(' '),
          fill: 'none',
          stroke,
          'stroke-width': 2,
          'stroke-dasharray': dash,
        });
        svgNode('circle', {
          cx: x(row.acquired),
          cy: y(values[row.acquired - 1] + add),
          r: 4,
          fill: stroke,
        });
      }

      if (stoppedEarly) {
        const stopX = x(row.acquired);
        const stopY = y(row.trace.stop[row.acquired - 1]);
        svgNode('circle', {
          cx: stopX,
          cy: stopY,
          r: 7.5,
          fill: 'none',
          stroke: 'var(--endpoint)',
          'stroke-width': 1.75,
        });
        const labelX = Math.min(Math.max(stopX, 82), 354);
        svgNode('text', {
          x: labelX,
          y: 15,
          'text-anchor': 'middle',
          fill: 'var(--endpoint)',
          'font-size': 10.5,
          'font-weight': 700,
        }, `STOP AT k = ${row.acquired}`);
      }
    };

    const renderEpochTabs = () => {
      if (epochTabAuto) epochTabAuto.setAttribute('aria-pressed', String(inspectMode === 'auto'));
      if (epochTab1) epochTab1.setAttribute('aria-pressed', String(inspectMode === 0));
      if (epochTab2) epochTab2.setAttribute('aria-pressed', String(inspectMode === 1));
    };

    const setInspectMode = mode => {
      inspectMode = mode;
      render();
    };

    function render() {
      const mode = getMode();
      const scenarioKey = traces[scenario.value] ? scenario.value : 'higher';
      const cost = traces[scenarioKey].cost;
      const viewIndex = getViewIndex();
      const viewed = rows[viewIndex];

      replay.dataset.step = String(count);
      replay.dataset.historyMode = mode;
      replay.dataset.inspect = String(inspectMode);
      if (backButton) backButton.disabled = count === 0 || replay.dataset.state === 'playing';
      renderEpochTabs();

      rows.forEach((row, index) => {
        row.element.classList.toggle('is-inspected', index === viewIndex);
        row.element.classList.toggle('is-active-epoch', index === active && !row.done && count > 0);
        row.element.classList.toggle('is-no-memory', index === 1 && mode === 'no-memory');
        const curPred = row.acquired && row.trace.pred ? row.trace.pred[row.acquired - 1] : null;
        row.status.textContent = row.done
          ? `Stop at ${row.acquired} s · output ${row.trace.label}`
          : row.acquired
            ? `${row.acquired} s acquired${curPred ? ` · ŷ = ${curPred}` : ''}`
            : 'Waiting';
        if (index === 0) {
          row.heading.textContent = 'Epoch 1 · no earlier history';
        } else if (mode === 'no-memory') {
          row.heading.textContent = rows[0].done
            ? 'Epoch 2 · no endpoint memory (counterfactual)'
            : 'Epoch 2 · endpoint memory omitted';
        } else {
          row.heading.textContent = rows[0].done
            ? `Epoch 2 · carries endpoint at ${rows[0].acquired} s`
            : 'Epoch 2 · awaits endpoint';
        }
        row.patches.forEach((patch, patchIndex) => {
          const k = patchIndex + 1;
          patch.className = row.done && k === row.acquired
            ? 'endpoint'
            : k <= row.acquired
              ? 'acquired'
              : row.done
                ? 'skipped'
                : '';
        });
      });

      if (cursorLabel) {
        if (!viewed.acquired) {
          cursorLabel.textContent = inspectMode === 'auto' ? 'READY' : `EPOCH ${viewIndex + 1} · WAITING`;
        } else {
          cursorLabel.textContent = `EPOCH ${viewIndex + 1} / k = ${viewed.acquired}${viewed.done ? ' · STOP' : ''}`;
        }
      }

      renderChart(viewed, cost);

      if (decisionEl) {
        const isCounterfactualActive = mode === 'no-memory' && viewIndex === 1 && viewed.acquired > 0;
        decisionEl.classList.toggle('is-counterfactual', isCounterfactualActive);
        if (!count) {
          decisionEl.textContent = mode === 'no-memory'
            ? 'Counterfactual mode (no endpoint memory): acquire the first patch to compare Epoch 2.'
            : 'Ready. Acquire the first patch to begin.';
          chart.setAttribute('aria-label', 'Risk chart, waiting for the first patch');
          return;
        }
        if (!viewed.acquired) {
          decisionEl.textContent = `Viewing Epoch ${viewIndex + 1} (${viewed.trace.label}): 0 s acquired yet (simulation at Epoch ${active + 1}, k = ${rows[active].acquired}).`;
          chart.setAttribute('aria-label', `Epoch ${viewIndex + 1}, waiting for the first patch`);
          return;
        }

        const stop = viewed.trace.stop[viewed.acquired - 1];
        const next = viewed.trace.next[viewed.acquired - 1] + cost;
        const predStage = viewed.trace.pred ? viewed.trace.pred[viewed.acquired - 1] : viewed.trace.label;
        const comparison = `Epoch ${viewIndex + 1} (${viewed.trace.label}) at k = ${viewed.acquired} s: Stop ${stop.toFixed(2)} ${round6(stop) <= round6(next) ? '≤' : '>'} continue ${next.toFixed(2)}.`;
        const stagePill = el('span', 'decision-stage', `ŷ = ${predStage}`);
        let outcome = '';
        if (viewed.done) {
          if (viewIndex === 1 && mode === 'no-memory') {
            const withMemStop = findStopPrefix(traces[scenarioKey]['with-memory'].epochs[1], cost);
            const extra = viewed.acquired - withMemStop;
            outcome = `Stop and emit ${viewed.trace.label} (+${extra} s vs. ${withMemStop} s with memory). ${horizon - viewed.acquired} s skipped.`;
          } else if (viewIndex === 0 && mode === 'with-memory') {
            outcome = `Stop and emit ${viewed.trace.label}; carry k = ${viewed.acquired} endpoint to Epoch 2. ${horizon - viewed.acquired} s skipped.`;
          } else if (viewIndex === 0 && mode === 'no-memory') {
            outcome = `Stop and emit ${viewed.trace.label} (endpoint dropped in counterfactual mode). ${horizon - viewed.acquired} s skipped.`;
          } else {
            outcome = `Stop and emit ${viewed.trace.label}. ${horizon - viewed.acquired} s skipped in this epoch.`;
          }
        } else if (viewIndex === 1 && mode === 'no-memory') {
          outcome = 'Continue: without Epoch 1 endpoint memory, N1 stays ambiguous at early prefixes.';
        } else {
          outcome = 'Continue: acquire the next patch.';
        }
        decisionEl.replaceChildren(
          document.createTextNode(`${comparison} `),
          stagePill,
          document.createTextNode(` ${outcome}`)
        );
        chart.setAttribute('aria-label', `Epoch ${viewIndex + 1}, ${viewed.acquired} second prefix, pred ${predStage}. ${comparison}`);
      }
    }

    const stopTimer = () => {
      if (timer !== null) window.clearInterval(timer);
      timer = null;
    };

    const step = () => {
      if (replay.dataset.state === 'complete') return;
      undoStack.push({
        active,
        count,
        rows: rows.map(row => ({ acquired: row.acquired, done: row.done })),
      });
      if (rows[active].done) active++;
      const row = rows[active];
      const k = ++row.acquired;
      count++;
      const scenarioKey = traces[scenario.value] ? scenario.value : 'higher';
      const cost = traces[scenarioKey].cost;
      row.done = k === horizon || round6(row.trace.stop[k - 1]) <= round6(row.trace.next[k - 1] + cost);
      if (row.done && active === rows.length - 1) {
        stopTimer();
        setState('complete');
      } else if (replay.dataset.state === 'ready') {
        setState('paused');
      }
      render();
    };

    const stepBack = () => {
      if (undoStack.length === 0 || replay.dataset.state === 'playing') return;
      stopTimer();
      const prev = undoStack.pop();
      active = prev.active;
      count = prev.count;
      prev.rows.forEach((saved, index) => {
        rows[index].acquired = saved.acquired;
        rows[index].done = saved.done;
      });
      setState(count === 0 ? 'ready' : 'paused');
      render();
    };

    const reset = () => {
      stopTimer();
      active = 0;
      count = 0;
      undoStack = [];
      inspectMode = 'auto';
      epochList.replaceChildren();
      const mode = getMode();
      const scenarioKey = traces[scenario.value] ? scenario.value : 'higher';
      rows = traces[scenarioKey][mode].epochs.map((trace, index) => {
        const element = el('div', 'epoch');
        element.dataset.epochIndex = String(index);
        element.title = `Click to inspect Epoch ${index + 1} risk trajectory`;
        element.addEventListener('click', () => setInspectMode(index));

        const header = el('div', 'epoch-heading');
        const heading = el('b');
        const status = el('span');
        header.append(heading, status);

        const rail = el('div', 'patches');
        rail.setAttribute('aria-hidden', 'true');
        const patches = Array.from({ length: horizon }, () => document.createElement('i'));
        rail.append(...patches);

        element.append(header, rail);
        epochList.append(element);
        return { element, trace, acquired: 0, done: false, heading, status, patches };
      });
      setState('ready');
      render();
    };

    stepButton.addEventListener('click', step);
    if (backButton) backButton.addEventListener('click', stepBack);
    playButton.addEventListener('click', () => {
      if (timer !== null) {
        stopTimer();
        setState('paused');
      } else {
        setState('playing');
        step();
        if (replay.dataset.state !== 'complete') timer = window.setInterval(step, 440);
      }
    });
    resetButton.addEventListener('click', reset);
    scenario.addEventListener('change', reset);
    if (historyModeSelect) historyModeSelect.addEventListener('change', reset);
    if (epochTabAuto) epochTabAuto.addEventListener('click', () => setInspectMode('auto'));
    if (epochTab1) epochTab1.addEventListener('click', () => setInspectMode(0));
    if (epochTab2) epochTab2.addEventListener('click', () => setInspectMode(1));
    reset();
  }

  // ---------------------------------------------------------------------------
  // 3. P0 Upgraded Evidence Explorer (#evidence)
  // ---------------------------------------------------------------------------
  const EVIDENCE = window.PAUSE_EVIDENCE;
  const dataset = get('dataset');
  const datasetMeta = get('dataset-meta');
  const results = get('results');
  const systemSource = get('system-source');
  const evTablist = get('evidence-tablist');

  if (!EVIDENCE || !dataset || !results) return;

  const STAGES = ['W', 'N1', 'N2', 'N3', 'REM'];
  const HIGHLIGHT_STAGES = new Set(['N1', 'REM']);
  const ETA_LABELS = {
    0.0005: '0.0005',
    0.006667: '0.0067',
    0.02: '0.02',
    0.03333: '0.033',
    0.06667: '0.067',
    0.13333: '0.133',
    0.23333: '0.233',
    0.5: '0.5',
  };
  const data = EVIDENCE.datasets;
  for (const row of data) dataset.add(new Option(row.name, row.id));

  const bar = (label, seconds, mf1) => {
    const item = el('div', 'bar-row');
    const name = el('span', 'bar-name', label);
    const track = el('div', 'bar-track');
    track.setAttribute('aria-hidden', 'true');
    const fill = el('div', 'bar-fill');
    fill.style.width = `${(seconds / horizon) * 100}%`;
    track.append(fill);
    const value = el('span', 'bar-value');
    const duration = el('strong', null, `${seconds.toFixed(2)} s`);
    const score = el('span', null, `${mf1.toFixed(2)}% MF1`);
    value.append(duration, score);
    item.append(name, track, value);
    return item;
  };

  const renderStageBreakdown = (stageValues, options = {}) => {
    const { maxScale = 22, signed = false, stageShares = null, highlightMode = 'minorities' } = options;
    const highlightSet = highlightMode === 'top2'
      ? new Set(STAGES.slice().sort((a, b) => stageValues[b] - stageValues[a]).slice(0, 2))
      : HIGHLIGHT_STAGES;
    const grid = el('div', `stage-grid${signed ? ' is-signed' : ''}`);
    for (const stage of STAGES) {
      const val = stageValues[stage];
      const highlight = highlightSet.has(stage);
      const item = el('div', `stage-item${highlight ? ' is-highlight' : ''}${val < 0 ? ' is-negative' : ''}`);
      const head = el('div', 'stage-head');
      head.append(el('span', 'stage-name', stage));
      if (stageShares && stageShares[stage] !== undefined) {
        head.append(el('span', 'stage-share', `${stageShares[stage].toFixed(1)}%`));
      }
      const track = el('div', 'stage-track');
      track.setAttribute('aria-hidden', 'true');
      const fill = el('div', 'stage-fill');
      const pct = Math.min(100, Math.max(0, (Math.abs(val) / maxScale) * 100));
      fill.style.width = `${pct.toFixed(1)}%`;
      track.append(fill);
      const valText = `${formatSigned(val, 2)} pp`;
      const valEl = el('span', 'stage-gain', valText);
      item.append(head, track, valEl);
      grid.append(item);
    }
    return grid;
  };

  const renderPanel1 = row => {
    if (datasetMeta) {
      datasetMeta.textContent = `${row.epochs.toLocaleString('en-US')} epochs · ${row.recordings} recordings · ${row.channel} · 20-fold`;
    }

    const { baseline, firstExceeds, gain, gainPrecision, byStage } = row.system;
    const bars = el('div', 'bars');
    const scale = el('div', 'bar-scale');
    scale.setAttribute('aria-hidden', 'true');
    for (const tick of ['0', '15', '30 s']) scale.append(el('span', null, tick));
    bars.append(
      bar(`PAUSE (η = ${ETA_LABELS[firstExceeds.eta] || firstExceeds.eta})`, firstExceeds.seconds, firstExceeds.mf1),
      bar(`${baseline.name} at its peak`, baseline.seconds, baseline.mf1),
      scale
    );

    const note = el('div', 'data-note');
    const ratio = (baseline.seconds / firstExceeds.seconds).toFixed(1);
    note.append(
      el('p', 'metric', `+${gain.toFixed(gainPrecision)} pp`),
      el('p', null, `Average matched-budget macro-F1 gain on ${row.name} over the four-baseline envelope; first-exceeds point uses ${ratio}× less EEG than ${baseline.name}'s peak.`)
    );

    const stageSection = el('div', 'stage-breakdown');
    const stageHeader = el('div', 'stage-breakdown-head');
    stageHeader.append(
      el('span', 'label', 'STAGE-WISE MATCHED-BUDGET GAIN (PP)'),
      el('span', 'stage-caption', 'N1 & REM highlighted (minorities)')
    );
    stageSection.append(
      stageHeader,
      renderStageBreakdown(byStage, { maxScale: 22, stageShares: row.stageShares })
    );

    const ws = EVIDENCE.windowSensitivity || {
      budgetSeconds: 10,
      lengths: [16, 32, 64],
      atL32: { dreams: 79.13, edf20: 78.12, edf78: 73.28, shhs: 73.94 },
      dreamsByLength: { l16: 77.65, l32: 79.13, l64: 75.95 },
      l16DropRangePp: [1.46, 1.64],
      l64DropRangePp: [1.90, 3.92],
      l16GainOverEctsRangePp: [9.66, 12.34],
      l64GainOverEctsRangePp: [7.28, 10.64],
    };
    const lWindowText = row.id === 'dreams'
      ? `Without retraining (L ∈ {16, 32, 64}): DREAMS-SUB L = 16: ${ws.dreamsByLength.l16.toFixed(2)}%, L = 32: ${ws.dreamsByLength.l32.toFixed(2)}%, L = 64: ${ws.dreamsByLength.l64.toFixed(2)}%. Across all 4 datasets, L = 16 / 64 reduce MF1 by ${ws.l16DropRangePp[0].toFixed(2)}–${ws.l16DropRangePp[1].toFixed(2)} / ${ws.l64DropRangePp[0].toFixed(2)}–${ws.l64DropRangePp[1].toFixed(2)} pp yet still beat the best ECTS baselines by ${ws.l16GainOverEctsRangePp[0].toFixed(2)}–${ws.l16GainOverEctsRangePp[1].toFixed(2)} / ${ws.l64GainOverEctsRangePp[0].toFixed(2)}–${ws.l64GainOverEctsRangePp[1].toFixed(2)} pp.`
      : `Without retraining, L = 16 / 64 reduce MF1 by ${ws.l16DropRangePp[0].toFixed(2)}–${ws.l16DropRangePp[1].toFixed(2)} / ${ws.l64DropRangePp[0].toFixed(2)}–${ws.l64DropRangePp[1].toFixed(2)} pp yet still beat the best ECTS baselines by ${ws.l16GainOverEctsRangePp[0].toFixed(2)}–${ws.l16GainOverEctsRangePp[1].toFixed(2)} / ${ws.l64GainOverEctsRangePp[0].toFixed(2)}–${ws.l64GainOverEctsRangePp[1].toFixed(2)} pp (DREAMS-SUB L = 16: ${ws.dreamsByLength.l16.toFixed(2)}%, L = 32: ${ws.dreamsByLength.l32.toFixed(2)}%, L = 64: ${ws.dreamsByLength.l64.toFixed(2)}%).`;

    const tenSecChip = el('div', 'ten-sec-chip');
    tenSecChip.append(
      el('span', 'label', '10 s/EPOCH BUDGET BENCHMARK'),
      el('strong', null, `10 s/epoch budget benchmark: ${row.atTenSeconds.toFixed(2)}% MF1 (L = 32 epochs)`),
      el('span', 'ten-sec-note', '32-epoch causal inference window'),
      el('p', 'l-window-note', lWindowText)
    );

    results.replaceChildren(bars, note, stageSection, tenSecChip);

    if (systemSource && row.sources) {
      systemSource.textContent = `${row.sources.system} · ${row.sources.systemGain} · Stages: ${row.sources.systemStages} · 10 s: ${row.sources.atTenSeconds}.`;
    }
  };

  const makeTable = (headers, rowsData, tableClass = 'ev-table') => {
    const wrap = el('div', 'table-scroll');
    wrap.tabIndex = 0;
    const table = el('table', tableClass);
    const thead = el('thead');
    const headRow = el('tr');
    for (const h of headers) {
      const th = el('th', h.className || null, h.text);
      th.scope = 'col';
      if (h.title) th.title = h.title;
      headRow.append(th);
    }
    thead.append(headRow);
    const tbody = el('tbody');
    for (const r of rowsData) {
      const tr = el('tr', r.className || null);
      r.cells.forEach((cellData, idx) => {
        const tag = idx === 0 && r.rowHeader !== false ? 'th' : 'td';
        const cell = el(tag, cellData.className || null);
        if (tag === 'th') cell.scope = 'row';
        if (cellData.title) cell.title = cellData.title;
        if (cellData.style) Object.assign(cell.style, cellData.style);
        if (cellData.lines) {
          cellData.lines.forEach((line, lineIdx) => {
            if (typeof line === 'string') {
              cell.append(el(lineIdx === 0 ? 'strong' : 'span', 'cell-sub', line));
            } else {
              const lineTag = line.tag || (lineIdx === 0 ? 'strong' : 'span');
              const lineClass = line.className ? `cell-sub ${line.className}` : 'cell-sub';
              cell.append(el(lineTag, lineClass, line.text));
            }
          });
        } else {
          cell.textContent = cellData.text;
        }
        tr.append(cell);
      });
      tbody.append(tr);
    }
    table.append(thead, tbody);
    wrap.append(table);
    return wrap;
  };

  const renderAblationsPanel = row => {
    const panel = get('ev-panel-ablations');
    if (!panel) return;
    const { table2, paretoGains, source, stagesSource } = row.ablation;
    const summary = EVIDENCE.ablationSummary;

    const header = el('div', 'ev-panel-head');
    header.append(
      el('h4', null, `${row.name} · Component ablations across acquisition costs`),
      el('p', 'ev-lead', 'Operating points from Table II at η = 0.5, 0.067, 0.0005 plus Table III(a)–(b) stage-wise matched-budget Pareto gains.')
    );

    const t2Rows = table2.map(entry => ({
      cells: [
        { text: `η = ${entry.eta}`, className: 'num' },
        { lines: [`${entry.pause.mf1.toFixed(2)}%`, `${entry.pause.seconds.toFixed(1)} s`], className: 'num is-pause-cell' },
        { lines: [`${entry.woPairedHistory.mf1.toFixed(2)}% (${formatSigned(entry.woPairedHistory.mf1 - entry.pause.mf1, 2)})`, `${entry.woPairedHistory.seconds.toFixed(1)} s`], className: 'num' },
        { lines: [`${entry.woEndpointMemory.mf1.toFixed(2)}% (${formatSigned(entry.woEndpointMemory.mf1 - entry.pause.mf1, 2)})`, `${entry.woEndpointMemory.seconds.toFixed(1)} s`], className: 'num' },
      ],
    }));
    const t2Table = makeTable(
      [
        { text: 'Cost η' },
        { text: 'PAUSE (MF1 / E[τ])', className: 'num' },
        { text: 'w/o Paired-History (Δ vs PAUSE)', className: 'num' },
        { text: 'w/o Endpoint Memory (Δ vs PAUSE)', className: 'num' },
      ],
      t2Rows
    );

    const cards = el('div', 'ev-two-col');
    const makeAblationCard = (titleText, gainObj, crossSummary, highlightMode, captionText) => {
      const card = el('article', 'ev-card');
      const stageHead = el('div', 'stage-breakdown-head');
      stageHead.append(
        el('span', 'label', 'STAGE-WISE GAIN (PP)'),
        el('span', 'stage-caption', captionText)
      );
      card.append(
        el('p', 'label', titleText),
        el('p', 'metric', `+${gainObj.mf1.toFixed(2)} pp`),
        el('p', 'ev-card-desc', `Matched-budget macro-F1 gain on ${row.name} (${crossSummary}).`),
        stageHead,
        renderStageBreakdown(gainObj.byStage, { maxScale: 21, highlightMode })
      );
      return card;
    };
    cards.append(
      makeAblationCard(
        'TABLE III(a) · GAIN OVER w/o ENDPOINT MEMORY',
        paretoGains.vsWoEndpointMemory,
        `4-dataset range ${summary.vsWoEndpointMemory.mf1Range[0].toFixed(2)}–${summary.vsWoEndpointMemory.mf1Range[1].toFixed(2)} pp; N1 ${summary.vsWoEndpointMemory.n1Range[0].toFixed(2)}–${summary.vsWoEndpointMemory.n1Range[1].toFixed(2)} pp; REM ${summary.vsWoEndpointMemory.remRange[0].toFixed(2)}–${summary.vsWoEndpointMemory.remRange[1].toFixed(2)} pp`,
        'minorities',
        'N1 & REM highlighted (minorities)'
      ),
      makeAblationCard(
        'TABLE III(b) · GAIN OVER w/o PAIRED-HISTORY',
        paretoGains.vsWoPairedHistory,
        `4-dataset range ${summary.vsWoPairedHistory.mf1Range[0].toFixed(2)}–${summary.vsWoPairedHistory.mf1Range[1].toFixed(2)} pp`,
        'top2',
        'Top-2 stage gains highlighted (Table III)'
      )
    );

    panel.replaceChildren(header, t2Table, cards, el('p', 'source', `${source}; ${stagesSource}`));
  };

  const renderHistoryPanel = row => {
    const panel = get('ev-panel-history');
    if (!panel) return;
    const h = EVIDENCE.history;

    const header = el('div', 'ev-panel-head');
    header.append(
      el('h4', null, 'History-switch reduction matrix I = (A − B) − (C − D) across all 32 settings'),
      el('p', 'ev-lead', 'Positive I (pp) indicates that paired-history training reduces macro-F1 loss when switching from complete 30 s histories to policy-generated histories, beyond its gain on complete histories.')
    );

    const cond = h.conditions || {
      A: 'Full-only training · Complete 30 s history',
      B: 'Full-only training · Policy-truncated history',
      C: 'Paired-history training · Complete 30 s history',
      D: 'Paired-history training · Policy-truncated history',
    };
    const abcdTable = makeTable(
      [
        { text: 'Training Regime' },
        { text: 'Complete 30 s History' },
        { text: 'Policy-Truncated History' },
        { text: 'Switch Drop (Complete − Policy)' },
      ],
      [
        {
          cells: [
            { text: 'Full-only (τ_m = 30)' },
            { lines: ['A · Full-only · Complete 30 s', cond.A] },
            { lines: ['B · Full-only · Policy', cond.B] },
            { lines: ['Δ_F = A − B', `${h.ranges.fullOnlyLoss[0].toFixed(2)}–${h.ranges.fullOnlyLoss[1].toFixed(2)} pp (missing EEG + shift)`] },
          ],
        },
        {
          cells: [
            { text: 'Paired-history (τ^F = 30, τ^R ~ Unif)' },
            { lines: ['C · Paired · Complete 30 s', cond.C] },
            { lines: ['D · Paired · Policy', cond.D] },
            { lines: ['Δ_P = C − D', 'Smaller switch drop under shift'] },
          ],
        },
        {
          cells: [
            { text: 'Paired − Full contrast' },
            { lines: ['C − A', `+${h.ranges.advantageOnComplete[0].toFixed(2)} to +${h.ranges.advantageOnComplete[1].toFixed(2)} pp`] },
            { lines: ['D − B', `+${h.ranges.advantageOnPolicy[0].toFixed(2)} to +${h.ranges.advantageOnPolicy[1].toFixed(2)} pp`] },
            { lines: ['I = Δ_F − Δ_P = (D − B) − (C − A)', `+${h.ranges.reduction[0].toFixed(2)}–+${h.ranges.reduction[1].toFixed(2)} pp (all 32 settings)`], className: 'is-pause-cell' },
          ],
        },
      ],
      'ev-table abcd-table'
    );

    const summaryStrip = el('div', 'ev-summary-strip');
    const statItems = [
      { label: 'FULL-ONLY SWITCHING LOSS Δ_F', val: `${h.ranges.fullOnlyLoss[0].toFixed(2)}–${h.ranges.fullOnlyLoss[1].toFixed(2)} pp`, sub: 'Δ_F = A − B (missing EEG + shift, 8 costs)' },
      { label: 'SWITCH-LOSS REDUCTION I', val: `${h.ranges.reduction[0].toFixed(2)}–${h.ranges.reduction[1].toFixed(2)} pp`, sub: 'I = Δ_F − Δ_P > 0 in all 32 pairs' },
      { label: '1 s PREFIX GAIN (η = 0.5)', val: `+${h.ranges.oneSecondAdvantageAtTightestCost[0].toFixed(2)} to +${h.ranges.oneSecondAdvantageAtTightestCost[1].toFixed(2)} pp`, sub: 'Paired vs. Full-only on shared policy history' },
    ];
    for (const stat of statItems) {
      const card = el('div', 'ev-stat-card');
      card.append(el('p', 'label', stat.label), el('p', 'ev-stat-val', stat.val), el('p', 'ev-stat-sub', stat.sub));
      summaryStrip.append(card);
    }

    const protocolText = h.protocolNote || 'Frozen Full-only and Paired backbones evaluated under shared endpoint positions from the frozen PAUSE policy; scores use uncalibrated ML (argmax_c s_{n,k}(c)) averaged over all 30 prefixes and 20 folds (excluding window-start epoch n = 1).';
    const gapText = h.gapNote || 'Full-only drop Δ_F = A − B (0.89–12.02 pp across all 8 costs, mean E[τ] ≈ 1.1–21 s) combines unavoidable missing-EEG loss with shift; I = (A − B) − (C − D) isolates the net reduction in history-switch sensitivity.';
    const historyCaveat = el('p', 'ev-caveat', `Controlled protocol: ${protocolText} ${gapText}`);

    const heatHeaders = [{ text: 'Dataset' }].concat(
      h.costs.map(c => ({
        text: `η=${ETA_LABELS[c] || c}`,
        className: 'num',
        title: `η = ${c}`,
      }))
    );

    const minVal = h.ranges.reduction[0];
    const maxVal = h.ranges.reduction[1];
    const heatRows = data.map(ds => {
      const values = h.reductionByDataset[ds.id];
      return {
        className: ds.id === row.id ? 'is-selected-row' : null,
        cells: [{ text: ds.name }].concat(
          values.map((val, cIdx) => {
            const norm = (val - minVal) / (maxVal - minVal || 1);
            // Smooth light-to-medium blue tint keeping >=10:1 contrast with #122548 text.
            const r = Math.round(241 - norm * (241 - 186));
            const g = Math.round(246 - norm * (246 - 214));
            const b = Math.round(255 - norm * (255 - 254));
            return {
              text: val.toFixed(2),
              className: 'num heatmap-cell',
              title: `${ds.name}, η = ${h.costs[cIdx]}: I = ${val.toFixed(2)} pp`,
              style: {
                backgroundColor: `rgb(${r}, ${g}, ${b})`,
                color: '#122548',
              },
            };
          })
        ),
      };
    });
    const heatmap = makeTable(heatHeaders, heatRows, 'ev-table heatmap-table');

    const oneSecStrip = el('div', 'one-sec-strip');
    oneSecStrip.append(el('span', 'label', 'FIG. 3(b) · 1 s PREFIX ADVANTAGE AT η = 0.5:'));
    for (const ds of data) {
      const gain1s = h.oneSecondByDataset[ds.id];
      const chip = el('span', `one-sec-chip${ds.id === row.id ? ' is-selected' : ''}`, `${ds.name}: +${gain1s.toFixed(2)} pp`);
      oneSecStrip.append(chip);
    }

    panel.replaceChildren(header, abcdTable, summaryStrip, historyCaveat, heatmap, oneSecStrip, el('p', 'source', h.source));
  };

  const renderPolicyPanel = row => {
    const panel = get('ev-panel-policy');
    if (!panel) return;
    const { matched } = row;
    const fullObs = EVIDENCE.fullObservation;
    const cw = EVIDENCE.classWeight;

    const header = el('div', 'ev-panel-head');
    header.append(
      el('h4', null, `${row.name} · Frozen-backbone policy comparison & 30 s full-observation check`),
      el('p', 'ev-lead', 'Isolates the utility-aware Bayes stopping policy on the same calibrated backbone (Table III(c)) and confirms 30 s backbone competitiveness (Table IV).')
    );

    const matchedCard = el('article', 'ev-card policy-matched-card');
    const policyMeta = EVIDENCE.policyComparisonMeta || {
      comparators: ['Fixed', 'Threshold', 'AAGC', 'GP (α = 0)'],
      note: 'All policies share the frozen PAUSE backbone and calibration; Table III(c) reports matched-budget F1 gains (pp) over their upper envelope.',
      n3MainVsSuppNote: 'Sleep-EDF-20 N3 prints 2.92 in Manuscript Table III(c) and 2.93 in Supplement Table XXIV.',
    };
    const comparators = policyMeta.comparators || ['Fixed', 'Threshold', 'AAGC', 'GP (α = 0)'];
    const matchedStats = el('div', 'policy-metrics-row');
    matchedStats.append(
      el('div', null, `${row.name} matched-budget gain: `),
      el('strong', 'policy-pill', `+${matched.gain.toFixed(2)} pp`),
      el('span', 'policy-peak', `Peak gain vs. best control policy: +${matched.peakGain.toFixed(2)} pp`)
    );
    const matchedStageHead = el('div', 'stage-breakdown-head');
    matchedStageHead.append(
      el('span', 'label', 'STAGE-WISE MATCHED-BUDGET GAIN (PP)'),
      el('span', 'stage-caption', 'Top-2 stage gains highlighted (Table III)')
    );
    matchedCard.append(
      el('p', 'label', 'TABLE III(c) · PAUSE VS. CONTROL POLICIES ON FROZEN BACKBONE'),
      el('p', 'ev-card-desc', `Frozen-backbone comparators: ${comparators.join(', ')}. ${policyMeta.note || ''}`),
      matchedStats,
      matchedStageHead,
      renderStageBreakdown(matched.byStage, { maxScale: 16, signed: true, highlightMode: 'top2' })
    );
    if (row.id === 'edf20') {
      matchedCard.append(
        el('p', 'ev-caveat', policyMeta.n3MainVsSuppNote || 'Sleep-EDF-20 N3: Main Table III(c) prints 2.92; Supp. Table XXIV prints 2.93.')
      );
    }

    const t4Headers = [{ text: 'Backbone (30 s full)' }].concat(
      data.map(ds => ({
        text: ds.name,
        className: `num${ds.id === row.id ? ' is-selected-col' : ''}`,
      }))
    );
    const bestByDataset = {};
    for (const ds of data) {
      const entries = fullObs.byDataset[ds.id];
      bestByDataset[ds.id] = {
        mf1: Math.max(...entries.map(item => item.mf1)),
        mgm: Math.max(...entries.map(item => item.mgm)),
        acc: Math.max(...entries.map(item => item.acc)),
      };
    }
    const t4Rows = fullObs.methods.map(methodName => ({
      className: methodName === 'PAUSE' ? 'is-pause-row' : null,
      cells: [{ text: methodName }].concat(
        data.map(ds => {
          const entry = fullObs.byDataset[ds.id].find(item => item.method === methodName);
          const best = bestByDataset[ds.id];
          const isBestMf1 = entry.mf1 === best.mf1;
          const isBestMgm = entry.mgm === best.mgm;
          const isBestAcc = entry.acc === best.acc;
          const hasBest = isBestMf1 || isBestMgm || isBestAcc;
          const metricLine = (metricName, mean, sd, isBest) => ({
            text: `${metricName} ${mean.toFixed(2)}±${sd.toFixed(2)}`,
            tag: isBest ? 'strong' : 'span',
            className: isBest ? 'is-best-metric' : null,
          });
          return {
            className: `num${ds.id === row.id ? ' is-selected-col' : ''}${hasBest ? ' is-best-metric' : ''}`,
            lines: [
              metricLine('MF1', entry.mf1, entry.mf1Sd, isBestMf1),
              metricLine('MGm', entry.mgm, entry.mgmSd, isBestMgm),
              metricLine('ACC', entry.acc, entry.accSd, isBestAcc),
            ],
          };
        })
      ),
    }));
    const t4Card = el('article', 'ev-card');
    t4Card.append(
      el('p', 'label', 'TABLE IV · FULL-OBSERVATION 30 s BACKBONE BENCHMARK (%)'),
      el('p', 'ev-card-desc', `${fullObs.note} Bold marks the column maximum per metric (MF1, MGm, ACC).`),
      makeTable(t4Headers, t4Rows, 'ev-table backbone-table'),
      el('p', 'ev-caveat', fullObs.caveatNote || 'PAUSE leads mean MF1 on 3/4 datasets (trails AttnSleep by 0.38 pp on Sleep-EDF-78) and mean ACC on 4/4, while trailing AttnSleep/LGSleepNet on MGm on Sleep-EDF-20/78. Between-fold SDs are sizable and N3 remains the backbone weakness (Supplement §VI-D, Table XXVII).')
    );

    const cwCard = el('article', 'ev-card class-weight-card');
    const defaultCw = cw.values.find(v => v.alpha === cw.defaultAlpha);
    cwCard.append(
      el('p', 'label', `TABLE V · CLASS-WEIGHT SENSITIVITY (${cw.dataset}, ${cw.patchRange[0].toFixed(2)}–${cw.patchRange[1].toFixed(2)} s)`),
      el('p', 'ev-card-desc', `Default α = ${cw.defaultAlpha} maximizes macro-F1 at ${defaultCw.mf1.toFixed(2)}% (+${cw.gainOverZero.toFixed(2)} pp over class-neutral α = 0).`)
    );
    const cwGrid = el('div', 'alpha-strip');
    for (const pt of cw.values) {
      const chip = el('div', `alpha-chip${pt.alpha === cw.defaultAlpha ? ' is-default' : ''}`);
      chip.append(el('span', 'alpha-k', `α = ${pt.alpha}`), el('strong', null, `${pt.mf1.toFixed(2)}%`));
      cwGrid.append(chip);
    }
    cwCard.append(
      cwGrid,
      el('p', 'ev-caveat', cw.costFormulaNote || 'Misclassification cost Λ_{c,a} = ρ w_c I[c ≠ a] with ρ = 10; α = 0 yields calibrated MAP and α = 0.5 applies tempered inverse-prevalence weighting.')
    );

    panel.replaceChildren(
      header,
      matchedCard,
      t4Card,
      cwCard,
      el('p', 'source', `${row.sources.matched} · ${fullObs.source} · ${cw.source}`)
    );
  };

  const renderOnlinePanel = row => {
    const panel = get('ev-panel-online');
    if (!panel) return;
    const sc = EVIDENCE.spectralChange;
    const om = EVIDENCE.onlineMass;

    const modRange = om.moderateToHighDelayRatioRange || [23.6, 42.2];
    const header = el('div', 'ev-panel-head');
    header.append(
      el('h4', null, 'Spectral-change endpoint intervention & online PAUSE vs. offline MASS'),
      el('p', 'ev-lead', `Causal endpoint interventions confirm endpoint memory carries transition-sensitive information (Table VI), while online PAUSE trails offline MASS by ≤ ${om.moderateToHighDeltaWithinPp.toFixed(2)} pp at moderate-to-high ceilings (11.72–21.0 s) with ${modRange[0]}×–${modRange[1]}× lower delay (${om.delayRatioRange[0]}×–${om.delayRatioRange[1]}× across all 3.0–21.0 s ceilings).`)
    );

    const formula = sc.changeScoreFormula || 'D_{m,j} = ||v_{m,j} − v_{m,j−1}||_2 / √128 on training-normalized 128-bin log-PSD patches';
    const retainDef = sc.retainDefinition || 'τ_m − 1 → τ_m (effect of retaining the last observed second)';
    const completeDef = sc.completeDefinition || 'τ_m → 30 (effect of restoring unacquired suffix when D_{m,τ_m+1} ≥ q90)';
    const unroundedNote = sc.unroundedNote || 'Contrasts use unrounded estimates (Manuscript Table VI).';
    const retainCosts = (sc.summary && (sc.summary.retainHighPositiveCosts || sc.summary.retainPositiveCosts)) || '7 of 8 acquisition costs on both Sleep-EDF-78 and SHHS';
    const completeCosts = (sc.summary && sc.summary.completeHighMinusLowPositiveCosts) || '8 of 8 acquisition costs on both Sleep-EDF-78 and SHHS';

    const scCard = el('article', 'ev-card spectral-card');
    scCard.append(
      el('p', 'label', `TABLE VI · SPECTRAL-CHANGE ENDPOINT INTERVENTION (η = ${sc.eta})`),
      el('p', 'ev-card-desc', `${sc.metric} (${sc.thresholds.high} vs. ${sc.thresholds.low}) using ${formula}. High-change Retain last 1 s gains are positive at ${retainCosts}; Complete High − Low contrasts are positive at all ${completeCosts.replace(/^all\s+/i, '')}.`)
    );
    const scRows = [];
    for (const item of sc.retainLastObservedSecond) {
      scRows.push({
        className: item.id === row.id ? 'is-selected-row' : null,
        cells: [
          { text: `Retain last 1 s · ${item.dataset}` },
          { text: item.matchedPairs.toLocaleString('en-US'), className: 'num' },
          { text: formatSigned(item.high, 2), className: 'num' },
          { text: formatSigned(item.low, 2), className: 'num' },
          { text: formatSigned(item.highMinusLow, 2), className: 'num' },
        ],
      });
    }
    for (const item of sc.completePrecedingEpoch) {
      scRows.push({
        className: item.id === row.id ? 'is-selected-row' : null,
        cells: [
          { text: `Complete preceding · ${item.dataset}` },
          { text: item.matchedPairs.toLocaleString('en-US'), className: 'num' },
          { text: formatSigned(item.high, 2), className: 'num' },
          { text: formatSigned(item.low, 2), className: 'num' },
          { text: formatSigned(item.highMinusLow, 2), className: 'num' },
        ],
      });
    }
    scCard.append(
      makeTable(
        [
          { text: 'Intervention & Dataset' },
          { text: 'Matched pairs', className: 'num' },
          { text: 'High (D ≥ q90)', className: 'num' },
          { text: 'Low (D ≤ q50)', className: 'num' },
          { text: 'High − Low (pp)', className: 'num' },
        ],
        scRows
      ),
      el('p', 'ev-caveat', `Retain last 1 s: ${retainDef}. Complete preceding: ${completeDef}. ${unroundedNote}`)
    );

    const ceilingNote = om.effectiveCeilingNote || 'Budget ceilings (3.0 / 11.72 / 21.0 s) correspond to 10% / 40% / 72% nominal MASS integrity after integer epoch/patch masking over 32×30 tokens (Supplement Table XXVIII).';
    const omCard = el('article', 'ev-card mass-card');
    omCard.append(
      el('p', 'label', 'SUPPLEMENT TABLE XXIX · ONLINE PAUSE VS. OFFLINE MASKING (MASS)'),
      el('p', 'ev-card-desc', `MASS incurs a mean ${om.massMeanDelaySeconds} s label delay (${om.massDelayRangeSeconds[0]}–${om.massDelayRangeSeconds[1]} s across 32-epoch blocks). At moderate-to-high ceilings (11.72–21.0 s), PAUSE stays within ${om.moderateToHighDeltaWithinPp.toFixed(2)} pp (mean diff ${om.moderateToHighMeanDiffPp.toFixed(2)} pp) at ${modRange[0]}×–${modRange[1]}× lower delay (${om.delayRatioRange[0]}×–${om.delayRatioRange[1]}× across all three 3.0–21.0 s ceilings).`)
    );
    const omRows = [];
    for (const ds of data) {
      for (const entry of om.byDataset[ds.id]) {
        const ceilingLabel = Number.isInteger(entry.ceiling) ? entry.ceiling.toFixed(1) : String(entry.ceiling);
        omRows.push({
          className: ds.id === row.id ? 'is-selected-row' : null,
          cells: [
            { text: `${ds.name} · ${ceilingLabel} s (${entry.nominalIntegrityPct}%)` },
            { text: `${entry.mass.mf1.toFixed(2)}% (${entry.mass.delay} s)`, className: 'num' },
            { text: `${entry.pause.mf1.toFixed(2)}% (${entry.pause.delay.toFixed(2)} s)`, className: 'num' },
            { text: `${formatSigned(entry.deltaMf1, 2)} pp`, className: 'num' },
            { text: `${entry.delayRatio.toFixed(1)}×`, className: 'num' },
          ],
        });
      }
    }
    omCard.append(
      makeTable(
        [
          { text: 'Dataset & Budget Ceiling' },
          { text: 'MASS (MF1 / Delay)', className: 'num' },
          { text: 'PAUSE (MF1 / Delay)', className: 'num' },
          { text: 'ΔMF1', className: 'num' },
          { text: 'Speedup', className: 'num' },
        ],
        omRows,
        'ev-table mass-table'
      ),
      el('p', 'ev-caveat', ceilingNote)
    );

    panel.replaceChildren(header, scCard, omCard, el('p', 'source', `${sc.source} · ${om.source}`));
  };

  let activeEvTab = 'ablations';
  const evTabs = evTablist ? Array.from(evTablist.querySelectorAll('[data-ev-tab]')) : [];
  const evPanels = {
    ablations: get('ev-panel-ablations'),
    history: get('ev-panel-history'),
    policy: get('ev-panel-policy'),
    online: get('ev-panel-online'),
  };

  const syncEvTabs = () => {
    evTabs.forEach(tab => {
      const selected = tab.dataset.evTab === activeEvTab;
      tab.setAttribute('aria-selected', String(selected));
      tab.tabIndex = selected ? 0 : -1;
    });
    for (const [key, panel] of Object.entries(evPanels)) {
      if (panel) panel.hidden = key !== activeEvTab;
    }
  };

  const renderAllEvidence = () => {
    const row = data.find(entry => entry.id === dataset.value) || data[0];
    renderPanel1(row);
    renderAblationsPanel(row);
    renderHistoryPanel(row);
    renderPolicyPanel(row);
    renderOnlinePanel(row);
    syncEvTabs();
  };

  evTabs.forEach((tab, index) => {
    tab.addEventListener('click', () => {
      activeEvTab = tab.dataset.evTab;
      syncEvTabs();
    });
    tab.addEventListener('keydown', event => {
      let nextIndex = null;
      if (event.key === 'ArrowRight') nextIndex = (index + 1) % evTabs.length;
      else if (event.key === 'ArrowLeft') nextIndex = (index - 1 + evTabs.length) % evTabs.length;
      else if (event.key === 'Home') nextIndex = 0;
      else if (event.key === 'End') nextIndex = evTabs.length - 1;
      if (nextIndex === null) return;
      event.preventDefault();
      evTabs[nextIndex].focus();
      activeEvTab = evTabs[nextIndex].dataset.evTab;
      syncEvTabs();
    });
  });

  dataset.addEventListener('change', renderAllEvidence);
  renderAllEvidence();
})();
