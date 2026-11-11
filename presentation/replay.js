'use strict';

(() => {
  const get = id => document.getElementById(id);
  const horizon = 30;
  const example = {
    higher: {
      cost: .09,
      epochs: [
        { label: 'N2', stop: [.53, .40, .25, .17, .12, .10, .08, .07], next: [.35, .28, .20, .15, .08, .05, .04, .048] },
        { label: 'N1', stop: [.62, .55, .57, .44, .31, .22, .15], next: [.40, .39, .37, .30, .18, .14, .10] }
      ]
    },
    lower: {
      cost: .025,
      epochs: [
        { label: 'N2', stop: [.53, .40, .25, .17, .12, .10, .08, .07], next: [.35, .28, .20, .15, .08, .05, .04, .048] },
        { label: 'N1', stop: [.47, .49, .33, .22, .16, .13, .12, .08, .065], next: [.31, .30, .25, .15, .13, .09, .081, .055, .041] }
      ]
    }
  };

  const replay = get('replay');
  const scenario = get('scenario');
  const chart = get('risk-chart');
  const stepButton = get('replay-step');
  const playButton = get('replay-play');
  const resetButton = get('replay-reset');
  const epochList = get('epochs');
  const svgNs = 'http://www.w3.org/2000/svg';
  let rows = [];
  let timer = null;
  let active = 0;
  let count = 0;

  function setState(state) {
    const focused = document.activeElement;
    replay.dataset.state = state;
    playButton.textContent = state === 'playing' ? 'Pause' : state === 'paused' ? 'Resume' : 'Play';
    playButton.disabled = state === 'complete';
    stepButton.disabled = state === 'complete' || state === 'playing';
    if (state === 'complete' && (focused === playButton || focused === stepButton)) resetButton.focus();
  }

  function svgNode(tag, attrs, content) {
    const element = document.createElementNS(svgNs, tag);
    for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
    if (content !== undefined) element.textContent = content;
    chart.append(element);
    return element;
  }

  function renderChart(row, cost) {
    chart.replaceChildren();
    const x = k => 39 + (k - 1) * 12.45;
    const y = value => 133 - value * 175;
    for (const value of [0, .2, .4, .6]) {
      svgNode('line', { x1: 39, x2: 400, y1: y(value), y2: y(value), stroke: '#3a514a' });
      svgNode('text', { x: 30, y: y(value) + 4, 'text-anchor': 'end', fill: '#b3c7bd', 'font-size': 11 }, value.toFixed(1));
    }
    for (const k of [1, 5, 10, 15, 20, 25, 30]) {
      svgNode('text', { x: x(k), y: 151, 'text-anchor': 'middle', fill: '#b3c7bd', 'font-size': 11 }, String(k));
    }
    svgNode('text', { x: 400, y: 168, 'text-anchor': 'end', fill: '#b3c7bd', 'font-size': 11 }, 'ACQUIRED PREFIX (s)');
    if (!row.acquired) return;
    if (row.done && row.acquired < horizon) {
      svgNode('rect', { x: x(row.acquired) + 3, y: 28, width: 397 - x(row.acquired), height: 105, fill: '#23342e' });
      svgNode('text', { x: 396, y: 43, 'text-anchor': 'end', fill: '#b3c7bd', 'font-size': 11 }, 'SUFFIX UNOBSERVED');
    }
    const series = [[row.trace.stop, 0, '#ffce86', 'none'], [row.trace.next, cost, '#a9ffce', '5 4']];
    for (const [values, add, stroke, dash] of series) {
      const points = values.slice(0, row.acquired).map((value, index) => `${x(index + 1)},${y(value + add)}`);
      svgNode('polyline', { points: points.join(' '), fill: 'none', stroke, 'stroke-width': 2, 'stroke-dasharray': dash });
      svgNode('circle', { cx: x(row.acquired), cy: y(values[row.acquired - 1] + add), r: 4, fill: stroke });
    }
  }

  function render() {
    const cost = example[scenario.value].cost;
    const current = rows[active];
    replay.dataset.step = count;
    rows.forEach((row, index) => {
      row.status.textContent = row.done ? `Stop at ${row.acquired} s · output ${row.trace.label}` :
        row.acquired ? `${row.acquired} s acquired` : 'Waiting';
      row.heading.textContent = index === 0 ? 'Epoch 1 · no earlier history' : rows[0].done ?
        `Epoch 2 · carries endpoint at ${rows[0].acquired} s` : 'Epoch 2 · awaits endpoint';
      row.patches.forEach((patch, index) => {
        const k = index + 1;
        patch.className = row.done && k === row.acquired ? 'endpoint' : k <= row.acquired ? 'acquired' : row.done ? 'skipped' : '';
      });
    });
    get('cursor-label').textContent = count ? `EPOCH ${active + 1} / k = ${current.acquired}` : 'READY';
    renderChart(current, cost);
    if (!count) {
      get('decision').textContent = 'Ready. Acquire the first patch to begin.';
      chart.setAttribute('aria-label', 'Risk chart, waiting for the first patch');
      return;
    }
    const stop = current.trace.stop[current.acquired - 1];
    const next = current.trace.next[current.acquired - 1] + cost;
    const decision = current.done ? `Stop and emit ${current.trace.label}. ${horizon - current.acquired} s skipped in this epoch.` : 'Continue: acquire the next patch.';
    const comparison = `Stop ${stop.toFixed(3)} ${stop <= next ? '≤' : '>'} continue ${next.toFixed(3)}.`;
    get('decision').textContent = `${comparison} ${decision}`;
    chart.setAttribute('aria-label', `Epoch ${active + 1}, ${current.acquired} second prefix. ${comparison}`);
  }

  function stopTimer() {
    if (timer !== null) window.clearInterval(timer);
    timer = null;
  }

  function step() {
    if (replay.dataset.state === 'complete') return;
    if (rows[active].done) active++;
    const row = rows[active];
    const k = ++row.acquired;
    count++;
    row.done = k === horizon || row.trace.stop[k - 1] <= row.trace.next[k - 1] + example[scenario.value].cost;
    if (row.done && active === rows.length - 1) {
      stopTimer();
      setState('complete');
    } else if (replay.dataset.state === 'ready') {
      setState('paused');
    }
    render();
  }

  function reset() {
    stopTimer();
    active = 0;
    count = 0;
    epochList.replaceChildren();
    rows = example[scenario.value].epochs.map(trace => {
      const element = document.createElement('div');
      element.className = 'epoch';
      const header = document.createElement('div');
      header.className = 'epoch-heading';
      const heading = document.createElement('b');
      const status = document.createElement('span');
      header.append(heading, status);
      const rail = document.createElement('div');
      rail.className = 'patches';
      rail.setAttribute('aria-hidden', 'true');
      const patches = Array.from({ length: horizon }, () => document.createElement('i'));
      rail.append(...patches);
      element.append(header, rail);
      epochList.append(element);
      return { trace, acquired: 0, done: false, heading, status, patches };
    });
    setState('ready');
    render();
  }

  stepButton.addEventListener('click', step);
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
  reset();

  const dataset = get('dataset');
  const results = get('results');
  const data = window.PAUSE_EVIDENCE.datasets;
  for (const row of data) dataset.add(new Option(row.name, row.id));

  function bar(label, seconds, mf1) {
    const item = document.createElement('div');
    item.className = 'bar-row';
    const name = document.createElement('span');
    name.textContent = label;
    const track = document.createElement('div');
    track.className = 'bar-track';
    track.setAttribute('aria-hidden', 'true');
    const fill = document.createElement('div');
    fill.className = 'bar-fill';
    fill.style.width = `${seconds / horizon * 100}%`;
    track.append(fill);
    const value = document.createElement('span');
    value.className = 'bar-value';
    value.textContent = `${seconds.toFixed(2)} s · ${mf1.toFixed(2)}% MF1`;
    item.append(name, track, value);
    return item;
  }

  function renderResults() {
    const row = data.find(entry => entry.id === dataset.value);
    const { baseline, firstExceeds, gain } = row.system;
    const bars = document.createElement('div');
    bars.className = 'bars';
    bars.append(bar('PAUSE', firstExceeds.seconds, firstExceeds.mf1), bar(`${baseline.name} at its peak`, baseline.seconds, baseline.mf1));
    const note = document.createElement('p');
    note.className = 'data-note';
    note.textContent = `${row.name}: the first evaluated PAUSE point above every baseline peak. Average matched-budget macro-F1 gain over the four-baseline envelope: +${gain.toFixed(1)} pp.`;
    results.replaceChildren(bars, note);
  }

  dataset.addEventListener('change', renderResults);
  renderResults();
})();
