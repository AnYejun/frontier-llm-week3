/* ═══════════════════════════════════════════════════════════════
   Lab 1: SAE Dimension Explorer
   Interactive visualization of Superposition → Monosemanticity
   ═══════════════════════════════════════════════════════════════ */

const Lab1 = (() => {
  // ── Concepts ──
  const CONCEPTS = [
    { id: 'apple', label: '🍎 사과', labelEn: 'Apple', color: '#fb7185' },
    { id: 'newyork', label: '🗽 뉴욕', labelEn: 'New York', color: '#22d3ee' },
    { id: 'jobs', label: '💻 잡스', labelEn: 'Steve Jobs', color: '#a78bfa' },
    { id: 'music', label: '🎵 음악', labelEn: 'Music', color: '#fbbf24' },
    { id: 'ocean', label: '🌊 바다', labelEn: 'Ocean', color: '#34d399' },
    { id: 'fire', label: '🔥 불', labelEn: 'Fire', color: '#f97316' },
    { id: 'book', label: '📚 책', labelEn: 'Book', color: '#e879f9' },
    { id: 'star', label: '⭐ 별', labelEn: 'Star', color: '#facc15' },
    { id: 'code', label: '🖥️ 코드', labelEn: 'Code', color: '#4ade80' },
    { id: 'heart', label: '❤️ 사랑', labelEn: 'Love', color: '#f43f5e' },
  ];

  // ── Ground truth concept vectors (high-dimensional semantic space) ──
  function generateConceptVectors(numConcepts, trueDim = 50) {
    const vectors = [];
    for (let i = 0; i < numConcepts; i++) {
      const v = new Array(trueDim).fill(0);
      // each concept has a unique "true" direction
      for (let d = 0; d < trueDim; d++) {
        v[d] = Math.sin((i + 1) * (d + 1) * 0.7) * Math.cos(i * d * 0.3);
      }
      // normalize
      const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
      vectors.push(v.map(x => x / norm));
    }
    return vectors;
  }

  // ── Simulate encoding into n neurons ──
  function encodeIntoNeurons(conceptVecs, numNeurons) {
    const numConcepts = conceptVecs.length;
    const trueDim = conceptVecs[0].length;

    // Random projection matrix (trueDim → numNeurons)
    const W = [];
    for (let i = 0; i < numNeurons; i++) {
      const row = [];
      for (let j = 0; j < trueDim; j++) {
        row.push((Math.random() - 0.5) * 2 / Math.sqrt(trueDim));
      }
      W.push(row);
    }

    // Encode each concept
    const encoded = conceptVecs.map(cv => {
      const enc = W.map(row => {
        let dot = 0;
        for (let j = 0; j < trueDim; j++) dot += row[j] * cv[j];
        return dot;
      });
      return enc;
    });

    return encoded;
  }

  // ── Apply Top-K Sparsity ──
  function applyTopK(vectors, k) {
    return vectors.map(v => {
      const absVals = v.map((x, i) => ({ val: Math.abs(x), idx: i }));
      absVals.sort((a, b) => b.val - a.val);
      const topIndices = new Set(absVals.slice(0, k).map(x => x.idx));
      return v.map((x, i) => topIndices.has(i) ? x : 0);
    });
  }

  // ── Cosine Similarity ──
  function cosineSim(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      na += a[i] * a[i];
      nb += b[i] * b[i];
    }
    return na === 0 || nb === 0 ? 0 : dot / (Math.sqrt(na) * Math.sqrt(nb));
  }

  // ── Reconstruction Error ──
  function reconstructionError(original, encoded, numNeurons) {
    // Simple measure: how well can we match original cosine similarities?
    const n = original.length;
    let totalErr = 0;
    let count = 0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const origSim = cosineSim(original[i], original[j]);
        const encSim = cosineSim(encoded[i], encoded[j]);
        totalErr += Math.abs(origSim - encSim);
        count++;
      }
    }
    return count === 0 ? 0 : totalErr / count;
  }

  // ═══ Rendering ═══

  function renderHeatmap(containerId, vectors, labels, title) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const n = vectors.length;
    const matrix = [];
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        matrix.push(cosineSim(vectors[i], vectors[j]));
      }
    }

    let html = `<div style="margin-bottom:0.5rem;font-size:0.75rem;color:var(--text-secondary);font-weight:600;">${title}</div>`;
    html += `<div class="heatmap-labels" style="padding-left:55px;margin-bottom:2px;">`;
    for (let j = 0; j < n; j++) {
      html += `<div class="heatmap-label">${labels[j]}</div>`;
    }
    html += `</div>`;

    for (let i = 0; i < n; i++) {
      html += `<div style="display:flex;align-items:center;gap:3px;margin-bottom:3px;">`;
      html += `<div style="width:52px;text-align:right;font-size:0.6rem;color:var(--text-muted);font-family:var(--font-mono);">${labels[i]}</div>`;
      for (let j = 0; j < n; j++) {
        const val = matrix[i * n + j];
        const hue = val > 0 ? 260 : 350;
        const sat = Math.abs(val) * 80;
        const light = 12 + Math.abs(val) * 25;
        const alpha = 0.3 + Math.abs(val) * 0.7;
        html += `<div class="heatmap-cell" style="
          flex:1; height:36px;
          background:hsla(${hue},${sat}%,${light}%,${alpha});
          color:${Math.abs(val) > 0.3 ? 'var(--text-primary)' : 'var(--text-muted)'};
        ">${val.toFixed(2)}</div>`;
      }
      html += `</div>`;
    }

    container.innerHTML = html;
  }

  function renderNeurons(containerId, encodedVecs, concepts, sparsity) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const numNeurons = encodedVecs[0].length;
    const numConcepts = encodedVecs.length;
    let html = '';

    for (let n = 0; n < numNeurons; n++) {
      // Find which concepts activate this neuron
      const activations = [];
      for (let c = 0; c < numConcepts; c++) {
        const val = encodedVecs[c][n];
        if (Math.abs(val) > 0.01) {
          activations.push({ concept: concepts[c], val });
        }
      }

      const isInactive = activations.length === 0;
      const isPoly = activations.length > 1;
      const cls = isInactive ? 'inactive' : (isPoly ? 'polysemantic' : 'monosemantic');

      html += `<div class="neuron ${cls}">
        <div class="neuron-id">N${n}</div>
        <div class="neuron-concepts">
          ${activations.length === 0 ? '∅' :
            activations.map(a => `<span style="color:${a.concept.color}">${a.concept.label.split(' ')[0]}</span>`).join(' ')}
        </div>
      </div>`;
    }

    container.innerHTML = html;
  }

  function renderBarChart(containerId, data, title) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const maxVal = Math.max(...data.map(d => d.value), 0.01);

    let html = `<div style="margin-bottom:0.5rem;font-size:0.75rem;color:var(--text-secondary);font-weight:600;">${title}</div>`;
    html += `<div class="bar-chart">`;

    data.forEach(d => {
      const height = (d.value / maxVal) * 160;
      html += `<div class="bar-wrapper">
        <div class="bar" style="height:${height}px;background:${d.color || 'var(--accent-indigo)'};" data-value="${d.value.toFixed(3)}"></div>
        <div class="bar-label">${d.label}</div>
      </div>`;
    });

    html += `</div>`;
    container.innerHTML = html;
  }

  function renderPipeline(containerId, numNeurons, sparsityK) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = `
      <div class="pipeline">
        <div class="pipeline-node active-node">
          <div class="node-icon">📥</div>
          <div class="node-label">Input</div>
          <div class="node-dim">d_model = 3</div>
        </div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-node active-node">
          <div class="node-icon">⬆️</div>
          <div class="node-label">Encoder</div>
          <div class="node-dim">W_enc [3 × ${numNeurons}]</div>
        </div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-node" style="border-color:var(--accent-emerald);">
          <div class="node-icon">⚡</div>
          <div class="node-label">ReLU</div>
          <div class="node-dim">+ Bias</div>
        </div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-node" style="border-color:var(--accent-amber);">
          <div class="node-icon">🎯</div>
          <div class="node-label">Top-K</div>
          <div class="node-dim">k = ${sparsityK}</div>
        </div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-node active-node">
          <div class="node-icon">⬇️</div>
          <div class="node-label">Decoder</div>
          <div class="node-dim">W_dec [${numNeurons} × 3]</div>
        </div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-node" style="border-color:var(--accent-emerald);">
          <div class="node-icon">📤</div>
          <div class="node-label">Output</div>
          <div class="node-dim">d_model = 3</div>
        </div>
      </div>
    `;
  }

  // ═══ Main Update ═══

  let cachedConceptVecs = null;

  function update() {
    const numNeurons = parseInt(document.getElementById('dim-slider')?.value || 3);
    const sparsityK = parseInt(document.getElementById('sparsity-slider')?.value || 3);
    const numConcepts = parseInt(document.getElementById('concepts-slider')?.value || 3);

    // Update displays
    const dimDisp = document.getElementById('dim-value');
    const sparDisp = document.getElementById('sparsity-value');
    const conDisp = document.getElementById('concepts-value');
    if (dimDisp) dimDisp.textContent = numNeurons;
    if (sparDisp) sparDisp.textContent = Math.min(sparsityK, numNeurons);
    if (conDisp) conDisp.textContent = numConcepts;

    // Clamp sparsity to max neurons
    const effectiveK = Math.min(sparsityK, numNeurons);

    // Generate concept vectors
    const usedConcepts = CONCEPTS.slice(0, numConcepts);
    if (!cachedConceptVecs || cachedConceptVecs.length !== numConcepts) {
      cachedConceptVecs = generateConceptVectors(numConcepts);
    }

    // Encode
    const encoded = encodeIntoNeurons(cachedConceptVecs, numNeurons);

    // Apply sparsity
    const sparse = applyTopK(encoded, effectiveK);

    // Labels
    const labels = usedConcepts.map(c => c.label.split(' ')[0]);

    // Dense heatmap
    renderHeatmap('heatmap-dense', encoded, labels, `Dense (${numNeurons} neurons, 모두 활성)`);

    // Sparse heatmap
    renderHeatmap('heatmap-sparse', sparse, labels, `Sparse (${numNeurons} neurons, top-${effectiveK})`);

    // Neuron visualization
    renderNeurons('neuron-vis-dense', encoded, usedConcepts, numNeurons);
    renderNeurons('neuron-vis-sparse', sparse, usedConcepts, effectiveK);

    // Reconstruction error comparison
    const dims = [3, 5, 8, 10, 15, 20, 30];
    const errors = dims.map(d => {
      const enc = encodeIntoNeurons(cachedConceptVecs, d);
      const sp = applyTopK(enc, Math.max(1, Math.floor(d * 0.3)));
      return {
        label: `d=${d}`,
        value: reconstructionError(cachedConceptVecs, sp, d),
        color: d === numNeurons ? 'var(--accent-cyan)' : 'rgba(99,102,241,0.4)'
      };
    });
    renderBarChart('error-chart', errors, '차원 수에 따른 Similarity Distortion');

    // Pipeline
    renderPipeline('sae-pipeline', numNeurons, effectiveK);

    // Polysemanticity score
    updatePolyScore(encoded, sparse, usedConcepts);
  }

  function updatePolyScore(dense, sparse, concepts) {
    const el = document.getElementById('poly-score');
    if (!el) return;

    const numNeurons = dense[0].length;
    let polyCountDense = 0, polyCountSparse = 0;

    for (let n = 0; n < numNeurons; n++) {
      let denseActive = 0, sparseActive = 0;
      for (let c = 0; c < concepts.length; c++) {
        if (Math.abs(dense[c][n]) > 0.01) denseActive++;
        if (Math.abs(sparse[c][n]) > 0.01) sparseActive++;
      }
      if (denseActive > 1) polyCountDense++;
      if (sparseActive > 1) polyCountSparse++;
    }

    const denseRatio = ((polyCountDense / numNeurons) * 100).toFixed(0);
    const sparseRatio = ((polyCountSparse / numNeurons) * 100).toFixed(0);

    el.innerHTML = `
      <div class="metric-row">
        <span class="metric-label">Dense Polysemanticity</span>
        <div class="metric-bar-container">
          <div class="metric-bar" style="width:${denseRatio}%;background:var(--accent-rose);"></div>
        </div>
        <span class="metric-value" style="color:var(--accent-rose);">${denseRatio}%</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">Sparse Polysemanticity</span>
        <div class="metric-bar-container">
          <div class="metric-bar" style="width:${sparseRatio}%;background:var(--accent-emerald);"></div>
        </div>
        <span class="metric-value" style="color:var(--accent-emerald);">${sparseRatio}%</span>
      </div>
      <p style="font-size:0.72rem;color:var(--text-muted);margin-top:0.75rem;line-height:1.5;">
        ${sparseRatio < denseRatio
          ? `✅ Sparsity가 polysemanticity를 <strong style="color:var(--accent-emerald)">${denseRatio - sparseRatio}%p</strong> 감소시켰습니다. 각 뉴런이 더 적은 개념만 인코딩합니다.`
          : `⚠️ 차원이 부족합니다. 뉴런 수를 늘려보세요!`}
      </p>
    `;
  }

  // ═══ Init ═══
  function init() {
    const dimSlider = document.getElementById('dim-slider');
    const sparsitySlider = document.getElementById('sparsity-slider');
    const conceptsSlider = document.getElementById('concepts-slider');

    if (dimSlider) dimSlider.addEventListener('input', () => { cachedConceptVecs = null; update(); });
    if (sparsitySlider) sparsitySlider.addEventListener('input', update);
    if (conceptsSlider) conceptsSlider.addEventListener('input', () => { cachedConceptVecs = null; update(); });

    update();
  }

  return { init, update };
})();
