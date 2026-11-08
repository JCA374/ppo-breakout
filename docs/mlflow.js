// Training diagnostics on the Log page, drawn from data/mlflow.json (the MLflow file store exported by publish.py).
(async function () {
  const d = await get('mlflow.json');
  if (!d || !window.Chart) return;
  const exp = (d.experiments || []).find((e) => e.name === 'ppo-breakout');
  const evals = (d.experiments || []).find((e) => e.name === 'ppo-breakout-eval');
  const runs = exp ? exp.runs : [];
  const all = runs.concat(evals ? evals.runs : []);
  $('mlruns').innerHTML = '<tr><th>MLflow run</th><th>Experiment</th><th>Status</th><th>Key settings</th><th>Last values</th></tr>' +
    all.map((r) => {
      const p = r.params || {}, l = r.last || {};
      const set = p.total_frames ? `lr ${p.learning_rate}, clip ${p.clip_coef}, ${p.num_envs} envs × ${p.num_steps} steps, ${p.update_epochs} epochs × ${p.num_minibatches} minibatches, ${(p.total_frames / 1e6)} M frames`
        : `policy ${p.policy || ''}, ${p.episodes || ''} games, seed ${p.seed || ''}`;
      const last = l.frames ? `${(l.frames / 1e6).toFixed(2)} M frames, score ${l.ep_return_mean ?? '–'}, KL ${l.approx_kl ?? '–'}, clip ${l.clipfrac ?? '–'}, ${Math.round(l.sps || 0)} steps/s`
        : (l.eval_mean != null ? `mean ${l.eval_mean.toFixed(1)} (min ${l.eval_min}, max ${l.eval_max})` : '');
      return `<tr><td>${esc(r.name)}<div class="muted small">${esc(r.id.slice(0, 8))}</div></td><td>${esc(evals && evals.runs.includes(r) ? 'evaluation' : 'training')}</td><td>${esc(r.status)}</td><td class="small">${esc(set)}</td><td class="small">${esc(last)}</td></tr>`;
    }).join('');
  const panels = [['approx_kl', 'approx KL (old ‖ new policy)'], ['clipfrac', 'clip fraction'], ['entropy', 'policy entropy (nats; max ln 4 = 1.39)'],
    ['value_loss', 'value loss'], ['explained_variance', 'explained variance of returns'], ['sps', 'agent steps / second']];
  const colors = ['--accent', '--warn', '--good', '--bad'];
  $('diag').innerHTML = panels.map(([k, t]) => `<div class="chartbox"><div class="ttl small">${t}</div><div class="h2"><canvas id="c_${k}"></canvas></div></div>`).join('');
  for (const [k] of panels) {
    const ds = runs.filter((r) => r.curves && r.curves[k]).map((r, i) => ({ label: r.name, data: r.curves[k].map(([s, v]) => ({ x: s * 4 / 1e6, y: v })),
      borderColor: cssv(colors[i % 4]), pointRadius: 0, borderWidth: 1.2, tension: .2 }));
    new Chart($('c_' + k), { type: 'line', data: { datasets: ds }, options: { animation: false, maintainAspectRatio: false,
      scales: { x: { type: 'linear', title: { display: true, text: 'M frames', color: cssv('--muted') }, ticks: { color: cssv('--muted') }, grid: { color: cssv('--line') } },
        y: { ticks: { color: cssv('--muted') }, grid: { color: cssv('--line') } } }, plugins: { legend: { display: ds.length > 1, labels: { color: cssv('--ink'), boxWidth: 10 } } } } });
  }
})();
