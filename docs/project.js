// Project-specific log text (ppo-breakout).
window.eventText = function (e) {
  const M = (f) => (f / 1e6).toFixed(1) + ' M frames';
  switch (e.kind) {
    case 'checkpoint': return [`Checkpoint at ${M(e.frames)}`, (e.note ? e.note + '. ' : '') + (e.score !== undefined && e.score !== '' ? `Training games ending around then scored ${e.score} on average. ` : '') + (e.replay ? 'Replay: the first 40 s of a fresh game, 2× speed.' : '')];
    case 'resume': return [`Run resumed from its checkpoint at ${M(e.frames)}`, ''];
    case 'run_start': return [`Run ${e.run} started`, e.run === 'smoke' ? '10-minute smoke test (5 + 5 minutes with a resume in between).' : '10 M frames with the CleanRL Atari settings.'];
    case 'run_end': return [`Run finished at ${M(e.frames)}`, ''];
    default: return null;
  }
};
