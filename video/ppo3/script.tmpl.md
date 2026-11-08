# The clipped objective

## Why be careful

PPO learns from games its own policy just played. If one update changes the policy a lot, the new player is a stranger to that data. Its next move can be bad, the next batch is collected by a worse player, and hours of learning can unravel. Older policy gradient methods were known for such collapses.

## The ratio

So PPO measures how far it has moved, one action at a time. The ratio is the new probability of an action divided by the old one. A ratio of one means nothing changed. One point two means the action became twenty percent more likely.

## The clip

The objective multiplies the ratio by the advantage. Then it clips the ratio between zero point nine and one point one. For a good action, the payoff rises until the ratio reaches one point one, and then it goes flat. For a bad action, it falls until zero point nine, and then it goes flat.

## Why flat means safe

On a flat stretch the gradient is zero. So once an action has moved ten percent the way its advantage asks, that sample stops pushing. And the objective takes the smaller of the clipped and unclipped values. If a change went the wrong way, the full penalty still counts. The clip limits greed, not correction.

## This run's numbers

These are the real values from this run. {KL_SENTENCE} {CLIP_SENTENCE} {TREND_SENTENCE}

## Small, safe steps add up

{SUMMARY_SENTENCE}
