# The clipped objective

## Why be careful

PPO learns from games its own policy just played. If one update changes the policy a lot, the data no longer describes the new player. A bad step makes a worse player, who collects worse data, and hours of learning can unravel.

## The ratio

So PPO measures how far it has moved, one action at a time. The ratio is the action's new probability divided by its old one. A ratio of one means nothing changed. Above one, the action became more likely. Below one, less likely.

## The clip

The objective multiplies the ratio by the advantage, and then clips the ratio to within ten percent of one. For a good action, the payoff rises until the action is ten percent more likely, and then it goes flat. For a bad action, it falls until the action is ten percent less likely, and then it goes flat.

## Why flat means safe

On a flat stretch the gradient is zero. So once an action has moved ten percent the way its advantage asks, that sample stops pushing. And because the objective takes the smaller value, a change in the wrong direction still pays the full penalty. The clip limits greed, not correction.

## This run's numbers

{KL_SENTENCE} {CLIP_SENTENCE} {TREND_SENTENCE}

## Small, safe steps add up

{SUMMARY_SENTENCE}
