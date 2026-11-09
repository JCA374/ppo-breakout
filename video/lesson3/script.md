# Reading PPO's health signals

## Why look inside

The score moves slowly, and it is noisy. So after every round of updates, PPO logs a few signals. They show whether learning is healthy, long before the score does.

## Entropy

Entropy measures how random the policy still is. With four buttons, it starts at its maximum: pure chance. In this run it fell to about two thirds of that maximum by the end, and it never collapsed. If it dropped to near zero early, the agent would stop exploring, and stop improving.

## KL divergence

The approximate KL divergence says how much the policy changed in one round. Here the typical value was five thousandths. Values ten times larger would mean the steps are too big, and the clip is not holding them back.

## Clip fraction

The clip fraction is the share of samples whose probability ratio hit the clip. Here it was about ten percent. Zero would mean the clip never matters. Half would mean almost every update is pushing against it.

## The critic

Explained variance says how much of the returns the critic predicts. One is perfect, and zero is useless. Here it started near one. That sounds great, but it wasn't: while the agent scored almost nothing, the returns were easy to predict. Once it began breaking bricks, explained variance fell to about one half and stayed there. The critic's job got harder, not worse.

## A flat start is not a bug

For the first two million frames, the score stayed at two points, while all these signals looked healthy. That is what a slow start looks like. Still, it was the flat score that made me check the plan, and find the units mistake.
