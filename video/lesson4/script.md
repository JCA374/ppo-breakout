# Speed and budgets

## Measure first

Before the long run, I timed everything. The simulator alone runs about four thousand agent steps a second. The full learning loop runs about two hundred and ninety. So the game is not what slows us down.

## Where the time goes

One round of PPO takes about three and a half seconds. Less than one second of that is playing. The rest is the update: sixteen passes of backpropagation through the network.

## More threads, slower

Eight threads for the network were slower than four. This laptop has four real cores, and the game's own threads need them too.

## The budget's units

The plan said ten million frames, copied from the reference code. But that code counts agent steps, and each step is four frames. Its results are for forty million frames. The flat score early on made me check, and the fix was to stretch the run from its last checkpoint.

## What it costs

Forty million frames take about nine hours here, one night. Because that fits the time limit, nothing had to be rewritten in a faster language. Measuring first saved a day of work that wasn't needed.
