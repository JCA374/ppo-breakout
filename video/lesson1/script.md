# How this project is set up

## The pieces

This project has four parts that run on their own. A trainer that plays and learns. An evaluator that tests the agent on games it has never seen. A publisher that turns the logs into this website. And a small studio that makes the videos and podcasts. They talk to each other only through files on disk.

## The trainer

The trainer runs eight Breakout games in envpool, a fast simulator written in C++. PyTorch runs the network on the laptop's processor. Every update writes its numbers to MLflow and to a plain table file. Every half a million frames, it saves a checkpoint.

## Surviving a crash

A small shell script, the supervisor, watches the trainer. If the trainer dies, the supervisor starts it again from the latest checkpoint. The checkpoint holds the network, the optimizer, the step count, the random number state and the MLflow run. So the run continues as if nothing had happened.

## Testing honestly

Every two and a half million frames, the evaluator plays ten full games with that checkpoint. All five lives count, the score is not clipped, and the game seeds were never used in training. Those results fill the goal table.

## The website

Every fifteen minutes, the publisher reads MLflow and exports it to small data files. It renders replays and saliency maps at low priority, so training keeps the processor. Then it commits and pushes, and GitHub Pages serves the result. The same loop restarts the trainer if it ever finds it gone.
