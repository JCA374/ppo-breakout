# Advantages and GAE

## Which move earned the points?

This is a real game from this run, played by the agent after ten million frames. A brick breaks, and the agent gets a point. But the move that made it possible came earlier: the paddle had to meet the ball, many steps before. PPO has to share out the credit fairly. That share is called the advantage.

## The critic's yardstick

The critic gives every moment a value: its guess of the reward still to come. Watch it on this game. It changes from moment to moment, between about half a point and two and a half. Just before the life is lost, it falls by more than half as the ball slips past the paddle.

## One step of surprise

The simplest grade looks one step ahead. Take the reward just received, add the critic's value of the next moment, and subtract its value of this one. That is the one-step surprise. When this life ends, the surprise is about minus two thirds. It is steady, but it trusts the critic completely.

## Looking all the way ahead

The other extreme adds up every real reward until the life ends, and subtracts the critic's guess. It trusts the critic not at all. But one lucky bounce changes everything, so it is three times as noisy here.

## Generalized advantage estimation

GAE blends the two. It adds up the one-step surprises into the future, each one weighted down by six percent per step. After twelve steps, less than a second of play, half the weight is gone. So the next few moves count most, and the far future barely counts.

## Three grades for the same game

Here are all three on the same stretch of game. The one-step grade is calm but short-sighted. The full return is honest but jumpy. The blend in the middle is what this agent was trained with.
