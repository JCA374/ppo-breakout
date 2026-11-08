# From screen to input

## The raw screen

The Atari screen is two hundred and ten by one hundred and sixty pixels, in colour, sixty times a second. That is far more than the agent needs.

## Skip and merge

The agent chooses only on every fourth frame, and its button is held for all four. The rewards of those frames are added up. The last two frames are merged by keeping the brighter pixel. The Atari draws some objects only on every other frame, and this way the ball never disappears.

## Shrink to gray

The merged frame turns gray and shrinks to eighty-four by eighty-four pixels. The ball becomes a dot of one or two pixels.

## Stack four

One picture can't show motion. So the input is the last four pictures together. From the small shifts between them, the network can tell where the ball is heading, and how fast.

## Two training shortcuts

Training uses two shortcuts that the test does not. When a life is lost, the training episode ends, so losing the ball looks like the end of the world. And every reward is clipped to one, so a red brick counts the same as a blue one. The test plays all five lives, and counts the real score.
