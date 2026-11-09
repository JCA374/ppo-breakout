# Episode 07: Four hundred

Good morning. The run is finished, and the goal is met.

Training ended at half past five, after forty million frames and nine and a half hours on the laptop. Then came the final test, by the rules I wrote down in the middle of the night. Three candidate models each played one hundred fresh games, on seeds nobody had used before. The final model won. It averaged four hundred and eight points, give or take seven. Ninety nine of its hundred games scored three hundred or more. The model from two and a half million frames earlier averaged three hundred and eighty six, and the one before that, three hundred and thirty six. So the last hours of training really did help.

Then the official goal check: the same ten fixed games that every checkpoint played. The final model averaged three hundred and ninety two points. Its worst game scored three hundred and forty one, and its best four hundred and twenty four. The goal was three hundred. With the training wrappers, the same ten games averaged four hundred and eight.

Four hundred is a telling number in Breakout. The first wall is worth four hundred and thirty two points. So the agent clears nearly all of it, almost every game. And it does dig tunnels. One more correction, then: last time I said I saw no tunnels. My detector looked for a red ball, but the ball takes the colour of the row it flies through. With a better test, the twenty and forty million frame models both break through and send the ball above the wall. Most of their points come after that.

What worked. Measuring first. Checking the units of the budget. Writing the test rules down before the answer existed. What I got wrong along the way: the frame budget at the start, a protocol gap that was mostly noise, and the tunnel detector. Each one is written down on the site, with its correction.

Everything is on the site. Four explainer videos about PPO, four lesson videos about how the project works, the replays on the Play tab, and the final model as a download. Thanks for following along.
