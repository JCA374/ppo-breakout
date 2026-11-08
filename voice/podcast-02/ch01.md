# Episode 02: The ball is found

In the last episode the score was stuck at about two points a game, and I had stretched the run from ten to forty million frames. This is the update two hours later. The short version: learning took off.

For the first two million frames, nothing moved. The average game stayed at two points, about what random button presses get. That flat start is normal for Breakout. The reward comes many steps after the move that earned it, and the agent first has to stumble on the link between the paddle and the ball.

Then, just after two million frames, the curve bent upward. By two and a half million frames the training games averaged about four points. By three and a third million, about nine. By four million, fourteen. Now, at six million frames, it is around twenty points a game. That is ahead of what I wrote down before the run, which was ten points at four million frames.

The first proper goal check was at two and a half million frames. Ten full games, all five lives, on game seeds the agent never trained on. It averaged five point four points, with a worst game of one and a best of nine. So the training numbers and the test numbers agree. That matters, because the training games are counted a little differently.

You can now see what the agent looks at. On the site's How it sees tab, there are saliency maps. They blur a small patch of the input and measure how much the network's answer changes. The actor's map sits right on the ball and on the paddle. The agent found the ball by itself, from gray pixels.

The training itself looks healthy. The policy's randomness is falling slowly, not collapsing. The average change per update stays small, as PPO intends.

What's next. The run is at fifteen percent, and it has about eight hours to go. Three hundred points is still far away. The big jump in Breakout usually comes when the agent learns to dig a tunnel up one side, so the ball bounces around behind the wall. I'll tell you when that happens. Meanwhile, the Lessons tab is getting a second part: short videos that explain how this whole project is set up, from the screen to the published site.
