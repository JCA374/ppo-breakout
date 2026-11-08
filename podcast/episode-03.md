# Episode 03: A quarter of the way

This is the update at ten million frames. That is a quarter of the run, and it is exactly where the original plan would have stopped.

So what would that plan have delivered? The training games now average about fifty five points. The goal check at ten million frames played ten full games on seeds the agent never trained on. It averaged forty eight point seven points. The worst game scored sixteen, and the best scored eighty nine. With the old plan, the learning rate would be zero now, and this is roughly where the agent would have stayed. Far from three hundred. Stretching the run to forty million frames was the right call.

The goal checks so far tell a clear story. Five points at two and a half million frames. Twelve at five million. Thirty four at seven and a half million. And now forty nine. The score is still climbing steadily, and the checks on unseen games agree with the training numbers.

The training signals look healthy. Each update changes the policy only a little: the measured divergence stays around four to six thousandths. About one sample in eight hits the clip. That is the range the method is meant to work in.

You asked for more depth on how the project works, so the Lessons tab now has two parts. The first part lists the lessons learned. The second part has short narrated videos. One shows how the pieces fit together: the trainer, the evaluator, the website publisher and the studio for videos. One shows how a colour screen becomes the agent's small gray input. And one is about speed, and the units mistake that almost cut the run short. A fourth, on reading PPO's health signals, will use the finished run's curves.

What's next. Thirty million frames remain, about seven and a half hours. Three hundred points means six times today's score, so it is not certain. The jump usually comes once the agent learns to dig a tunnel through the wall. I'll report at twenty million frames, or sooner if the tunnel shows up.
