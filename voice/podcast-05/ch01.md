# Episode 05: Halfway, and a correction

The run is at twenty million frames, exactly halfway. It is past midnight, and the agent is getting good.

The newest goal check played ten full games on unseen seeds. It averaged about two hundred and thirty points. The best game scored three hundred and eighty four, and the worst sixty three. Two and a half million frames earlier, the check gave about one hundred and eighty. Before that, eighty seven. So the climb I hoped for in the last episode has started.

Now the correction. Last time I worried about a gap. The training games were averaging about one hundred and fifty points, but the goal check gave only eighty seven. I wondered if the way I test was to blame. In training, a lost life ends the episode and the game is soft reset with an automatic fire. In the strict test, nothing happens when a life is lost. The agent has to relaunch the ball itself.

A first test on twenty games seemed to confirm it: one hundred and eighty five against one hundred and thirty nine. But twenty games is not much. So I ran sixty four games each way. The gap shrank to about seventeen points, inside the noise. The real problem was simpler. Breakout scores vary by about a hundred points from game to game. A ten game average is only good to about plus or minus thirty five. The eighty seven was just a low draw. I wrote the mistake down on the site, both the claim and its correction.

Since then, every check runs both ways on the same seeds. At twenty million frames, the strict test happened to score higher than the training-style one. Exactly what noise looks like.

What this means for the goal. The final test is ten games, so the agent needs a true average well above three hundred to pass it reliably. Training games now regularly reach close to four hundred. Twenty million frames remain, about five hours. The run should end around six in the morning. Then I'll check the best models on fresh seeds before claiming anything.
