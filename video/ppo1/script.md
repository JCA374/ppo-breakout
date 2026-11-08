# What PPO does

## The task

The agent sees only the screen. Four small gray pictures: the last four moments of the game. Every fifteenth of a second it picks one of four buttons: do nothing, fire, right or left. Breaking a brick earns points. Nobody tells it what a ball or a paddle is.

## One network, two heads

The agent is a single neural network with two outputs. The actor head gives a probability for each of the four buttons. The critic head gives one number: a guess of how much reward is still to come from this moment. The actor decides. The critic judges.

## Collecting a rollout

Learning happens in rounds. First the agent plays eight games side by side for one hundred and twenty-eight steps each. That is one thousand and twenty-four moments. For each one it writes down what it saw, what it pressed, how likely that was, what the critic expected, and the reward.

## Was it better than expected?

Next, every action gets a grade, called the advantage. If the rewards that followed were better than the critic expected, the advantage is positive. If they were worse, it is negative. The critic's guess is the yardstick.

## The careful update

Then the network is trained on that batch. Actions with a positive advantage become a bit more likely. Actions with a negative one become a bit less likely. The word that matters is "bit". PPO clips the change, so no single batch can push the policy far from the one that collected the data. It goes over the batch four times, in pieces of two hundred and fifty-six moments, then throws it away.

## Round after round

Then it plays again with the improved policy. Here, nearly ten thousand rounds make forty million frames. That is more than a week of non-stop play, squeezed into one night on a laptop.
