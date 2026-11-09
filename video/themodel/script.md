# Inside the model

## One small network

The whole agent is one neural network with about one million seven hundred thousand weights. Three convolution layers find patterns in the pictures. One dense layer sums them up in five hundred and twelve numbers. Ninety five percent of the weights sit in that one dense layer.

## Two heads, one moment

From the summary, two small heads give the answers. The actor gives a probability for each button. At this moment, fire sixty two percent, do nothing thirty three, left five. The critic gives one number, the value: here about three. That is its guess of the reward still to come, counted in bricks and discounted over time.

## Over a game

Watch both over a stretch of a real game. The value climbs as the ball heads for the bricks, then drops the moment a brick breaks, because that reward is now in the past. The probabilities change from moment to moment, as the paddle chases the ball.

## The model's own map

Take the five hundred and twelve numbers for every moment of a game, and flatten them onto a page. They form two groups. One is the early game, with most of the wall still standing and a high value. The other is the late game, with only a few bricks left and a low value. Nobody told the network about game phases. It separated them on its own.

## How training shaped it

Compare the weights after half a million frames with the final ones. The first layer changed least: it still looks much like it did early on. The deeper layers changed more, and the heads most of all. The actor's weights grew ten times larger, as the policy became confident.

## The whole picture

So, pixels go in, patterns are found, a summary is formed, and two heads read it: one to act, one to judge. Everything else is learned from points.
