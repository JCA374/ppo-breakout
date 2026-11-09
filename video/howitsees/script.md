# How it sees

## Only numbers

The agent never sees a ball or a paddle. It gets numbers. Each picture is eighty four by eighty four pixels of gray, from zero for black to two hundred and fifty five for white. Zoom in on the ball, and it is just a few pixels around one hundred, in a sea of zeros.

## Four pictures, one input

It gets the last four pictures at once. One picture can't show motion. But four can: here, the colours mark where the ball was in each one. The trail shows which way it is going, and how fast.

## The first layer

The first layer has thirty two small filters. Each one is eight by eight pixels deep in all four pictures, and it slides across the screen in steps of four. Where the pixels match its pattern, it lights up. Some filters weigh the newest picture against the one before it, with opposite signs. Those are motion detectors. Nobody designed them. Training found them.

## Layer by layer

The first layer gives thirty two maps, twenty by twenty. Some light up on walls and brick rows, others only on the ball or the paddle. The second layer combines them into sixty four smaller maps, and the third into sixty four maps of seven by seven. Each step keeps less about where, and more about what.

## A summary of the moment

Then everything is squeezed into five hundred and twelve numbers. At this moment, eighty nine of them are active, and the rest are exactly zero. Both heads of the network read only this summary.

## Where it looks

Saliency maps show which pixels matter. Blur a small patch, and see how much the answer changes. For choosing a button, the agent looks at the ball and the paddle. It learned that from points alone.
