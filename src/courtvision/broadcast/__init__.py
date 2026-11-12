"""Reading the broadcast itself: the score graphic and the stadium serve-speed board.

Broadcast graphics are an independent measurement of the match: the score says when a point ended
and who won it, the serve marker says who served, and the stadium radar board says how fast. The
pipeline uses them as weak ground truth to check its own events and speeds, never as an input to a
line call.
"""
