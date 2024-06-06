from typing import List

class FillBallDetections:
    """ 
    Simple idea is use Classical ML Regression model. In the future try RNN model 
    """
    def __init__(self) -> None:
        pass

    def __call__(self, frames)-> List:
        # some code ...
        assert isinstance(frames, List)
        return frames