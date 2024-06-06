from typing import List

# Shot Recognition
class ShotType:
    def __init__(self) -> None:
        pass

    def __call__(self, frames: List)-> List:
        #   some code ...
        assert isinstance(frames, List)
        return frames