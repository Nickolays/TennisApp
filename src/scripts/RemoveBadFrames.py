

from typing import Any, List


class RemoveBadFrames:
    """
    """
    def __init__(self, path) -> None:
        # assert isinstance(path, Str)

        self.path = path
        pass

    def __call__(self, frames, *args: Any, **kwds: Any) -> Any:

        assert isinstance(frames, List), 'WRONG'
        return frames
    