import abc, typing


class BaseDetector:
    @abc.abstractmethod
    def __init__(self, path) -> None:
        """ """
        self.path = path

    @abc.abstractmethod
    def predict(self, frames):
        # some code ...
        # 
        predict = self.postprocess(frames)

        assert len(frames) == len(predict)
        assert type(frames) == type(predict)
        # raise NotImplementedError
        return predict
    
    @abc.abstractmethod
    def evaluate(self) -> None:
        raise NotImplementedError
    
    @abc.abstractmethod
    def annotated_video(self):
        pass

    @abc.abstractmethod
    def postprocess(self, detections):
        assert isinstance(detections, typing.List)
        # Some code ..


        # assert isinstance(frames, typing.List)
        # raise NotImplementedError
        return detections
    # pass


class CourtDetector(BaseDetector):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        # """ """
        # self.path = path
        # 1. Read local

        # 2. Init model

        # 3. Load from mlflow

    pass


class BallDetector(BaseDetector):
    pass


class PlayerDetector(BaseDetector):
    def __init__(self, path) -> None:
        super().__init__(path)

    def predict(self, frames):
        return super().predict(frames)

    def postprocess(self, detections):
        """ """
        detections = self._choose_main_players(detections)
        return super().postprocess(detections)
    
    def _choose_main_players(self, detections):
        """
        """
        return detections
        # pass
    # pass


class NetDetector(BaseDetector):
    """
        Hypotesis: Predict will be better for every part of Image (Up net, bellow net)
    """
    pass