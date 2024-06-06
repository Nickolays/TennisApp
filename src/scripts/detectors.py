import abc, typing
from tqdm import tqdm

import pickle, cv2
import pandas as pd
import numpy as np

import torch
from torchvision import models, transforms
from ultralytics import YOLO



class BaseDetector:
    @abc.abstractmethod
    def __init__(self, model_path) -> None:
        """ """
        self.model_path = model_path

    @abc.abstractmethod
    def predict(self, frames):
        # frames = self.preprocess(frames)
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
    def preprocess(self, frames):
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
    """
    
        # 1. Read local
        # 2. Init model
        # 3. Load from mlflow
    """
    MODEL_INPUT_SIZE = (224, 224)
    OUTPUT_WIDTH = 224
    OUTPUT_HEIGHT = 224

    # def __init__(self, model_path: str) -> None:
    #     super().__init__(model_path)
    #     # self.model_path = model_path
        
    def __init__(self, model_path, device='cuda') -> None:
        """
        """
        self.device = device
        # Load a model
        # model = torch.load_model()
        # self.model = model.to(device)
        # Preprocess before use a Model. Init model and load
        model = models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, 14 * 2) # Replaces the last layer 
        weight_path = "/home/suetin/Projects/VSCode/TennisApp/SimpleApp/models/keypoints_model.pth"
        try:
            model.load_state_dict(torch.load(weight_path))
            model = model.to(device)
        except:
            print("Device is CPU")
            model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))

        self.model = model
        self.input_size = None

    def predict(self, frames, read_from_stub=False, stub_path=None):
        """
        
        """
        if self.input_size is None:
            self.input_size = frames[0].shape[:2]
            print(f'Input shape is: {self.input_size}')   # (464, 848), (h, w)???

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                keypoints = pickle.load(f)
            return keypoints
        
        for frame in tqdm(frames):
            # 1. 
            frame = self.preprocess(frame)
            # 2. 
            predict = self.get_predict(frame)
            # 3. 
            keypoints = self.postprocess(predict)
        # 
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(keypoints, f)

        return keypoints

    def preprocess(self, image):
        """  """
        pre_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.MODEL_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = pre_transforms(image)
        image = torch.unsqueeze(image, 0)
        image = image.to(self.device)
        
        return image

    def postprocess(self, predict):
        """  """
        keypoints = predict.reshape((14, 2))

        w, h = self.input_size
        # print(keypoints)
        # print(f"coef 0 is : {w / self.OUTPUT_WIDTH}")
        # print(f"coef 1 is : {h / self.OUTPUT_HEIGHT}")
        # Scale every points to original shape
        keypoints[:, 0] *= h / self.OUTPUT_HEIGHT 
        keypoints[:, 1] *= w / self.OUTPUT_WIDTH

        # print("AFTER  ", keypoints)

        return keypoints

    def get_predict(self, image):
        """ Return array, shape is (14, 2)"""
        predict = self.model(image)
        # Clear cash for GPU
        predict = predict.detach()   # 
        return predict


class BallDetector(BaseDetector):
    """ FIRST VERSION """
    def __init__(self, model_path) -> None:
        # super().__init__(model_path)
        self.model = YOLO(model_path, verbose=0)

    def predict(self, frames, read_from_stub=False, stub_path=None):
        # return super().predict(frames)
        return self.detect_frames(frames, read_from_stub, stub_path)

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        """ """
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in tqdm(frames):
            player_dict = self.detect_frame(frame)
            ball_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        assert isinstance(ball_detections, typing.List)
        assert len(frames) == len(ball_detections)
        return ball_detections 

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.3, verbose=False)[0]  # persist=True

        ball_dict = {
        }
        for box in results.boxes: 
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
        
        return ball_dict


class PlayerDetector(BaseDetector):
    def __init__(self, model_path) -> None:
        super().__init__(model_path)

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