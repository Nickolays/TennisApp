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
        
        output = []
        for frame in tqdm(frames):
            # 1. 
            frame = self.preprocess(frame)
            # 2. 
            predict = self.get_predict(frame)
            # 3. 
            keypoints = self.postprocess(predict)
            # 4. Save
            output.append(keypoints)

        # 
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(output, f)

        assert len(frames) == len(output)
        return output

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
    
    def draw_keypoints(self, frames, keypoints):

        assert len(frames) == len(keypoints)
        # Plot keypoints on the image
        # print(image.shape)
        
        # Reshape image. This will resize the image to have n cols (width) and k rows (height):
        # image = cv2.resize(image, self.MODEL_INPUT_SIZE)
        out_frames = []
        for frame, kpoints in zip(frames, keypoints):
            # height, width, channels = image.shape
            # Normilize array
            frame = (frame - frame.min()) / (frame.max() - frame.min())
            frame = frame * 255
            frame = frame.astype(np.uint8)
            # print(image)
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            # 
            if self.device == torch.device("cuda"):
                kpoints = kpoints.cpu()
            # else:
            #     try:
            #         keypoints = keypoints.detach().numpy()
            #     except:
            #         keypoints = keypoints.cpu()
            for i, points in enumerate(kpoints):
                y = int(points[0])
                x = int(points[1])
                
                cv2.putText(frame, str(i), (y, x-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.circle(frame, (y, x), 2, (0, 0, 255), -1)
                frame[x, y, :] = 255, 255, 255

            out_frames.append(frame)
        
        assert len(out_frames) == len(frames), f"Length of in data doesn't equals output lenght, {len(frames)} and {len(out_frames)}"
        return out_frames


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
    
    def draw_bboxes(self, video_frames, ball_detections):
        """ """
        output_video_frames = [] 
        for frame, ball_dict in zip(video_frames, ball_detections):
            # Draw Bounding Box
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, 
                            f"Ball: {track_id}", 
                            (int(bbox[0]), int(bbox[1]-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.9,
                            (0, 0, 255),
                            2,
                )
                frame = cv2.rectangle(frame, 
                                      (int(x1), int(y1)), 
                                      (int(x2), int(y2)), 
                                      (0, 0, 255), 
                                      2)
            output_video_frames.append(frame)
        return output_video_frames


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
    
    def draw_bboxes(self, video_frames, player_detections):
        """ """
        assert len(video_frames) == len(player_detections), "Difference size of arrays"

        output_video_frames = [] 
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw Bounding Box
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                frame = cv2.putText(frame, 
                            f"Player ID: {track_id}", 
                            (int(bbox[0]), int(bbox[1]-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.9,
                            (0, 0, 255),
                            2,
                )
                frame = cv2.rectangle(frame,
                                      (int(x1), int(y1)), 
                                      (int(x2), int(y2)), 
                                      (0, 0, 255), 
                                      2)
                # 
            output_video_frames.append(frame)
        return output_video_frames
        # pass
    # pass


class NetDetector(BaseDetector):
    """
        Hypotesis: Predict will be better for every part of Image (Up net, bellow net)
    """
    pass