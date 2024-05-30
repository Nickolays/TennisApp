import cv2 
import sympy
from sympy import Line 


def read_video(video_path):
    """ 
    """
    assert type(video_path) == str
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    return frames

def save_video(output_video_frames, output_video_path):
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')  
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, 
                          fourcc, 
                          24, 
                          (output_video_frames[0].shape[1], output_video_frames[0].shape[0])
                          )
    for frame in output_video_frames:
        out.write(frame)
    out.release()


def line_intersection(line1, line2):
    """
    Find 2 lines intersection point
    """
    l1 = Line((line1[0], line1[1]), (line1[2], line1[3]))
    l2 = Line((line2[0], line2[1]), (line2[2], line2[3]))

    intersection = l1.intersection(l2)
    point = None
    if len(intersection) > 0:
        if isinstance(intersection[0], sympy.geometry.point.Point2D):
            point = intersection[0].coordinates
    return point