    
class MiniCourt:
    def __init__(self) -> None:
        pass

    def convert_bounding_boxes_to_mini_court_coordinates(self, player_boxes, ball_boxes, original_court_key_points ):
        
        pass
        
        player_heights = {
            1: constants.PLAYER_1_HEIGHT_METERS,
            2: constants.PLAYER_2_HEIGHT_METERS
        }

        output_player_boxes= []
        output_ball_boxes= []

        assert len(player_boxes) == len(ball_boxes)

        for frame_num, player_bbox in enumerate(player_boxes):
            ball_box = ball_boxes[frame_num]  # [1]
            ball_box = list(ball_box.values())
            # print()
            # print(frame_num)
            # print()
            # # print(list(ball_box.values()))
            # print()
            # print(type(ball_box))
            # print()
            try:
                ball_position = get_center_of_bbox(ball_box[0])
            except:
                print("WE GET NONE HERE")
                ball_position = (0, 0)

            closest_player_id_to_ball = min(player_bbox.keys(), key=lambda x: measure_distance(ball_position, get_center_of_bbox(player_bbox[x])))
            if closest_player_id_to_ball > 1:
                closest_player_id_to_ball = 1

            output_player_bboxes_dict = {}
            for player_id, bbox in player_bbox.items():
                if player_id > 1:
                    player_id = 1   # Because here is a List. In the future, when we use validation, it's necerary
                foot_position = get_foot_position(bbox)

                # Get The closest keypoint in pixels
                closest_key_point_index = get_closest_keypoint_index(foot_position,original_court_key_points, [0,2,12,13])
                closest_key_point = (original_court_key_points[closest_key_point_index*2], 
                                     original_court_key_points[closest_key_point_index*2+1])

                # Get Player height in pixels
                frame_index_min = max(0, frame_num-20)
                frame_index_max = min(len(player_boxes), frame_num+50)
                bboxes_heights_in_pixels = [get_height_of_bbox(player_boxes[i][player_id]) for i in range (frame_index_min,frame_index_max)]
                max_player_height_in_pixels = max(bboxes_heights_in_pixels)

                mini_court_player_position = self.get_mini_court_coordinates(foot_position,
                                                                            closest_key_point, 
                                                                            closest_key_point_index, 
                                                                            max_player_height_in_pixels,
                                                                            player_heights[player_id]
                                                                            )
                
                output_player_bboxes_dict[player_id] = mini_court_player_position

                if closest_player_id_to_ball == player_id:
                    # Get The closest keypoint in pixels
                    closest_key_point_index = get_closest_keypoint_index(ball_position,original_court_key_points, [0,2,12,13])
                    closest_key_point = (original_court_key_points[closest_key_point_index*2], 
                                        original_court_key_points[closest_key_point_index*2+1])
                    
                    mini_court_player_position = self.get_mini_court_coordinates(ball_position,
                                                                            closest_key_point, 
                                                                            closest_key_point_index, 
                                                                            max_player_height_in_pixels,
                                                                            player_heights[player_id]
                                                                            )
                    output_ball_boxes.append({1:mini_court_player_position})
            output_player_boxes.append(output_player_bboxes_dict)

        return output_player_boxes , output_ball_boxes