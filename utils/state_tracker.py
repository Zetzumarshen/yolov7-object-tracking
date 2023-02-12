import pandas as pd
# from utils.count_utils import check_box_position, frame_to_timestamp, get_line_orientation

class BBoxState:
    def __init__(self, object_id, bbox_xyxy, class_label, class_confidence, orientation_label = None):
        self.object_id = object_id # object ID are generated by the tracker
        self.class_label = class_label
        self.class_confidence = class_confidence
        self.bbox_xyxy = bbox_xyxy 
        self.orientation_label = orientation_label # right, left, intersect, above, bottom
        self.in_out_counter = None
        self.original_orientation_label = None

    def set_origin_label(self, original_orientation_label):
        self.original_orientation_label = original_orientation_label

    def __getitem__(self, key):
        if key == 'object_id':
            return self.object_id
        elif key == 'class_label':
            return self.class_label
        elif key == 'class_confidence':
            return self.class_confidence
        elif key == 'bbox_xyxy':
            return self.bbox_xyxy
        elif key == 'orientation_label':
            return self.orientation_label
        elif key == 'in_out_counter':
            return self.in_out_counter
        elif key == 'original_orientation_label':
            return self.original_orientation_label
        else:
            raise KeyError("Key '{}' not found in BBoxState".format(key))

class StateTracker:
    def __init__(self, line_roi, fps, in_orientation=None):
        """
        Initialize the StateTracker object.

        Parameters:
        - line_roi (str): The region of interest of the line.
        - in_orientation (str): What counts as in orientation opposed to out orientation. 
          Can be "left", "right", "above", or "bottom". E.g. if an object start from "right" and goes to left, 
          then increment in_counter. Also if given nothing: 
            If line_roi is "horizontal", in_orientation defaults to "right" 
            If line_roi is "vertical", in_orientation defaults to "bottom"

        Attributes:
        - state_history (list): A list of state history. Each state is a list of dictionary, where each dictionary represents
            an one frame passed and its bounding box objects in that frame.
        - fps (int): frame per second
        - line_roi (tuple): The region of interest of the line. structured in line_point_start and line_point_end xyxy coordinates
        - in_orientation (str): The starting orientation of the line.
        """
        self.state_history = []
        self.fps = fps
        self.line_roi = line_roi # region of interest
        self.curr_in_count = 0
        self.curr_out_count = 0

        if get_line_orientation(line_roi) == "vertical":
            if in_orientation == None:
                self.in_orientation = "right" # defaults using right to left flow incounter
            else:
                self.in_orientation = in_orientation 
        else:  # "horizontal"
            if in_orientation == None:
                self.in_orientation = "bottom" # defaults using bottom to above flow incounter
            else:
                self.in_orientation = in_orientation 
    
    def get_final_bboxes(self):
        """
        This function aggregates information from the `state_history` attribute and returns a list of dictionaries representing the 
        final bounding boxes.
        
        The function first initializes an empty list `bboxes` and a set `seen_object_ids` to keep track of the unique object IDs 
        encountered. It then loops through each frame in `state_history` and adds any new object IDs to the `bboxes` list and the 
        `seen_object_ids` set. If there are duplicates, a warning message is printed.
        
        Returns:
            object_id (int): The id of the object
            final_class_label (str): final class label of the object, e.g. "truck"
	        final_class_confidence (float): confidence in the final class label
            final_count (str): final counter for this object, e.g. "in","out","return-in","return-out" 
            timestamps (list of dicts):
                frame_number (int): The number of the frame in which the change in `in_out_counter` was detected
                count (str): The change in `in_out_counter` ('in', 'out', 'return-in' and 'return-out')
                timestamp (str): The timestamp of the event in HHH:MM:SS.sss
            class_labels (list of dicts):
                class_label (str): class label of the object
                class_confidence (float): confidence in the class label
        """
        bboxes = []
        seen_object_ids = set()
        for frame in self.state_history:
            for bbox in frame:
                object_id = bbox['object_id']
                if object_id not in seen_object_ids:
                    bboxes.append({'object_id': object_id, 'timestamps': [], 'class_labels': [], 'final_count': None, 'final_class_label': None, 'final_class_confidence': None})
                    seen_object_ids.add(object_id)
        
        if len(seen_object_ids) != len(bboxes):
            # duplicates found, log a warning or return an error
            print("duplicates found. Length of seen_object_ids: {}, length of bboxes: {}".format(len(seen_object_ids), len(bboxes)))
            pass

        counter_timestamps = self.detect_change_in_out_counter()
        class_labels = self.agg_labels_and_conf_by_obj_id()
        final_class_labels = self.get_highest_confidence()
        final_counts = self.get_final_count()
        
        bboxes_dict = {bbox['object_id']: bbox for bbox in bboxes}
        if counter_timestamps:
            for c_timestamp in counter_timestamps:
                bboxes_dict[c_timestamp['object_id']]['timestamps'].append(c_timestamp)
        
        if class_labels:
            for class_label in class_labels:
                bboxes_dict[class_label['object_id']]['class_labels'].append(class_label)

        if final_counts:
            for final_count in final_counts:
                bboxes_dict[final_count['object_id']]['final_count'] = final_count['count']
        
        if final_class_labels:
            for final_class_label in final_class_labels:
                bboxes_dict[final_class_label['object_id']]['final_class_label'] = final_class_label['class_label']
                bboxes_dict[final_class_label['object_id']]['final_class_confidence'] = final_class_label['class_confidence']
        
        return list(bboxes_dict.values())

    def update_state_tracker_in_out_counter(self):
        ds = self.get_final_count()
        self.curr_in_count = 0
        self.curr_out_count = 0
        for d in ds:
            if d['count'] == "in":
                self.curr_in_count += 1
            elif d['count'] == 'return-in':
                self.curr_in_count -= 1
            elif d['count'] == "out":
                self.curr_out_count += 1
            elif d['count'] == "return-out":
                self.curr_out_count -= 1                


    def get_final_count(self):
        """
        Get the final count state for each object.

        The function takes a list of dictionaries and finds the final 'count' state for each 'object_id' by finding the latest 'frame_number'.

        Returns:
        A list of dictionaries with the corresponding final 'count' state. Each dictionary will contain keys:
            'object_id': an integer representing the object id
            'count': a string corresponding final count state 
            'frame_number':  The number of the frame in which corresponding final count state last changes 
            'timestamp': The timestamp of the event in HHH:MM:SS.sss
        """
        obj_count_dict = {}
        ds = self.detect_change_in_out_counter()
        if not ds:
            return []
        for d in ds:
            obj_id = d['object_id']
            frame_num = d['frame_number']
            if obj_id not in obj_count_dict or frame_num > obj_count_dict[obj_id]['frame_number']:
                obj_count_dict[obj_id] = d
        return list(obj_count_dict.values())


    def get_highest_confidence(self):
        """
        Find the highest confidence value for each object in state_history.
        
        Returns:
        list: List of dictionaries with the highest confidence value for each object. Each dictionary will contain keys:
            'object_id': an integer representing the object id
            'class_label': a string representing the class label of the object
            'class_confidence': a float representing the confidence in the class label
        """
        obj_conf_dict = {}
        ds = self.agg_labels_and_conf_by_obj_id()
        if not ds:
            return []
        for d in ds:
            obj_id = d['object_id']
            conf = d['class_confidence']
            if obj_id not in obj_conf_dict or conf > obj_conf_dict[obj_id]['class_confidence']:
                obj_conf_dict[obj_id] = d
        return list(obj_conf_dict.values())


    def detect_change_in_out_counter(self):
        """
        This function takes a list of `state_history` as input and detects the changes in the `in_out_counter` for each `object_id` 
        in each frame. It returns a list of dictionaries that contains information about the `object_id`, `frame_number`, 
        the change in the `in_out_counter` (i.e., 'in', 'out', 'return-in' and 'return-out'), and the timestamp. 

        Returns:
            result (list): A list of dictionaries, where each dictionary contains the following key-value pairs:
                object_id (int): The id of the object
                frame_number (int): The number of the frame in which the change in `in_out_counter` was detected
                count (str): The change in `in_out_counter` ('in', 'out', 'return-in' and 'return-out')
                timestamp (str): The timestamp of the event in HHH:MM:SS.sss
            
        Example output:
        [
            {
                "object_id": 1,
                "frame_number": 0,
                "count" : "in",
                "timestamp": 001:34:20.123
            },
        ]
        """
        result = []
        for frame_number, frame in enumerate(self.state_history):
            for obj in frame:
                obj_id = obj['object_id']
                count = obj['in_out_counter']
                if frame_number == 0:
                    if count == 1:
                        result.append({'object_id': obj_id, 'frame_number': frame_number, 'count': 'in', 'timestamp': frame_to_timestamp(frame_number, self.fps)})
                    elif count == -1:
                        result.append({'object_id': obj_id, 'frame_number': frame_number, 'count': 'out', 'timestamp': frame_to_timestamp(frame_number, self.fps)})
                else:
                    prev_frame = self.state_history[frame_number - 1]
                    prev_obj = next((x for x in prev_frame if x['object_id'] == obj_id), None)
                    prev_count = prev_obj['in_out_counter'] if prev_obj else 0
                    if prev_count == 0 and count == 1:
                        result.append({'object_id': obj_id, 'frame_number': frame_number, 'count': 'in', 'timestamp': frame_to_timestamp(frame_number, self.fps)})
                    elif prev_count == 1 and count == 0:
                        result.append({'object_id': obj_id, 'frame_number': frame_number, 'count': 'return-in', 'timestamp': frame_to_timestamp(frame_number, self.fps)})
                    elif prev_count == 0 and count == -1:
                        result.append({'object_id': obj_id, 'frame_number': frame_number, 'count': 'out', 'timestamp': frame_to_timestamp(frame_number, self.fps)})
                    elif prev_count == -1 and count == 0:
                        result.append({'object_id': obj_id, 'frame_number': frame_number, 'count': 'return-out', 'timestamp': frame_to_timestamp(frame_number, self.fps)})
        return result


    def agg_labels_and_conf_by_obj_id(self):
        """
        Aggregate object labels and confidence by object ID.

        :Example output:
        [
            {
                "object_id": 1,
                "class_label": "cat",
                "class_confidence": 0.52
            },
        ]
        """
        # Flatten the list of dictionaries
        flat_list = [item for sublist in self.state_history for item in sublist]

        # Create a dictionary to store the aggregated results
        result = {}

        # Iterate over the flat list
        for item in flat_list:
            object_id = item['object_id']
            class_label = item['class_label']
            class_confidence = item['class_confidence']

            # Check if the object ID already exists in the result dictionary
            if object_id in result:
                result[object_id]['class_confidence'] += class_confidence
                result[object_id]['count'] += 1
            else:
                result[object_id] = {
                    'object_id': object_id,
                    'class_label': class_label,
                    'class_confidence': class_confidence,
                    'count': 1
                }

        # Calculate the average class confidence
        for key, value in result.items():
            result[key]['class_confidence'] /= value['count']

        # Convert the result dictionary to a list of dictionaries
        result_list = [value for key, value in result.items()]

        return result_list


    def add_bounding_box(self, object_id, bbox_xyxy, class_label, class_confidence):
        """
        Add an bounding box to the current frame in the state history, and compares it with region of interest
        
        Parameters:
        object_id (int): unique identifier for the object
        bbox_xyxy (tuple): bounding box coordinates in the format (xmin, ymin, xmax, ymax)
        class_label (str): class label for the object
        class_confidence (float): confidence score for the class label

        """
        prev_frame_bbox_states = self.state_history[-2] if self.state_history else []
        prev_frame_bbox_state = next((x for x in prev_frame_bbox_states if x.object_id == object_id), None)


        curr_frame_bbox_state = BBoxState(object_id, bbox_xyxy, class_label, class_confidence)
        curr_frame_bbox_state.orientation_label = check_box_position(curr_frame_bbox_state.bbox_xyxy, self.line_roi)
        

        if prev_frame_bbox_state == None:
            curr_frame_bbox_state.original_orientation_label = curr_frame_bbox_state.orientation_label
            curr_frame_bbox_state.in_out_counter = self.update_in_out_counter(None, curr_frame_bbox_state)
        else:
            curr_frame_bbox_state.original_orientation_label = prev_frame_bbox_state.original_orientation_label
            curr_frame_bbox_state.in_out_counter = self.update_in_out_counter(prev_frame_bbox_state, curr_frame_bbox_state)

        if self.state_history:
            self.state_history[-1].append(curr_frame_bbox_state)
        else:
            self.state_history = [[curr_frame_bbox_state]]
    
    def update_in_out_counter(self, prev_frame_object_state, current_frame_object_state):
        """
        Updates the in/out counter of an object as it moves through the line of interest.
        The in/out counter is updated based on the direction of the object movement and the orientation of the line.

        Parameters:
        - prev_frame_object_state (dict): The state of the object in the previous frame.
        - current_frame_object_state (dict): The state of the object in the current frame.

        Returns:
        int: The updated in/out counter.
        """
        in_orientation = self.in_orientation
        curr_label = current_frame_object_state.orientation_label
        original_orientation_label = current_frame_object_state.original_orientation_label
        if prev_frame_object_state == None:
            prev_label = None
            prev_in_out_counter = 0
        else:
            prev_label = prev_frame_object_state.orientation_label 
            prev_in_out_counter = prev_frame_object_state.in_out_counter

        if prev_label == None:
            return 1 if curr_label == "intersect" else 0

        if prev_label == curr_label:
            return prev_in_out_counter

        if prev_label == "intersect":
            if curr_label == original_orientation_label: 
                return 0
            else:
                return 1 if curr_label != in_orientation else -1

        return 1 if original_orientation_label == in_orientation else -1

    def process_frame(self):
        """
        This method will process the empty frame by appending an empty list to the `state_history` attribute.
        """

        self.state_history.append([])

if __name__ == '__main__':
    import json
    from count_utils import *
    #sys.path.append('/..')
    
    s = StateTracker((10,0,10,50),1,"left")


    s.process_frame()
    s.process_frame()
    s.process_frame()
    s.add_bounding_box(1,[11,11,16,16],'person',0.23)
    s.add_bounding_box(2,[30,30,32,32],'person',0.50)

    s.process_frame()
    s.add_bounding_box(1,[7,7,12,12],'person',0.23)
    s.add_bounding_box(2,[7,7,12,12],'person',0.52)
    
    s.process_frame()
    s.add_bounding_box(1,[11,11,16,16],'person',0.23)
    s.add_bounding_box(2,[30,30,32,32],'person',0.50)
    
    s.process_frame()
    s.add_bounding_box(1,[7,7,12,12],'person',0.23)
    s.add_bounding_box(2,[7,7,12,12],'person',0.52)

    s.process_frame()
    s.add_bounding_box(1,[7,7,12,12],'person',0.23)
    s.add_bounding_box(2,[7,7,12,12],'person',0.52)

    s.process_frame()
    s.add_bounding_box(1,[0,0,5,5],'person',0.23)
    s.add_bounding_box(2,[30,30,32,32],'person',0.55)
    
    x = s.get_final_bboxes()
    json_string = json.dumps(x) 
    print(json_string)