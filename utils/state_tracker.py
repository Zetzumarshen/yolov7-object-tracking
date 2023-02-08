from utils.count_utils import check_box_position, frame_to_timestamp

class ObjectState:
    def __init__(self, object_id, class_label, orientation_label, in_out_counter):
        self.object_id = object_id
        self.class_label = class_label
        self.orientation_label = orientation_label
        self.in_out_counter = in_out_counter 
        
class StateTracker:
    def __init__(self):
        self.state_history = []
 
    def get_current_frame_objects(self):
        return self.state_history[-1] if self.state_history else []
 
    def get_previous_frame_objects(self):
        return self.state_history[-2] if len(self.state_history) > 1 else []
 
    def add_object(self, obj):
        if self.state_history:
            self.state_history[-1].append(obj)
        else:
            self.state_history.append([obj])
    
    def get_current_frame_number(self):
        return len(self.state_history) - 1

    def get_previous_frame_number(self):
        return len(self.state_history) - 2