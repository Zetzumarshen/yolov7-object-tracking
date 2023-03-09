if __name__ == '__main__':
    import json
    from utils.state_tracker import StateTracker
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
    s.process_frame()
    s.process_frame()
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