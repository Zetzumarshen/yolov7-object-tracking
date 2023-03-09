import os
import sys
import json

if __name__ == '__main__':
   
    # NOTE: my vscode project explorer runs at ../.. 
    # Get the absolute path to the parent directory of test_source_processor.py
    test_dir = os.path.dirname(os.path.abspath(__file__))

    # Add the tested directory to the Python path
    tested_dir = os.path.join(test_dir, '..')
    sys.path.append(tested_dir)

    login_json = 'login.json'
    vid_path = 'mb-tiny.mp4'
    #source = vid_path
    #source = 0

    if os.path.exists(login_json):
        with open('login.json', 'r') as f:
            data = json.load(f)
    else:
        raise Exception('user login json not found')
    source = 'rtsp://' + data['upass'] + '@' + data['ip_address']

    # Import util.py from the utils directory
    from processor.webcam import WebcamProcessor

    src = WebcamProcessor(is_download=False,
                          conf_thres=0.5,
                          is_save_txt=True,
                          is_save_conf=True,
                          is_agnostic_nms=True,
                          source=source,
                          name='refactor',
                          is_save_bbox_dim=True,
                          is_view_img=True
                          )

    src.detect()

    print('done testing')
    