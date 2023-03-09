import os
import cv2
import time
import torch
from pathlib import Path
from numpy import random
from random import randint

# import torch.backends.cudnn as cudnn
from utils.my_torch_utils import get_torch_backend

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, \
                check_imshow, non_max_suppression, apply_classifier, \
                scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
                increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, \
                time_synchronized, TracedModel
from utils.download_weights import download
from utils.count_utils import check_box_position
from utils.state_tracker import StateTracker
import json

#For SORT tracking
import skimage
from sort import *

class SourceProcessor:
    def __init__(self, weights='yolov7.pt', 
                 source='inference/images', is_download=True,
                 img_size=640, conf_thres=0.25, iou_thres=0.45, 
                 device='', is_view_img=False, is_save_txt=False, 
                 is_save_conf=False, is_nosave=False, classes=None, 
                 is_agnostic_nms=False, augment=False, is_update=False, 
                 project='runs/detect', name='object_tracking', 
                 is_exist_ok=False, is_no_trace=False, in_orientation = "left",
                 is_colored_trk=False, is_save_bbox_dim=False, 
                 is_save_with_object_id=False, line_roi=(500,0,480,800)):
        
        # setting models
        self.weights = weights
        self.source = source
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.classes = classes
        self.agnostic_nms = is_agnostic_nms
        self.augment = augment
        self.update = is_update
        self.project = project
        self.name = name

        # setting state tracker
        self.line_roi = line_roi
        self.in_orientation = in_orientation
        self.download = is_download

        # setting flags
        self.is_exist_ok = is_exist_ok
        self.is_trace = not is_no_trace
        self.is_colored_trk = is_colored_trk
        self.is_save_bbox_dim = is_save_bbox_dim
        self.is_save_with_object_id = is_save_with_object_id
        self.is_view_img = is_view_img
        self.is_save_txt = is_save_txt
        self.is_save_conf = is_save_conf
        self.is_nosave = is_nosave
        self.is_update = is_update

        self.backend = get_torch_backend() 

        if is_download and not os.path.exists(str(weights)):
            print('Model weights not found. Attempting to download now...')
            is_download('./')


    def detect(self):
        with torch.no_grad():
            if self.is_update:  # update all models (to fix SourceChangeWarning)
                for self.weights in ['yolov7.pt']:
                    self._detect()
                    strip_optimizer(opt.weights)
            else:
                self._detect()
    

    def download_weights(self, url, path):
        if os.path.exists(path):
            return

        print(f'Downloading {url} to {path}')
        response = requests.get(url, stream=True)

        # Raise an exception if the response status is not OK
        response.raise_for_status()

        # Write the contents of the response to the file
        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    def draw_boxes(self, img, bbox, identities=None, categories=None, names=None, save_with_object_id=False, path=None,offset=(0, 0)):
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            cat = int(categories[i]) if categories is not None else 0
            id = int(identities[i]) if identities is not None else 0
            data = (int((box[0]+box[2])/2),(int((box[1]+box[3])/2)))
            label = str(id) + ":"+ names[cat]
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,20), 2)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,144,30), -1)
            cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, [255, 255, 255], 1)
            # cv2.circle(img, data, 6, color,-1)   #centroid of box
            txt_str = ""
            if save_with_object_id:
                txt_str += "%i %i %f %f %f %f %f %f" % (
                    id, cat, int(box[0])/img.shape[1], int(box[1])/img.shape[0] , int(box[2])/img.shape[1], int(box[3])/img.shape[0] ,int(box[0] + (box[2] * 0.5))/img.shape[1] ,
                    int(box[1] + (
                        box[3]* 0.5))/img.shape[0])
                txt_str += "\n"
                with open(path + '.txt', 'a') as f:
                    f.write(txt_str)
        return img

    def insert_boxes_to_statetracker(self, statetracker: StateTracker, bbox, identities, categories, names, confidences):
        for i, box in enumerate(bbox):
            #x1, y1, x2, y2 = [int(i) for i in box]
            cat = int(categories[i]) if categories is not None else 0
            id = int(identities[i]) if identities is not None else 0
            conf = confidences[i] if confidences is not None else 0
            statetracker.add_bounding_box(id, box, names[cat], conf)

    def draw_in_out_counter(self, img, in_count=0, out_count=0):
        # get image height and width
        # height, width = img.shape[:2]
        
        # set font type, font scale and thickness
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        
        # set text color and position
        text_color = (255, 255, 255) # blue
        text_position_in = (20, 20)
        text_position_out = (20, 50)
        
        # draw in_count text on the image
        cv2.putText(img, "In Count: {}".format(in_count), text_position_in, font, 
                    font_scale, text_color, thickness, cv2.LINE_AA)
        
        # draw out_count text on the image
        cv2.putText(img, "Out Count: {}".format(out_count), text_position_out, font, 
                    font_scale, text_color, thickness, cv2.LINE_AA)
        
        return img

    def draw_lines(self, img, bboxes, line):
        x1, y1, x2, y2 = line
        
        # check if bbox intersects line
        is_intersect = False
        for box in bboxes:
            if check_box_position(box, line) == "intersect":
                is_intersect = True
                break
        
        # draw a line for region of interest
        start_point = (x1, y1)
        end_point = (x2, y2)
        if is_intersect:
            cv2.line(img, start_point, end_point, (255,0,0), 4)
        else:
            cv2.line(img, start_point, end_point, (122,255,0), 4)

    def init_sort(self, sort_max_age=5, sort_min_hits=2, sort_iou_thresh=0.2):
        return Sort(max_age=sort_max_age, min_hits=sort_min_hits, iou_threshold=sort_iou_thresh)
    
    def color_tracks(random=True):
        rand_color_list = []
        if random:
            for i in range(0,5005):
                r = randint(0, 255)
                g = randint(0, 255)
                b = randint(0, 255)
                rand_color = (r, g, b)
                rand_color_list.append(rand_color)
            return rand_color_list
        else:
            raise ValueError("color_tracks() for nonrandom color is not implemented yet.")

    def init_device(self, device):
        set_logging()
        device = select_device(device)
        half = device.type != 'cpu'  # half precision only supported on CUDA
        return device, half
    
    def create_directory(self, project, name, exist_ok, save_txt, save_with_object_id):
        save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))  # increment run
        (save_dir / 'labels' if save_txt or save_with_object_id else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        return save_dir

    def load_model(self, device, weights, imgsz, trace, half):
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if trace:
            model = TracedModel(model, device, imgsz)
        if half:
            model.half()  # to FP16
        return model, stride

    def set_dataloader(self, webcam, source, imgsz, stride, backend):
        if webcam:
            check_imshow()
            backend.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            # NOTE: loadImages can load a couple videos at once, it might have different fps and breakthings
            dataset = LoadImages(source, img_size=imgsz, stride=stride)
        return dataset

    def init_gpu(self, device, imgsz, model):
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    def preprocess_image(self, half, device, img):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

    def set_save_path(self, p, save_dir, dataset, frame):
        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)  # img.jpg
        txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
        return p, save_path, txt_path

    def draw_colored_track(self, colored_trk, im0, track, rand_color_list):
        if colored_trk:
            [cv2.line(im0, (int(track.centroidarr[i][0]),
                        int(track.centroidarr[i][1])), 
                        (int(track.centroidarr[i+1][0]),
                        int(track.centroidarr[i+1][1])),
                        rand_color_list[track.id], thickness=2) 
                        for i,_ in  enumerate(track.centroidarr) 
                        if i < len(track.centroidarr)-1 ] 
        #draw same color tracks
        else:
            [cv2.line(im0, (int(track.centroidarr[i][0]),
                        int(track.centroidarr[i][1])), 
                        (int(track.centroidarr[i+1][0]),
                        int(track.centroidarr[i+1][1])),
                        (255,0,0), thickness=2) 
                        for i,_ in  enumerate(track.centroidarr) 
                        if i < len(track.centroidarr)-1 ] 
            
    def update_coordinates_info(self, txt_str, save_bbox_dim,track, im0):
        # Normalize coordinates
        txt_str += "%i %i %f %f" % (track.id, track.detclass, track.centroidarr[-1][0] / im0.shape[1], track.centroidarr[-1][1] / im0.shape[0])
        if save_bbox_dim:
            txt_str += " %f %f" % (np.abs(track.bbox_history[-1][0] - track.bbox_history[-1][2]) / im0.shape[0], np.abs(track.bbox_history[-1][1] - track.bbox_history[-1][3]) / im0.shape[1])
        txt_str += "\n"

    def save_result(self, save_img, dataset, save_path, im0, vid_cap, vid_path):
        vid_writer = None
        if save_img:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
                print(f" The image with the result is saved in: {save_path}")
            else:  # 'video' or 'stream'
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

    def _detect(self, is_save_img=False):
        source = self.source
        weights = self.weights 
        imgsz = self.img_size 
        trace = self.is_trace 
        colored_trk = self.is_colored_trk 
        weights = self.weights 
        project = self.project
        name = self.name
        line_roi = self.line_roi
        in_orientation = self.in_orientation
        device = self.device
        backend = self.backend
        augment = self.augment
        conf_thres = self.conf_thres 
        iou_thres = self.iou_thres
        classes = self.classes

        # setting flags
        is_save_txt = self.is_save_txt
        is_save_with_object_id = self.is_save_with_object_id 
        is_save_bbox_dim = self.is_save_bbox_dim 
        is_exist_ok = self.is_exist_ok
        is_view_img = self.is_view_img 
        is_agnostic_nms = self.agnostic_nms
        is_nosave = self.is_nosave

        # allocate variable
        vid_path, vid_writer, stride, dataset = None, None, None, None
    
        is_save_img = not is_nosave and not source.endswith('.txt')  # save inference images
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        #.... Initialize SORT .... 
        sort_tracker = self.init_sort(5,2,0.2)

        #........Rand Color for every trk.......
        rand_color_list = self.color_tracks()

        # Directories
        save_dir = self.create_directory(project, name, is_exist_ok, is_save_txt, is_save_with_object_id)

        # Initialize device
        device, half = self.init_device(device)

        # Load model
        model, stride = self.load_model(device, weights, imgsz, trace, half)

        # Set Dataloader
        dataset = self.set_dataloader(webcam, source, imgsz, stride, backend)

        # Initiate statetracker
        statetracker = StateTracker(line_roi, dataset.get_fps(), in_orientation )

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names

        # init GPU and allocate memory
        self.init_gpu(device, imgsz, model)

        # set the inference image into 1:1    
        old_img_w = old_img_h = imgsz

        # set batch size = 1
        old_img_b = 1

        # set timer
        t0 = time.time()

        # Run inference
        for path, img, im0s, vid_cap in dataset:

            # preprocess image
            #self.preprocess_image(half, device, img)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)


            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment)[0]

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, is_agnostic_nms)
            t3 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                # save labels
                p, save_path, txt_path = self.set_save_path(p, save_dir, dataset, frame)
                
                # normalization gain whwh
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

                # advance statetracker frame
                statetracker.process_frame()

                # draw counter
                statetracker.update_state_tracker_in_out_counter()
                self.draw_in_out_counter(im0, statetracker.curr_in_count, statetracker.curr_out_count)

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    #..................USE TRACK FUNCTION....................
                    dets_to_sort = np.empty((0,6))
                    
                    # NOTE: detected object class in detclass
                    for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                        # NOTE: dets_to_sort structure [x1, y1, x2, y2, conf, detclass]
                        dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, conf, detclass])))
                    
                    # Run SORT
                    # NOTE: tracked_dets structure: [x1,y1,x2,y2,0,object_id]
                    tracked_dets = sort_tracker.update(dets_to_sort)
                    tracks = sort_tracker.getTrackers()

                    # initialize txt string to save
                    txt_str = ""

                    #loop over tracks
                    for track in tracks:

                        # draw tracks
                        self.draw_colored_track(colored_trk, im0, track, rand_color_list)
                        
                        # prepare text if save_txt
                        if is_save_txt and not is_save_with_object_id:
                            self.update_coordinates_info(txt_str, is_save_bbox_dim, track, im0)

                    # write coordinates info into file
                    if is_save_txt and not is_save_with_object_id:
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(txt_str)  

                    # draw boxes for visualization
                    if len(tracked_dets)>0:
                        bbox_xyxy = tracked_dets[:, :4]
                        identities = tracked_dets[:, 8]
                        categories = tracked_dets[:, 4]
                        confidences = dets_to_sort[:, 4]
                        self.draw_boxes(im0, bbox_xyxy, identities, categories, names, is_save_with_object_id, txt_path)
                        self.draw_lines(im0, bbox_xyxy, line_roi)
                        self.insert_boxes_to_statetracker(statetracker, bbox_xyxy, identities, categories, names, confidences)    
                # End processing if there are at least one detection ...................................

                # Print time (inference + NMS)
                print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
                
                # Stream results
                if is_view_img:
                    cv2.imshow(str(p), im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        cv2.destroyAllWindows()
                        raise StopIteration

                # Save results (image with detections)
                self.save_result(is_save_img, dataset, save_path, im0, vid_cap, vid_path)

            # End processing one image of a video ...................................

        if is_save_txt or is_save_img or is_save_with_object_id:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if is_save_txt else ''
            #print(f"Results saved to {save_dir}{s}")

        json_data = json.dumps(statetracker.get_final_bboxes())
        with open("data.json","w") as file:
            file.write(json_data)

        print(f'Done. ({time.time() - t0:.3f}s)')   
        # End processing a video or all image ...................................