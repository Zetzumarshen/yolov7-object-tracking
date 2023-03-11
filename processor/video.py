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
from utils.datasets import LoadStreams, LoadImages, LoadWebcam
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
import datetime

#For SORT tracking
import skimage
from sort import *


class VideoProcessor:
    def __init__(self, master=None, weights='yolov7.pt', 
                 source='inference/images', is_download=True,
                 img_size=640, conf_thres=0.25, iou_thres=0.45, 
                 device='', is_view_img=False, is_save_txt=False, 
                 is_save_conf=False, is_nosave=False, classes=None, 
                 is_agnostic_nms=False, augment=False, is_update=False, 
                 project='runs/detect', name='object_tracking', 
                 is_exist_ok=False, is_no_trace=False, in_orientation = "right",
                 is_colored_trk=False, is_save_bbox_dim=False, is_video_player = True,
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
        self.master = master

        # setting state tracker
        self.line_roi = line_roi
        self.in_orientation = in_orientation
        self.is_download = is_download
        self.fps = None 

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
        self.is_statetracking = False
        self.is_video_player = is_video_player

        with torch.no_grad():
            self.save_dir = self.get_save_dir(self.project, self.name, self.is_exist_ok, self.is_save_txt, self.is_save_with_object_id)
            self.old_img_w = self.old_img_h = img_size
            self.old_img_b = 1
            self.source_type = self.get_source_type(self.source)
            self.backend = get_torch_backend() 
            self.is_save_img = not is_nosave # save inference images
            self.vid_path, self.vid_writer, self.stride, self.dataset = None, None, None, None
            self.sort_tracker = self.init_sort(5,2,0.2)
            self.rand_color_list = self.color_tracks()
            self.device, self.half = self.init_device(self.device)
            self.model, self.stride = self.load_model(self.device, self.weights, self.img_size, self.is_trace, self.half)
            self.dataset = self.get_dataloader(self.source, self.source_type, self.img_size, self.stride, self.backend)
            self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            self.width = self.dataset.width
            self.height = self.dataset.height
            self.cap = self.dataset.cap
            self.fps = self.dataset.fps
            self.init_gpu(self.device, img_size, self.model)

        if is_download and not os.path.exists(str(weights)):
            print('Model weights not found. Attempting to download now...')
            download('./')

    def process_frame(self):
        with torch.no_grad():
            if self.is_update:  # update all models (to fix SourceChangeWarning)
                for self.weights in ['yolov7.pt']:
                    im0 = self._detect()
                    strip_optimizer(self.weights)
            else:
                im0 = self._detect()
        return  cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
    
    def get_current_timestamp(self):
        """Returns the current frame timestamp in seconds."""
        current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        return current_frame / self.fps

    def seek_to_timestamp(self, timestamp):
        """Seeks to the specified timestamp in seconds."""
        frame_number = int(timestamp * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def get_duration(self):
        """Returns the duration of the video in seconds."""
        num_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return num_frames / self.fps

    def start_state_tracker(self):
        self.is_statetracking = True
        self.statetracker = StateTracker(self.line_roi, self.dataset.get_fps(), self.in_orientation, master=self, is_webcam= not self.is_video_player)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.master.update_message(f"[{current_time}] ====START COUNTING====")

    def counts_changed(self):
        ls = self.statetracker.detect_change_in_out_counter()
        if len(ls) > 0:
            log_dict = ls[-1]
            log_str =  f"[{log_dict['timestamp']}] {log_dict['object_id']}:{log_dict['class_label']}({log_dict['class_confidence']:.0%})  Moving {log_dict['count']}"
            self.master.update_message(log_str)
            print(log_str)

    def stop_state_tracking(self):
        self.is_statetracking = False
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.master.update_message(f"[{current_time}] ====END COUNTING====")
        self.master.update_message(self.get_summary())
        
    def get_summary(self):
        summary_str = "Vehicle counts: \n"
        summary_str += f"{self.statetracker.truck_in_count} trucks in, \n"
        summary_str += f"{self.statetracker.truck_out_count} trucks out, \n"
        summary_str += f"{self.statetracker.car_in_count} cars in, \n"
        summary_str += f"{self.statetracker.car_out_count} cars out, \n"
        summary_str += f"{self.statetracker.motorcycle_in_count} motorcycles in, \n"
        summary_str += f"{self.statetracker.motorcycle_out_count} motorcycles out, \n"
        summary_str += f"{self.statetracker.bus_in_count} buses in, \n"
        summary_str += f"{self.statetracker.bus_out_count} buses out.\n"
        summary_str += f"============================"
        return summary_str

    def _detect(self):
        is_colored_trk = self.is_colored_trk 
        line_roi = self.line_roi
        self.device = self.device
        augment = self.augment
        conf_thres = self.conf_thres 
        iou_thres = self.iou_thres
        classes = self.classes

        # setting flags
        is_save_txt = self.is_save_txt
        is_save_with_object_id = self.is_save_with_object_id 
        is_save_bbox_dim = self.is_save_bbox_dim 
        is_view_img = self.is_view_img 
        is_agnostic_nms = self.agnostic_nms

        # set timer
        t0 = time.time()

        # Run inference for one video
        for path, img, im0s, vid_cap in self.dataset:

            # setting video capture
            self.cap = vid_cap
            self.fps = vid_cap.get(cv2.CAP_PROP_FPS)

            # preprocess image
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if self.device.type != 'cpu' and (self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
                self.old_img_b = img.shape[0]
                self.old_img_h = img.shape[2]
                self.old_img_w = img.shape[3]
                for i in range(3):
                    self.model(img, augment)[0]

            # Inference
            t1 = time_synchronized()
            pred = self.model(img, augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, is_agnostic_nms)
            t3 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, '', im0s, getattr(self.dataset, 'frame', 0)

                # save labels
                p, save_path, txt_path = self.set_save_path(p, self.save_dir, self.dataset, frame)
                
                # normalization gain whwh
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

                # advance statetracker frame
                if self.is_statetracking == True:
                    self.statetracker.process_frame()

                # init bbox
                bbox_xyxy = []

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    #..................USE TRACK FUNCTION....................
                    dets_to_sort = np.empty((0,6))
                    
                    # NOTE: detected object class in detclass
                    for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                        # NOTE: dets_to_sort structure [x1, y1, x2, y2, conf, detclass]
                        dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, conf, detclass])))
                    
                    # Run SORT
                    # NOTE: tracked_dets structure: [x1,y1,x2,y2,0,object_id]
                    tracked_dets = self.sort_tracker.update(dets_to_sort)
                    tracks = self.sort_tracker.getTrackers()

                    # initialize txt string to save
                    txt_str = ""

                    #loop over tracks
                    for track in tracks:

                        # draw tracks
                        self.draw_colored_track(is_colored_trk, im0, track, self.rand_color_list)
                        
                        # prepare text if save_txt
                        if is_save_txt and not is_save_with_object_id:
                            txt_str = self.get_coordinates_info(txt_str, is_save_bbox_dim, track, im0)

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
                        self.draw_boxes(im0, bbox_xyxy, identities, categories, self.names, is_save_with_object_id)
                        if self.is_statetracking == True:
                            self.insert_boxes_to_statetracker(self.statetracker, bbox_xyxy, identities, categories, self.names, confidences)    
                # End processing if there are at least one detection ...................................

                # draw line
                self.draw_lines(im0, bbox_xyxy, line_roi)

                # draw counter
                if self.is_statetracking == True:
                    self.statetracker.update_state_tracker_in_out_counter()
                    self.draw_in_out_counter(im0, self.statetracker)

                # Print time (inference + NMS)
                print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                return im0
            # End processing one image of a video ...................................



    def draw_boxes(self, img, bbox, identities=None, categories=None, names=None, save_with_object_id=False, path=None, offset=(0, 0)):
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

    def draw_in_out_counter(self, img, statetracker):
        # set font type, font scale and thickness
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # set text color and position
        text_color = (255, 255, 255) # white
        text_position_truck_in = (20, 20)
        text_position_truck_out = (20, 35)
        
        cv2.putText(img, "Truck In: {}".format(statetracker.truck_in_count), text_position_truck_in, font, 
                    font_scale, text_color, thickness, cv2.LINE_AA)
        
        cv2.putText(img, "Truck Out: {}".format(statetracker.truck_out_count), text_position_truck_out, font, 
                    font_scale, text_color, thickness, cv2.LINE_AA)
        
        text_position_car_in = (20, 50)
        text_position_car_out = (20, 65)
        
        cv2.putText(img, "Car In: {}".format(statetracker.car_in_count), text_position_car_in, font, 
                    font_scale, text_color, thickness, cv2.LINE_AA)
        
        cv2.putText(img, "Car Out: {}".format(statetracker.car_out_count), text_position_car_out, font, 
                    font_scale, text_color, thickness, cv2.LINE_AA)

        text_position_motorcycle_in = (20, 80)
        text_position_motorcycle_out = (20, 95)
        
        cv2.putText(img, "Motorcycle In: {}".format(statetracker.motorcycle_in_count), text_position_motorcycle_in, font, 
                    font_scale, text_color, thickness, cv2.LINE_AA)
        
        cv2.putText(img, "Motorcycle Out: {}".format(statetracker.motorcycle_out_count), text_position_motorcycle_out, font, 
                    font_scale, text_color, thickness, cv2.LINE_AA)
        
        text_position_bus_in = (20, 110)
        text_position_bus_out = (20, 125)
        
        cv2.putText(img, "bus In: {}".format(statetracker.bus_in_count), text_position_bus_in, font, 
                    font_scale, text_color, thickness, cv2.LINE_AA)
        
        cv2.putText(img, "bus Out: {}".format(statetracker.bus_out_count), text_position_bus_out, font, 
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
    
    def get_save_dir(self, project, name, exist_ok, save_txt, save_with_object_id):
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

    def get_dataloader(self, source, source_type, imgsz, stride, backend):
        if source_type == 'stream':
            check_imshow()
            backend.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        elif source_type == 'image':
            # NOTE: loadImages can load a couple videos at once, it might have different fps and breakthings
            dataset = LoadImages(source, img_size=imgsz, stride=stride)
        elif source_type == 'webcam':
            check_imshow()
            backend.benchmark = True
            dataset = LoadWebcam(source, imgsz, stride)

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
            
    def get_coordinates_info(self, txt_str, save_bbox_dim,track, im0):
        # Normalize coordinates
        txt_str += "%i %i %f %f" % (track.id, track.detclass, track.centroidarr[-1][0] / im0.shape[1], track.centroidarr[-1][1] / im0.shape[0])
        if save_bbox_dim:
            txt_str += " %f %f" % (np.abs(track.bbox_history[-1][0] - track.bbox_history[-1][2]) / im0.shape[0], np.abs(track.bbox_history[-1][1] - track.bbox_history[-1][3]) / im0.shape[1])
        txt_str += "\n"
        return txt_str

    def save_result(self, source_type, save_path, im0, vid_cap, vid_path):
        vid_writer = None        
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

    def get_source_type(self, source):
        if str(source).isnumeric() or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
            return 'webcam' # 0 is pipe
        elif os.path.isfile(source) or source.endswith('.txt'):
            return 'image'
        else:
            raise Exception('Source unhandled')
        # End processing a video or all image ...................................