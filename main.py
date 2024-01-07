from ultralytics import YOLO
import cv2

#load models

coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./license_plate_detector.pt')

#load video
cap = cv2.VideoCapture('./sample.mp4')
vehicles = [2,3,5,7]
#read frames
frame_nmr = -1


ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret and frame_nmr <10:
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id =  detection
            if int(class_id) in vehicles:
                detections_.append ([x1,y1, x2, y2, score])
                pass
        
# track vehicles
