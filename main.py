import cv2
from motpy import Detection, MultiObjectTracker
import torch


def draw_boxes(frame, track_results):
    # Draw bounding boxes for tracked objects
    for object in track_results:
        x, y, w, h = object.box
        x, y, w, h = int(x), int(y), int(w), int(h)
        object_id = object.id
        confidence = object.score
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
        cv2.putText(frame, f"{str(id_dict[object_id])}: {str(round(confidence, 2))}", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.putText(frame, "People Count: {}".format(len(track_results)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


def update_id_dict(id_dict, j, track_results):
    # Update the dictionary with new object IDs and corresponding labels
    for track_result in track_results:
        if track_result.id not in id_dict:
            id_dict[track_result.id] = j
            j += 1
    return id_dict, j


if __name__ == '__main__':
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Open video file
    cap = cv2.VideoCapture('video1.mp4')
    cap_fps = cap.get(cv2.CAP_PROP_FPS)

    cv2.namedWindow('FRAME')

    # Initialize MultiObjectTracker
    tracker = MultiObjectTracker(dt=1 / cap_fps, tracker_kwargs={'max_staleness': 10})

    # Initialize ID dictionary and counter
    id_dict = {}
    j = 0

    while True:
        ret, frame = cap.read()

        frame = cv2.resize(frame, (1020, 500))
        results = model(frame)
        output = results.pandas().xyxy[0]

        # Filter objects with label "person"
        objects = output[output['name'] == 'person']

        detections = []

        # Pass YOLO detections to motpy
        for index, obj in objects.iterrows():
            coordinates = [int(obj['xmin']), int(obj['ymin']), int(obj['xmax']), int(obj['ymax'])]
            detections.append(Detection(box=coordinates, score=obj['confidence'], class_id=obj['class']))

        # Perform object tracking
        tracker.step(detections=detections)
        track_results = tracker.active_tracks()

        # Update ID dictionary
        id_dict, j = update_id_dict(id_dict, j, track_results)

        # Draw bounding boxes on frame
        draw_boxes(frame, track_results)
        cv2.imshow('FRAME', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
