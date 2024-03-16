from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Error opening video stream or file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define line points
line_points = [(20, 400), (1080, 400)]

# Video writer
video_writer = cv2.VideoWriter("object_counting_output.mp4",
                       cv2.VideoWriter_fourcc(*'mp4v'),
                       fps,
                       (w, h))
customname = {0: 'hvn'}
print(model.names)
# Init Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True,
                 reg_pts=line_points,
                 classes_names=customname,
                 draw_tracks=True)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
    tracks = model.track(im0, persist=True, show=False)
    im0 = counter.start_counting(im0, tracks)

    cv2.imshow("Processing", im0)

    video_writer.write(im0)

    # q үсэг дээр дарж камерыг хаана
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
video_writer.release()
cv2.destroyAllWindows()
