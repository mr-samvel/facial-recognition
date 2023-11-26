from deepface import DeepFace
from threading import Thread, Lock
import cv2

IS_FACE_MATCHED = False
MATCHED_FACE_COORDS = None

def is_input_quit():
    key = cv2.waitKey(1)
    return key == ord('q')

def setup_cam():
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cam

def check_face(frame, reference_img):
    global IS_FACE_MATCHED, MATCHED_FACE_COORDS
    try:
        result = DeepFace.verify(frame, reference_img)
        if result['verified']:
            IS_FACE_MATCHED = True
            MATCHED_FACE_COORDS = result['facial_areas']['img1']
        else:
            IS_FACE_MATCHED = False
            MATCHED_FACE_COORDS = None
    except ValueError:
        IS_FACE_MATCHED = False
        MATCHED_FACE_COORDS = None

cam = setup_cam()
reference_img = cv2.imread('references/eu.jpg')
frame_counter = 0
update_on_frame_count = 30

while True:
    captured, frame = cam.read()
    if not captured:
        if is_input_quit(): 
            break
        continue
    
    frame_counter += 1
    
    if frame_counter % update_on_frame_count == 0:
        try:
            Thread(target=check_face, args=(frame.copy(), reference_img.copy())).start()
        except ValueError:
            pass
    
    if IS_FACE_MATCHED:
        cv2.putText(frame, "match", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if MATCHED_FACE_COORDS is not None:
            x, y, w, h = MATCHED_FACE_COORDS.values()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        cv2.putText(frame, "no match", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("camera", frame)

    if is_input_quit(): 
        break

cv2.destroyAllWindows()