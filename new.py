import cv2
import dlib
import time
from imutils import face_utils
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
from datetime import datetime
import tkinter as tk
from tkinter import messagebox

cam = cv2.VideoCapture(0)
# Defined the constants for fatigue detection
FATIGUE_DURATION_THRESHOLD = 1.5
FATIGUE_BLINK_FREQUENCY_THRESHOLD = 10
eye_closed = False
isFatigue = False
#------------Variables---------#
blink_thresh = 0.5
tt_frame = 3
blink_count = 0
count = 0
avg_values = []
timestamps = []

#------#
detector = dlib.get_frontal_face_detector()
lm_model = dlib.shape_predictor('Model\shape_predictor_68_face_landmarks.dat')

#--Eye ids ---#
(L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

ptime = 0

def EAR_cal(eye):
    #----vertical-#
    v1 = dist.euclidean(eye[1], eye[5])
    v2 = dist.euclidean(eye[2], eye[4])

    #-------horizontal----#
    h1 = dist.euclidean(eye[0], eye[3])

    ear = (v1 + v2) / h1
    return ear

# Helper function to display text on OpenCV window
def put_text(image, text, position, color=(0, 0, 255), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2):
    cv2.putText(image, text, position, font, font_scale, color, thickness)

#funtion to show alert box of fatigue detected
# Function to display alert box
def show_alert():
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("Alert", "Fatigue Detected!")
    root.destroy()

while True:
    if cam.get(cv2.CAP_PROP_POS_FRAMES) == cam.get(cv2.CAP_PROP_FRAME_COUNT):
        cam.set(cv2.CAP_PROP_POS_FRAMES, 0)

    _, frame = cam.read()
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #--------fps --------#
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    put_text(frame, f'FPS:{int(fps)}', (50, 50))

    #-----facedetection----#
    faces = detector(img_gray)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200), 2)

        #---------Landmarks------#
        shapes = lm_model(img_gray, face)
        shape = face_utils.shape_to_np(shapes)

        #-----Eye landmarks---#
        lefteye = shape[L_start:L_end]
        righteye = shape[R_start:R_end]

        for Lpt, rpt in zip(lefteye, righteye):
            cv2.circle(frame, Lpt, 2, (200, 200, 0), 2)
            cv2.circle(frame, rpt, 2, (200, 200, 0), 2)

        left_EAR = EAR_cal(lefteye)
        right_EAR = EAR_cal(righteye)

        avg = (left_EAR + right_EAR) / 2
        avg_values.append(avg)
        current_time = datetime.now()
        timestamps.append(current_time)

        # Eye fatigue detection
        if avg < blink_thresh:
            count += 1

        else:
            if count > tt_frame:
                blink_count += 1
                count = 0
            else:
                count = 0

        # Check for eye fatigue
        if blink_count >= FATIGUE_BLINK_FREQUENCY_THRESHOLD:
            isFatigue = True
            
        # Eye closure detection
        if avg < blink_thresh:
            if not eye_closed:
                eye_closed = True
                last_eye_closed_time = time.time()
                eye_open_time = None
                
        else:
            if eye_closed:
                eye_closed = False
                eye_open_time = time.time()

                # Calculate eye closure duration
                eye_closure_duration = eye_open_time - last_eye_closed_time

                # Check if eye closure duration exceeds the threshold
                if eye_closure_duration > FATIGUE_DURATION_THRESHOLD:
                    isFatigue = True
                    

    if isFatigue:
        cv2.putText(frame, "Fatigue Detected", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # show_alert()
                          

    put_text(frame, f'Blink Count: {blink_count}', (50, 100))

    frame = cv2.resize(frame, (1080, 640))
    cv2.imshow("Video", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Plotting the graph
plt.plot(timestamps, avg_values)
plt.xlabel('Time')
plt.ylabel('Average EAR')
plt.title('Average EAR Over Time')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot as an image file in your project folder
plt.savefig('Eye-Blink-Detector\\outputs\\output-graph.png')

# Display the plot
plt.show()

cam.release()
