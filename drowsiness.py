import cv2
from scipy.spatial import distance
import dlib
import imutils
from imutils import face_utils

#to determine the ratio of closure of eye

def ratio_eye(eye):
    p1 = distance.euclidean(eye[1], eye[5])
    p2 = distance.euclidean(eye[2], eye[4])
    p3 = distance.euclidean(eye[0], eye[3])
    aspect_ratio = (p1 + p2 )/(2.0 * p3)
    return aspect_ratio

#it is the minm cut off value for eye to be considered open
threshold_value=0.25
#no of consecutive frames to be checked to ensure that  eye is open/close
no_of_frames = 20
#detect the face in frame
face_detection =dlib.get_frontal_face_detector()
#face shape prediction
predict = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
(left_start, left_end) = face_utils.FACIAL_LANDMARKS_68_IDXS['left_eye']
(right_start, right_end) = face_utils.FACIAL_LANDMARKS_68_IDXS['right_eye']

#webcam on
cap = cv2.VideoCapture(0)
flag = 0

while True:
    ret, frame =cap.read()
    frame = imutils.resize(frame, width=450)
    #gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = face_detection(gray, 0)
    for subject in subjects:
        #detecting the face in frame
        shape = predict(gray,subject)
        #covert in numerical array
        shape = face_utils.shape_to_np(shape)
        #Extracting the eyes from the shape
        leftEye = shape[left_start:left_end]
        rightEye = shape[right_start:right_end]
        #calculating aspect ratios
        left_aspect_ratio = ratio_eye(leftEye)
        right_aspect_ratio = ratio_eye(rightEye)
        net_aspect_ratio = (left_aspect_ratio+right_aspect_ratio)/2.0
        #draw contours
        left_hull = cv2.convexHull(leftEye)
        right_hull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [left_hull], -1, (0,255,255), 1)
        cv2.drawContours(frame, [right_hull], -1, (0, 255, 255), 1)
        if net_aspect_ratio < threshold_value:
            flag +=1
            #we are checking for 20 consecutive frames.
            if flag > no_of_frames:
                cv2.putText(frame, "Drowsy!!!!", (10, 30), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1.4, (255, 0, 255), 1)
                cv2.putText(frame, "Drowsy", (10, 325), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1.4, (255, 0, 255), 1)
        else:
            #streak of flag is destroyed
            flag = 0
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        #quit
        break
cv2.destroyAllWindows()
cap.release()
