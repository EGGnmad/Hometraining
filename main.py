import cv2
import mediapipe as mp
import numpy as np

window_name = 'Hometraining'

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cnt = 0
next = True

def calculate_angle(a, b):
    a = np.array(a)
    b = np.array(b)

    radians = np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians*180.0/np.pi)

    angle %= 180

    return angle

cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # BGR -> RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False


        # 포즈 정보
        results = pose.process(image)

        # RGB -> BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 랜드 마크 변수에 저장
        try:
            landmarks = results.pose_landmarks.landmark
            window = cv2.getWindowImageRect(window_name)


            lShoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x ,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            lElbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x ,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            # lHip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x ,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            rShoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            rElbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            # rHip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            angle = calculate_angle(lShoulder, lElbow)

            if(angle > 150 and next):
                cnt += 1
                next = False
            elif(angle < 100 and not next):
                next = True

            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]


            cv2.putText(image, str(cnt), np.multiply( nose , [window[2] - 100, window[3] - 500 ]).astype(int), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 117, 66), 5)
        except:
            pass


        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(255, 117, 66), thickness=5, circle_radius=8),
                                  mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
                                  )

        cv2.imshow(window_name, image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()