import cv2
import mediapipe as mp
import serial
import time
from scipy.spatial import distance as dist

# 1. 아두이노 시리얼 통신 설정 (본인 포트에 맞게 수정)
arduino_port = '/dev/ttyACM0'  # 리눅스/우분투의 경우 '/dev/ttyACM0' 등
try:
    py_serial = serial.Serial(port=arduino_port, baudrate=9600, timeout=1)
    print(f"아두이노 연결 성공 ({arduino_port})")
except:
    print(f"아두이노 연결 실패. 포트({arduino_port})를 확인하세요.")
    py_serial = None

# 2. 비율 계산 함수
def calculate_ratio(pts_vertical1, pts_vertical2, pts_horizontal1, pts_horizontal2):
    A = dist.euclidean(pts_vertical1, pts_vertical2)
    B = dist.euclidean(pts_horizontal1, pts_horizontal2)
    return A / B

# 3. MediaPipe 및 카메라 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
cap = cv2.VideoCapture(0)

# 매개변수 설정
EAR_THRESHOLD = 0.22     # 눈 감김 임계값
MAR_THRESHOLD = 0.50     # 입 벌림(하품) 임계값
CLOSED_SECONDS = 1.5     # 감지 지속 시간 (초)

# 현재 경고 상태 저장 변수 (0: 정상, 1: 졸음, 2: 하품)
current_alert_state = '0'

closed_start_time = None
yawn_start_time = None

# MediaPipe Face Mesh 기준 주요 랜드마크 인덱스
LEFT_EYE =
RIGHT_EYE =

MOUTH_TOP = 13          
MOUTH_BOTTOM = 14       
MOUTH_LEFT = 78         
MOUTH_RIGHT = 308       
MOUTH_ALL = 

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            
            all_x = [int(lm.x * w) for lm in face_landmarks.landmark]
            all_y = [int(lm.y * h) for lm in face_landmarks.landmark]
            face_x1, face_y1 = min(all_x), min(all_y)
            face_x2, face_y2 = max(all_x), max(all_y)
            box_color = (0, 255, 0) 
            
            def get_pt(idx):
                return (int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h))
            
            def get_pts(indices):
                return [get_pt(i) for i in indices]

            # 눈 EAR 계산
            left_eye_pts = get_pts(LEFT_EYE)
            right_eye_pts = get_pts(RIGHT_EYE)
            left_ear = (dist.euclidean(left_eye_pts, left_eye_pts) + dist.euclidean(left_eye_pts, left_eye_pts)) / (2.0 * dist.euclidean(left_eye_pts, left_eye_pts))
            right_ear = (dist.euclidean(right_eye_pts, right_eye_pts) + dist.euclidean(right_eye_pts, right_eye_pts)) / (2.0 * dist.euclidean(right_eye_pts, right_eye_pts))
            avg_ear = (left_ear + right_ear) / 2.0

            # 입 MAR 계산
            p_top, p_bottom = get_pt(MOUTH_TOP), get_pt(MOUTH_BOTTOM)
            p_left, p_right = get_pt(MOUTH_LEFT), get_pt(MOUTH_RIGHT)
            mar = dist.euclidean(p_top, p_bottom) / dist.euclidean(p_left, p_right)

            # 졸음 및 하품 감지 체크
            drowsy_detected = False
            if avg_ear < EAR_THRESHOLD:
                if closed_start_time is None:
                    closed_start_time = time.time()
                elif time.time() - closed_start_time >= CLOSED_SECONDS:
                    drowsy_detected = True
            else:
                closed_start_time = None

            yawn_detected = False
            if mar > MAR_THRESHOLD:
                if yawn_start_time is None:
                    yawn_start_time = time.time()
                elif time.time() - yawn_start_time >= CLOSED_SECONDS:
                    yawn_detected = True
            else:
                yawn_start_time = None

            # 🚨 아두이노 LED 신호 전송 제어 로직
            target_state = '0' # 기본 정상
            if drowsy_detected:
                target_state = '1' # 졸음 상태 신호
                box_color = (0, 0, 255)
                cv2.putText(frame, "DROWSINESS ALERT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            elif yawn_detected:
                target_state = '2' # 하품 상태 신호
                box_color = (0, 165, 255)
                cv2.putText(frame, "YAWNING ALERT!", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)

            # 상태가 변경되었을 때만 아두이노에 시리얼 신호 전송 (통신 부하 감소)
            if py_serial and target_state != current_alert_state:
                if target_state == '0':
                    py_serial.write(b'0')
                elif target_state == '1':
                    py_serial.write(b'1')
                elif target_state == '2':
                    py_serial.write(b'2')
                current_alert_state = target_state

            # 화면 시각화 (얼굴, 눈, 입 박스)
            cv2.rectangle(frame, (face_x1, face_y1), (face_x2, face_y2), box_color, 2)
            for eye_pts in [left_eye_pts, right_eye_pts]:
                ex1, ey1 = min([p for p in eye_pts]), min([p for p in eye_pts])
                ex2, ey2 = max([p for p in eye_pts]), max([p for p in eye_pts])
                cv2.rectangle(frame, (ex1-3, ey1-3), (ex2+3, ey2+3), (255, 255, 0), 1)

            mouth_pts = get_pts(MOUTH_ALL)
            mx1, my1 = min([p for p in mouth_pts]), min([p for p in mouth_pts])
            mx2, my2 = max([p for p in mouth_pts]), max([p for p in mouth_pts])
            cv2.rectangle(frame, (mx1-5, my1-5), (mx2+5, my2+5), (0, 165, 255), 1)

            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (50, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    cv2.imshow('Drowsiness & Yawn Detection', frame)
    if cv2.waitKey(1) & 0xFF == 27: # ESC로 종료
        break

if py_serial:
    py_serial.write(b'0') # 종료 시 모든 LED 끄기
    py_serial.close()
cap.release()
cv2.destroyAllWindows()
