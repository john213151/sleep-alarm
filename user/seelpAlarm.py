import cv2
import mediapipe as mp
import serial
import time
from scipy.spatial import distance as dist

# 1. 아두이노 시리얼 통신 설정 (포트 번호는 본인 설정에 맞게 변경)
arduino_port = '/dev/ttyACM0'  # 리눅스/우분투의 경우 '/dev/ttyACM0' 또는 '/dev/ttyUSB0'
try:
    py_serial = serial.Serial(port=arduino_port, baudrate=9600, timeout=1)
    print(f"아두이노 연결 성공 ({arduino_port})")
except:
    print(f"아두이노 연결 실패. 포트({arduino_port})를 확인하세요.")
    py_serial = None

# 2. 비율 계산 함수 (눈 EAR 및 입 MAR용)
def calculate_ratio(pts_vertical1, pts_vertical2, pts_horizontal1, pts_horizontal2):
    # 세로 거리 계산
    A = dist.euclidean(pts_vertical1, pts_vertical2)
    # 가로 거리 계산
    B = dist.euclidean(pts_horizontal1, pts_horizontal2)
    # 비율 계산
    return A / B

# 3. MediaPipe 및 카메라 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
cap = cv2.VideoCapture(0) # 0번이 PC 내장 카메라

# 📌 졸음 및 하품 매개변수 설정
EAR_THRESHOLD = 0.22     # 눈 감김 임계값
MAR_THRESHOLD = 0.50     # 입 벌림(하품) 임계값 (입이 가로 대비 세로로 크게 벌어질 때)
CLOSED_SECONDS = 1.5     # 지속 시간 기준 (초)

alert_active = False     # 아두이노 경고 상태 변수
closed_start_time = None
yawn_start_time = None

# MediaPipe Face Mesh 기준 주요 랜드마크 인덱스
# 눈 EAR 계산용 (각 6개 포인트 고정)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# 입 내부 윤곽선 인덱스 (하품 MAR 계산용)
MOUTH_TOP = 13          # 윗입술 아래 중앙
MOUTH_BOTTOM = 14       # 아랫입술 위 중앙
MOUTH_LEFT = 78         # 입술 왼쪽 끝
MOUTH_RIGHT = 308       # 입술 오른쪽 끝
MOUTH_ALL = [78, 81, 13, 311, 308, 402, 14, 178] # 입 주변 박스용 인덱스

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
            
            # 얼굴 사각형 좌표 계산
            all_x = [int(lm.x * w) for lm in face_landmarks.landmark]
            all_y = [int(lm.y * h) for lm in face_landmarks.landmark]
            face_x1, face_y1 = min(all_x), min(all_y)
            face_x2, face_y2 = max(all_x), max(all_y)
            box_color = (0, 255, 0) # 기본 초록색
            
            # 좌표 추출 헬퍼 함수
            def get_pt(idx):
                return (int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h))
            
            def get_pts(indices):
                return [get_pt(i) for i in indices]

            # 1. 눈 EAR 계산
            left_eye_pts = get_pts(LEFT_EYE)
            right_eye_pts = get_pts(RIGHT_EYE)
            
            # 눈 세로/가로 거리 구조에 맞게 EAR 계산
            left_ear = (dist.euclidean(left_eye_pts[1], left_eye_pts[5]) + dist.euclidean(left_eye_pts[2], left_eye_pts[4])) / (2.0 * dist.euclidean(left_eye_pts[0], left_eye_pts[3]))
            right_ear = (dist.euclidean(right_eye_pts[1], right_eye_pts[5]) + dist.euclidean(right_eye_pts[2], right_eye_pts[4])) / (2.0 * dist.euclidean(right_eye_pts[0], right_eye_pts[3]))
            avg_ear = (left_ear + right_ear) / 2.0

            # 2. 입 MAR 계산 (세로 중앙 거리 / 가로 양끝 거리)
            p_top, p_bottom = get_pt(MOUTH_TOP), get_pt(MOUTH_BOTTOM)
            p_left, p_right = get_pt(MOUTH_LEFT), get_pt(MOUTH_RIGHT)
            mar = dist.euclidean(p_top, p_bottom) / dist.euclidean(p_left, p_right)

            # 🛑 [판단 1] 졸음 감지 (눈)
            drowsy_detected = False
            if avg_ear < EAR_THRESHOLD:
                if closed_start_time is None:
                    closed_start_time = time.time()
                elif time.time() - closed_start_time >= CLOSED_SECONDS:
                    drowsy_detected = True
            else:
                closed_start_time = None

            # 🥱 [판단 2] 하품 감지 (입)
            yawn_detected = False
            if mar > MAR_THRESHOLD:
                if yawn_start_time is None:
                    yawn_start_time = time.time()
                elif time.time() - yawn_start_time >= CLOSED_SECONDS:
                    yawn_detected = True
            else:
                yawn_start_time = None

            # 🚨 경고 알림 통합 처리 (졸음 또는 하품 발생 시 아두이노 작동)
            if drowsy_detected or yawn_detected:
                box_color = (0, 0, 255) # 얼굴 사각형 빨간색으로 변경
                
                if drowsy_detected:
                    cv2.putText(frame, "DROWSINESS ALERT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                if yawn_detected:
                    cv2.putText(frame, "YAWNING ALERT!", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3) # 주황색 문자
                
                if py_serial and not alert_active:
                    py_serial.write(b'1') # 아두이노 경고 ON
                    alert_active = True
            else:
                if py_serial and alert_active:
                    py_serial.write(b'0') # 아두이노 경고 OFF
                    alert_active = False

            # [시각화] 1. 얼굴 사각형
            cv2.rectangle(frame, (face_x1, face_y1), (face_x2, face_y2), box_color, 2)

            # [시각화] 2. 양쪽 눈 사각형 (하늘색)
            for eye_pts in [left_eye_pts, right_eye_pts]:
                ex1, ey1 = min([p[0] for p in eye_pts]), min([p[1] for p in eye_pts])
                ex2, ey2 = max([p[0] for p in eye_pts]), max([p[1] for p in eye_pts])
                cv2.rectangle(frame, (ex1-3, ey1-3), (ex2+3, ey2+3), (255, 255, 0), 1)

            # [시각화] 3. 입 주변 사각형 (주황색)
            mouth_pts = get_pts(MOUTH_ALL)
            mx1, my1 = min([p[0] for p in mouth_pts]), min([p[1] for p in mouth_pts])
            mx2, my2 = max([p[0] for p in mouth_pts]), max([p[1] for p in mouth_pts])
            cv2.rectangle(frame, (mx1-5, my1-5), (mx2+5, my2+5), (0, 165, 255), 1)

            # 화면에 실시간 수치 표시
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (50, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    cv2.imshow('Drowsiness & Yawn Detection', frame)
    
    # ESC 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

if py_serial:
    py_serial.write(b'0')
    py_serial.close()
cap.release()
cv2.destroyAllWindows()
