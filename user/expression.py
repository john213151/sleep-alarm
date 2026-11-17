import cv2
import mediapipe as mp
import serial
import time
from scipy.spatial import distance as dist

# 1. 아두이노 시리얼 통신 설정
arduino_port = '/dev/ttyACM0' 
try:
    py_serial = serial.Serial(port=arduino_port, baudrate=9600, timeout=1)
    print(f"아두이노 연결 성공 ({arduino_port})")
except Exception as e:
    print(f"아두이노 연결 실패: {e}. 포트를 확인하세요.")
    py_serial = None

# 2. MediaPipe 및 카메라 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
cap = cv2.VideoCapture(0)

# 📌 임계값(Threshold) 설정
EAR_THRESHOLD = 0.22      # 눈 감김
MAR_THRESHOLD = 0.50      # 입 벌림(하품)
CLOSED_SECONDS = 1.5      # 지속 시간(초)

# 상태 관리 변수
alert_active = False       
closed_start_time = None
yawn_start_time = None

# 랜드마크 인덱스 정의
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308
MOUTH_ALL = [78, 81, 13, 311, 308, 402, 14, 178]

# 🤨 표정 인식용 추가 인덱스
BROW_LEFT_IN = 55         # 왼쪽 눈썹 안쪽 끝
BROW_RIGHT_IN = 285       # 오른쪽 눈썹 안쪽 끝
BROW_LEFT_TOP = 70        # 왼쪽 눈썹 위 중앙
BROW_RIGHT_TOP = 300      # 오른쪽 눈썹 위 중앙

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    drowsy_detected = False
    yawn_detected = False
    emotion = "Normal"  # 기본 표정 상태
    box_color = (0, 255, 0)  

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 얼굴 전체 크기 구하기
            all_x = [int(lm.x * w) for lm in face_landmarks.landmark]
            all_y = [int(lm.y * h) for lm in face_landmarks.landmark]
            face_x1, face_y1 = min(all_x), min(all_y)
            face_x2, face_y2 = max(all_x), max(all_y)
            
            # 얼굴 너비를 기준 단위(Scale)로 삼아 거리 정규화 (카메라 거리에 영향 안 받게 함)
            face_width = dist.euclidean((face_x1, face_y1), (face_x2, face_y1))

            def get_pt(idx):
                return (int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h))
            def get_pts(indices):
                return [get_pt(i) for i in indices]

            # 1. 눈 EAR 및 입 MAR 계산
            left_eye_pts = get_pts(LEFT_EYE)
            right_eye_pts = get_pts(RIGHT_EYE)
            
            left_ear = (dist.euclidean(left_eye_pts[1], left_eye_pts[5]) + dist.euclidean(left_eye_pts[2], left_eye_pts[4])) / (2.0 * dist.euclidean(left_eye_pts[0], left_eye_pts[3]))
            right_ear = (dist.euclidean(right_eye_pts[1], right_eye_pts[5]) + dist.euclidean(right_eye_pts[2], right_eye_pts[4])) / (2.0 * dist.euclidean(right_eye_pts[0], right_eye_pts[3]))
            avg_ear = (left_ear + right_ear) / 2.0

            p_top, p_bottom = get_pt(MOUTH_TOP), get_pt(MOUTH_BOTTOM)
            p_left, p_right = get_pt(MOUTH_LEFT), get_pt(MOUTH_RIGHT)
            mar = dist.euclidean(p_top, p_bottom) / dist.euclidean(p_left, p_right)

            # 2. 표정 분석용 거리 지표 계산 (얼굴 너비로 나눠서 정규화)
            mouth_width = dist.euclidean(p_left, p_right) / face_width
            brow_dist = dist.euclidean(get_pt(BROW_LEFT_IN), get_pt(BROW_RIGHT_IN)) / face_width
            
            # 눈동자 중심 근처에서 눈썹까지의 수직 거리 (눈썹 들림 측정)
            left_brow_lift = dist.euclidean(get_pt(BROW_LEFT_TOP), left_eye_pts[4]) / face_width
            right_brow_lift = dist.euclidean(get_pt(BROW_RIGHT_TOP), right_eye_pts[4]) / face_width
            avg_brow_lift = (left_brow_lift + right_brow_lift) / 2.0

            # 🎭 규칙 기반 표정 판별 (사용자 얼굴 특성에 맞춰 수치 조절 필요)
            if mouth_width > 0.38 and mar < 0.25:  # 입이 옆으로 벌어지고 크게 열리지 않았을 때
                emotion = "Happy"
            elif brow_dist < 0.16:                 # 미간이 좁아졌을 때
                emotion = "Angry"
            elif avg_brow_lift > 0.25 and mar > 0.15: # 눈썹이 높게 들리고 입이 살짝 벌어졌을 때
                emotion = "Surprised"
            else:
                emotion = "Normal"

            # 🛑 [판단 1] 졸음 감지
            if avg_ear < EAR_THRESHOLD:
                if closed_start_time is None:
                    closed_start_time = time.time()
                elif time.time() - closed_start_time >= CLOSED_SECONDS:
                    drowsy_detected = True
            else:
                closed_start_time = None

            # 🥱 [판단 2] 하품 감지
            if mar > MAR_THRESHOLD:
                if yawn_start_time is None:
                    yawn_start_time = time.time()
                elif time.time() - yawn_start_time >= CLOSED_SECONDS:
                    yawn_detected = True
            else:
                yawn_start_time = None

            # 🚨 시각화 및 안내 문구
            if drowsy_detected or yawn_detected:
                box_color = (0, 0, 255)
                if drowsy_detected:
                    cv2.putText(frame, "DROWSINESS ALERT!", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                if yawn_detected:
                    cv2.putText(frame, "YAWNING ALERT!", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)

            # 화면 텍스트 출력
            cv2.rectangle(frame, (face_x1, face_y1), (face_x2, face_y2), box_color, 2)
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (50, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            # 현재 표정 출력 (초록색 텍스트)
            cv2.putText(frame, f"Emotion: {emotion}", (50, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 특징점 시각화 (눈, 입 박스)
            for eye_pts in [left_eye_pts, right_eye_pts]:
                ex1, ey1 = min([p[0] for p in eye_pts]), min([p[1] for p in eye_pts])
                ex2, ey2 = max([p[0] for p in eye_pts]), max([p[1] for p in eye_pts])
                cv2.rectangle(frame, (ex1-3, ey1-3), (ex2+3, ey2+3), (255, 255, 0), 1)

            mouth_pts = get_pts(MOUTH_ALL)
            mx1, my1 = min([p[0] for p in mouth_pts]), min([p[1] for p in mouth_pts])
            mx2, my2 = max([p[0] for p in mouth_pts]), max([p[1] for p in mouth_pts])
            cv2.rectangle(frame, (mx1-5, my1-5), (mx2+5, my2+5), (0, 165, 255), 1)

    else:
        closed_start_time = None
        yawn_start_time = None

    # 🔌 아두이노 시리얼 제어
    if py_serial:
        if (drowsy_detected or yawn_detected):
            if not alert_active:
                py_serial.write(b'1')
                alert_active = True
        else:
            if alert_active:
                py_serial.write(b'0')
                alert_active = False

    cv2.imshow('Drowsiness & Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

if py_serial:
    py_serial.write(b'0')
    py_serial.close()
cap.release()
cv2.destroyAllWindows()
