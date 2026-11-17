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
except:
    print(f"아두이노 연결 실패. 포트({arduino_port})를 확인하세요.")
    py_serial = None

# 3. MediaPipe 및 카메라 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
cap = cv2.VideoCapture(0)

# 📌 졸음 및 하품 매개변수 설정
EAR_THRESHOLD = 0.22 
MAR_THRESHOLD = 0.50 
CLOSED_SECONDS = 1.5 
alert_active = False 
closed_start_time = None
yawn_start_time = None

# MediaPipe Face Mesh 주요 인덱스
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_TOP = 13 
MOUTH_BOTTOM = 14 
MOUTH_LEFT = 78 
MOUTH_RIGHT = 308 
MOUTH_ALL = [78, 81, 13, 311, 308, 402, 14, 178]

# 📌 [추가] 표정 분석을 위한 랜드마크 인덱스
BROW_INNER_LEFT = 55     # 왼쪽 눈썹 안쪽 끝
BROW_INNER_RIGHT = 285   # 오른쪽 눈썹 안쪽 끝
NOSE_BRIDGE = 6          # 미간 중심 부근
LIP_CORNER_LEFT = 61     # 윗입술 왼쪽 끝 구석
LIP_CORNER_RIGHT = 291    # 윗입술 오른쪽 끝 구석

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
            left_ear = (dist.euclidean(left_eye_pts[1], left_eye_pts[5]) + dist.euclidean(left_eye_pts[2], left_eye_pts[4])) / (2.0 * dist.euclidean(left_eye_pts[0], left_eye_pts[3]))
            right_ear = (dist.euclidean(right_eye_pts[1], right_eye_pts[5]) + dist.euclidean(right_eye_pts[2], right_eye_pts[4])) / (2.0 * dist.euclidean(right_eye_pts[0], right_eye_pts[3]))
            avg_ear = (left_ear + right_ear) / 2.0
            
            # 2. 입 MAR 계산
            p_top, p_bottom = get_pt(MOUTH_TOP), get_pt(MOUTH_BOTTOM)
            p_left, p_right = get_pt(MOUTH_LEFT), get_pt(MOUTH_RIGHT)
            mar = dist.euclidean(p_top, p_bottom) / dist.euclidean(p_left, p_right)
            
            # 📌 [추가] 3. 실시간 표정 분석 로직
            # 입꼬리와 입 중앙 간의 가로/세로 비율을 이용한 미소(Happy) 판단
            c_left, c_right = get_pt(LIP_CORNER_LEFT), get_pt(LIP_CORNER_RIGHT)
            smile_width = dist.euclidean(c_left, c_right)
            smile_ratio = smile_width / dist.euclidean(p_left, p_right)
            
            # 눈썹 사이 거리를 이용한 화남(Angry) 판단 (눈썹을 찌푸리면 거리가 좁아짐)
            b_left, b_right = get_pt(BROW_INNER_LEFT), get_pt(BROW_INNER_RIGHT)
            brow_dist = dist.euclidean(b_left, b_right)
            # 얼굴 크기에 대비하여 눈썹 거리를 정규화
            normalized_brow_dist = brow_dist / smile_width 

            # 임계값 기준으로 표정 결정
            if smile_ratio > 1.05:
                expression = "Happy"
                text_color = (0, 255, 255) # 노란색
            elif normalized_brow_dist < 0.22:
                expression = "Angry"
                text_color = (0, 0, 255) # 빨간색
            else:
                expression = "Neutral"
                text_color = (0, 255, 0) # 초록색

            # 🛑 [판단 1] 졸음 감지
            drowsy_detected = False
            if avg_ear < EAR_THRESHOLD:
                if closed_start_time is None:
                    closed_start_time = time.time()
                elif time.time() - closed_start_time >= CLOSED_SECONDS:
                    drowsy_detected = True
            else:
                closed_start_time = None
                
            # 🥱 [판단 2] 하품 감지
            yawn_detected = False
            if mar > MAR_THRESHOLD:
                if yawn_start_time is None:
                    yawn_start_time = time.time()
                elif time.time() - yawn_start_time >= CLOSED_SECONDS:
                    yawn_detected = True
            else:
                yawn_start_time = None
                
            # 🚨 경고 알림 통합 처리
            if drowsy_detected or yawn_detected:
                box_color = (0, 0, 255) # 얼굴 사각형 빨간색으로 변경
                if drowsy_detected:
                    cv2.putText(frame, "DROWSINESS ALERT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                if yawn_detected:
                    cv2.putText(frame, "YAWNING ALERT!", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
                
                if py_serial and not alert_active:
                    py_serial.write(b'1')
                    alert_active = True
            else:
                if py_serial and alert_active:
                    py_serial.write(b'0')
                    alert_active = False

            # [시각화] 1. 얼굴 사각형
            cv2.rectangle(frame, (face_x1, face_y1), (face_x2, face_y2), box_color, 2)
            
            # 📌 [추가] 얼굴 박스를 따라다니는 표정 텍스트 표시 (박스 상단에서 10픽셀 위)
            text_y = face_y1 - 10 if face_y1 - 10 > 20 else face_y1 + 30
            cv2.putText(frame, f"Emotion: {expression}", (face_x1, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

            # [시각화] 2. 양쪽 눈 사각형
            for eye_pts in [left_eye_pts, right_eye_pts]:
                ex1, ey1 = min([p[0] for p in eye_pts]), min([p[1] for p in eye_pts])
                ex2, ey2 = max([p[0] for p in eye_pts]), max([p[1] for p in eye_pts])
                cv2.rectangle(frame, (ex1-3, ey1-3), (ex2+3, ey2+3), (255, 255, 0), 1)
                
            # [시각화] 3. 입 주변 사각형
            mouth_pts = get_pts(MOUTH_ALL)
            mx1, my1 = min([p[0] for p in mouth_pts]), min([p[1] for p in mouth_pts])
            mx2, my2 = max([p[0] for p in mouth_pts]), max([p[1] for p in mouth_pts])
            cv2.rectangle(frame, (mx1-5, my1-5), (mx2+5, my2+5), (0, 165, 255), 1)
            
        # 기존 수치 화면 고정 표시 위치 유지
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (50, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
    cv2.imshow('Drowsiness & Yawn Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

if py_serial:
    py_serial.write(b'0')
    py_serial.close()
cap.release()
cv2.destroyAllWindows()
