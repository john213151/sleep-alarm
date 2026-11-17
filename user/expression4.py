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

# 상태 관리를 문자열 타입으로 통합 변경
current_alert = '0' 
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

# 📌 표정 분석을 위한 랜드마크 인덱스
BROW_INNER_LEFT = 55
BROW_INNER_RIGHT = 285
LIP_CORNER_LEFT = 61
LIP_CORNER_RIGHT = 291

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
            box_color = (0, 255, 0)
            
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
            
            # 📌 3. 실시간 표정 분석 로직
            c_left, c_right = get_pt(LIP_CORNER_LEFT), get_pt(LIP_CORNER_RIGHT)
            smile_width = dist.euclidean(c_left, c_right)
            smile_ratio = smile_width / dist.euclidean(p_left, p_right)
            
            b_left, b_right = get_pt(BROW_INNER_LEFT), get_pt(BROW_INNER_RIGHT)
            brow_dist = dist.euclidean(b_left, b_right)
            normalized_brow_dist = brow_dist / smile_width
            
            if smile_ratio > 1.05:
                expression = "Happy"
                text_color = (0, 255, 255)
            elif normalized_brow_dist < 0.22:
                expression = "Angry"
                text_color = (0, 0, 255)
            else:
                expression = "Neutral"
                text_color = (0, 255, 0)
                
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
                
            # 🚨 경고 알림 및 시리얼 신호 개별 처리
            if drowsy_detected or yawn_detected:
                box_color = (0, 0, 255)
                
                if drowsy_detected:
                    cv2.putText(frame, "DROWSINESS ALERT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    # 졸음 우선순위 분기 (동시 발생 시 졸음 LED를 우선 작동)
                    if py_serial and current_alert != 'D':
                        py_serial.write(b'D')
                        current_alert = 'D'
                        
                elif yawn_detected: # 하품만 감지된 경우
                    cv2.putText(frame, "YAWNING ALERT!", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
                    if py_serial and current_alert != 'Y':
                        py_serial.write(b'Y')
                        current_alert = 'Y'
            else:
                # 정상 상태일 때 신호 한 번만 전송
                if py_serial and current_alert != '0':
                    py_serial.write(b'0')
                    current_alert = '0'
                    
            # [시각화] 1. 얼굴 사각형 및 표정
            cv2.rectangle(frame, (face_x1, face_y1), (face_x2, face_y2), box_color, 2)
            text_y = face_y1 - 10 if face_y1 - 10 > 20 else face_y1 + 30
            cv2.putText(frame, f"Emotion: {expression}", (face_x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            
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
