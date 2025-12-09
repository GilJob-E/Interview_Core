import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math

# ============================================================
# 1. 모델 파일 경로
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "face_landmarker.task")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"'{model_path}' 파일을 찾을 수 없습니다. 이 파일과 같은 폴더에 두세요.")


# ============================================================
# 2. 공통 유틸 (블렌드셰이프, 행렬 → 오일러각)
# ============================================================

def get_blendshape_map(blendshapes):
    """MediaPipe blendshapes 리스트를 {name: score} 딕셔너리로 변환."""
    return {b.category_name: b.score for b in blendshapes}


def matrix_to_euler(matrix4: np.ndarray):
    """
    4x4 변환 행렬에서 pitch, yaw, roll(rad) 추출.
    MediaPipe FaceLandmarker 의 facial_transformation_matrixes 사용.
    """
    R = matrix4[:3, :3]
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(R[2, 1], R[2, 2])
        yaw   = np.arctan2(-R[2, 0], sy)
        roll  = np.arctan2(R[1, 0], R[0, 0])
    else:
        pitch = np.arctan2(-R[1, 2], R[1, 1])
        yaw   = np.arctan2(-R[2, 0], sy)
        roll  = 0.0

    return pitch, yaw, roll


def head_pose_matrix(result):
    """
    FaceLandmarkerResult → (pitch, yaw, roll) in rad
    facial_transformation_matrixes 가 없으면 (0,0,0) 리턴.
    """
    matrices = getattr(result, "facial_transformation_matrixes", None)
    if not matrices or len(matrices) == 0:
        return 0.0, 0.0, 0.0

    m = np.array(matrices[0], dtype=np.float32).reshape(4, 4).T  # column-major 보정
    pitch, yaw, roll = matrix_to_euler(m)
    return pitch, yaw, roll


# ============================================================
# 3. 표정 / 움직임 관련 (미소, 끄덕임, 앞으로 숙이기)
# ============================================================

def smile(blendshapes):
    """Duchenne smile 정도를 0~1 스케일로 반환."""
    m = get_blendshape_map(blendshapes)
    left  = m.get("mouthSmileLeft", 0.0)
    right = m.get("mouthSmileRight", 0.0)

    squint_l = m.get("eyeSquintLeft", 0.0)
    squint_r = m.get("eyeSquintRight", 0.0)

    base_smile = (left + right)
    duchenne_bonus = (squint_l + squint_r) / 2.0 * 0.5
    score = np.clip(base_smile + duchenne_bonus, 0.0, 1.0)

    return float(score)


# nod 감지에 사용할 pitch 임계값들
# DOWN_TH = -0.001
# UP_TH   = -0.0005
# MAX_NOD_FRAMES = 240  # 대략 1초 정도 (30fps 기준)

# def detect_nod(pitch_list):
#     """
#     pitch_list: 최근 pitch 값들.
#     '아래로 숙였다가 다시 올린' 패턴이 있으면 True.
#     """
#     if len(pitch_list) < 2:
#         return False

#     # 마지막 값이 충분히 아래로 숙인 상태인지
#     if pitch_list[-1] > DOWN_TH:
#         return False

#     # 최근 MAX_NOD_FRAMES 안에서 UP_TH 이상으로 올라간 적이 있었는지
#     recent = pitch_list[-MAX_NOD_FRAMES:]
#     went_up = any(p > UP_TH for p in recent)

#     return went_up

def detect_nod(pitch_history):
    """
    절대값이 아닌 '변화량(Amplitude)'으로 끄덕임 감지
    """
    if len(pitch_history) < 5: return False

    # 최근 N 프레임(약 0.5초~1초) 동안의 고개 각도
    recent = pitch_history[-10:] # 최근 10개 프레임만 봄 (스킵 고려)

    max_pitch = max(recent) # 고개를 가장 들었을 때
    min_pitch = min(recent) # 고개를 가장 숙였을 때
    
    amplitude = max_pitch - min_pitch # 움직임의 폭
    
    # [핵심] 고개 위치가 어디든, 위아래로 0.07 라디안(약 4도) 이상 움직였으면 끄덕임!
    # 이 값(0.07)만 조절하면 민감도가 바뀝니다.
    NOD_AMPLITUDE_TH = 0.05
    
    if amplitude > NOD_AMPLITUDE_TH:
        # 추가 검증: 현재 고개가 다시 올라오고 있는 중인가? (선택사항)
        # 숙였다가(min) 다시 올라오는 패턴인지 확인하면 더 정확함
        current_pitch = pitch_history[-1]
        if current_pitch > min_pitch + (NOD_AMPLITUDE_TH * 0.5):
            return True
            
    return False


def detect_lean_forward(prev_nose_z, cur_nose_z, move_th=0.02):
    """
    코 z 좌표가 카메라 쪽으로 가까워지는지(몸을 앞으로 숙이는지) 감지.
    """
    if prev_nose_z is None:
        return False

    dz = prev_nose_z - cur_nose_z  # z 감소 → 카메라와 거리 감소
    return dz > move_th


EYE_CONTACT_MEAN_RATIO = 0.70  # 외부 연구 기반 평균 eye-contact 비율
EYE_CONTACT_STD_RATIO  = 0.15  # 외부 연구 기반 표준편차

# ============================================================
# 2. Ye 논문 스타일 8D Geometry Gaze Feature
#    feats = [PPx, PPy, RX, RY, Hf, Wf, PME_x, PME_y]
# ============================================================

def compute_gaze_features_ye(landmarks):
    """
    Ye et al. 논문의 8D geometry feature 구조를 따른다.

    - PPx, PPy : 양 눈 pupil 위치 (MCA - iris) 를 얼굴 크기로 정규화 후 좌/우 합
    - RX, RY   : head pitch / yaw (랜드마크 z, y/x 비율로 계산)
    - Hf, Wf   : 얼굴 높이/너비 (3D 거리)
    - PME_x,y  : 얼굴 중앙(눈 사이)의 이미지 내 좌표
    """
    def p3(idx):
        lm = landmarks[idx]
        return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

    # Medial canthus / mid-eye / bottom of nose
    P_rMCA = p3(133)   # right medial canthus
    P_lMCA = p3(362)   # left medial canthus
    P_ME   = p3(168)   # mid between eyes
    P_BN   = p3(2)     # bottom of nose

    eps = 1e-6

    # --- head rotations (RY, RX) ---
    RY = (P_lMCA[2] - P_rMCA[2]) / (P_lMCA[0] - P_rMCA[0] + eps)
    RX = (P_ME[2]   - P_BN[2])   / (P_ME[1]   - P_BN[1]   + eps)

    # --- face size (3D) ---
    Hf = float(np.linalg.norm(P_ME - P_BN))          # 얼굴 높이
    Wf = float(np.linalg.norm(P_lMCA - P_rMCA))      # 얼굴 너비

    # --- head position (이미지 내 얼굴 위치) ---
    PME_x, PME_y = float(P_ME[0]), float(P_ME[1])

    # --- iris centers (3D) ---
    irisL = np.mean([p3(i) for i in [469, 470, 471, 472]], axis=0)
    irisR = np.mean([p3(i) for i in [474, 475, 476, 477]], axis=0)

    # 왼/오 pupil 위치 (MCA - iris), 얼굴 크기로 정규화
    PlP_x = (P_lMCA[0] - irisL[0]) / (Wf + eps)
    PlP_y = (P_lMCA[1] - irisL[1]) / (Hf + eps)
    PrP_x = (P_rMCA[0] - irisR[0]) / (Wf + eps)
    PrP_y = (P_rMCA[1] - irisR[1]) / (Hf + eps)

    # 두 눈 합 → PPx, PPy
    PPx = PlP_x + PrP_x
    PPy = PlP_y + PrP_y

    feat = np.array([PPx, PPy, RX, RY, Hf, Wf, PME_x, PME_y],
                    dtype=np.float32)
    return feat

# ============================================================
# 1. FaceLandmarker 빌드 (VIDEO 모드)
# ============================================================

def build_face_landmarker():
    BaseOptions = python.BaseOptions
    FaceLandmarker = vision.FaceLandmarker
    FaceLandmarkerOptions = vision.FaceLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    detector = FaceLandmarker.create_from_options(options)
    return detector


_frame_id = 0  # VIDEO 모드용 timestamp 대용 (frame index)

def detect_landmarks(detector, frame_bgr):
    """
    BGR 프레임 → mediapipe detect → (landmarks, blend, pitch,yaw, features)
    얼굴이 없으면 (None, None, None, None, None)
    """
    global _frame_id
    _frame_id += 1
    
    timestamp_ms = int(_frame_id * (1000 / 30))  # 대충 30fps 가정

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    result = detector.detect_for_video(mp_image, timestamp_ms)

    if not result.face_landmarks:
        return None, None, None, None

    face_landmarks = result.face_landmarks[0]
    blendshapes = result.face_blendshapes[0] if result.face_blendshapes else None
    pitch, yaw, roll = head_pose_matrix(result)

    return face_landmarks, blendshapes, pitch, yaw

# ============================================================
# 3. 중앙 한 점 캘리브레이션 (중앙만 보고 N 프레임 수집)
# ============================================================

def run_center_calibration(detector, num_samples=120):
    """
    화면 중앙에 점 하나를 띄우고, 그 점을 바라보는 동안
    Ye 스타일 8D gaze feature를 num_samples 개수만큼 수집.
    → mean_feat, std_feat 리턴 (나중에 z-score / eye-contact 판정에 사용)
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다 (VideoCapture(0) 실패).")

    print("=== Gaze Calibration (Center Point) ===")
    print("1) 카메라 창 중앙의 점을 응시하고 고개를 두바퀴 돌리세요.")
    print("2) 준비되면 'c' 를 누르면 수집이 시작됩니다.")
    print("3) 약 몇 초 동안 자동으로 수집합니다. (종료: q)")

    collecting = False
    collected = 0
    feats = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("웹캠 프레임을 읽지 못했습니다. 종료합니다.")
            break

        h, w, _ = frame.shape
        center = (w // 2, h // 2)
        cv2.circle(frame, center, 6, (0, 255, 0), -1)

        msg = "Press 'c' to start CENTER calibration"
        if collecting:
            msg = f"Collecting center gaze... {collected}/{num_samples}"
        cv2.putText(frame, msg, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Center Calibration", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("사용자에 의해 종료(q).")
            break

        if (not collecting) and key == ord('c'):
            print("캘리브레이션 시작! 중앙 점을 계속 바라보세요.")
            collecting = True
            collected = 0
            feats = []
            continue

        if collecting:
            lm, blend, pitch, yaw = detect_landmarks(detector, frame)
            if lm is not None:
                f = compute_gaze_features_ye(lm)
                feats.append(f)
                collected += 1

            if collected >= num_samples:
                print("필요한 샘플 수집 완료.")
                break

    cap.release()
    cv2.destroyWindow("Center Calibration")

    if len(feats) == 0:
        raise RuntimeError("캘리브레이션 동안 얼굴을 감지하지 못했습니다.")

    feats = np.stack(feats, axis=0)  # (N, 8)

    mean_feat = feats.mean(axis=0)
    std_feat = feats.std(axis=0, ddof=0)
    std_feat = np.where(std_feat < 1e-6, 1e-6, std_feat)  # 0 분산 방지

    print("\n=== CENTER CALIBRATION SUMMARY ===")
    print("num_samples :", feats.shape[0])
    print("mean_feat   :", mean_feat)
    print("std_feat    :", std_feat)

    return mean_feat, std_feat


# ============================================================
# 4. 프레임당 아이컨택 판정 (Ye geometry 기반 단일점 버전)
# ============================================================

def eye_contact_from_features_ye(
    feat,
    mean_feat,
    std_feat,
    thr_eye=2.0,   # 눈
    thr_head=4.0    # 머리
):
    """
    - feat, mean_feat, std_feat : 8D gaze feature (PPx,PPy,RX,RY,Hf,Wf,PME_x,y)
    - 논문에서 Gx ~ [RY, PPx, Wf, PME_x], Gy ~ [RX, PPy, Hf, PME_y] 를 쓰는 걸 따라
      여기서는 중앙 캘리브레이션 기준으로
        수평 dev_h = sqrt( z_RY^2 + z_PPx^2 )
        수직 dev_v = sqrt( z_RX^2 + z_PPy^2 )
      로 “중앙에서 얼마나 벗어났는지”를 본다.

    - dev_h, dev_v 가 각각 thr_h, thr_v 이하이면 아이컨택으로 판정.
    """
    z = (feat - mean_feat) / std_feat

    # index: [PPx, PPy, RX, RY, Hf, Wf, PME_x, PME_y]
    z_PPx = float(z[0])
    z_PPy = float(z[1])
    z_RX  = float(z[2])
    z_RY  = float(z[3])

    dev_eye  = np.sqrt(z_PPx**2 + z_PPy**2)
    dev_head = np.sqrt(z_RX**2  + z_RY**2)

    is_contact = (dev_eye <= thr_eye) and (dev_head <= thr_head)

    score_eye  = max(0.0, 1.0 - dev_eye / thr_eye)
    score_head = max(0.0, 1.0 - dev_head / thr_head)
    score = 0.7 * score_eye + 0.3 * score_head

    return is_contact, score, dev_eye, dev_head


def run_interview_session(detector, mean_feat, std_feat,
                          thr_h=2.0, thr_v=4.0):
    """
    - 각 프레임마다 Ye 8D feature 계산
    - 중앙 캘리브레이션(mean/std) 기준 dev_h, dev_v 계산
    - threshold 안이면 eye-contact=True
    - 전체 세션 동안 eye-contact ratio 계산
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다 (VideoCapture(0) 실패).")

    print("=== Interview Session (Eye Contact Tracking) ===")
    print("q 를 누르면 종료합니다.")

    total_face_frames = 0
    eye_contact_frames = 0

    # 미소 통계
    smile_sum = 0.0

    # nod / lean 통계
    pitch_history = []
    nod_count = 0
    prev_nose_z = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("웹캠 프레임을 읽지 못했습니다. 종료합니다.")
            break

        lm, blend, pitch, yaw = detect_landmarks(detector, frame)

        if lm is not None:
            feat = compute_gaze_features_ye(lm)

            total_face_frames += 1
            is_contact, score, dev_h, dev_v = eye_contact_from_features_ye(
                feat, mean_feat, std_feat, thr_eye=thr_h, thr_head=thr_v
            )
            if is_contact:
                eye_contact_frames += 1

            ratio = eye_contact_frames / max(total_face_frames, 1)
            z_eye = (ratio - EYE_CONTACT_MEAN_RATIO) / EYE_CONTACT_STD_RATIO

            # --- 미소 점수 ---
            smile_score = smile(blend) if blend is not None else 0.0
            smile_sum += smile_score

            # --- nod 감지 (pitch 이력 기반) ---
            pitch_history.append(pitch)
            if detect_nod(pitch_history):
                nod_count += 1
                pitch_history = []  # 한 번 카운트 후 이력 초기화

            h, w, _ = frame.shape
            cx = int(lm[168].x * w)
            cy = int(lm[168].y * h)
            color = (0, 255, 0) if is_contact else (0, 0, 255)
            cv2.circle(frame, (cx, cy), 4, color, -1)

            # 오버레이 정보
            cv2.putText(frame, f"Eye-contact ratio: {ratio*100:.1f}%",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)
            cv2.putText(frame, f"score: {score:.2f}",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)
            cv2.putText(frame, f"z-eye: {z_eye:.2f}", 
                        (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                        (0, 200, 255), 2)
            cv2.putText(frame, f"Smile(avg): {(smile_sum/max(total_face_frames,1))*100:.1f}", 
                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                        (255, 255, 0), 2)
            cv2.putText(frame, f"Nods: {nod_count}", 
                        (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                        (0, 180, 255), 2)
            cv2.putText(frame, f"Pitch: {pitch:.2f} deg",
                        (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face detected",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)

        cv2.imshow("Interview (Eye Contact)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyWindow("Interview (Eye Contact)")

    SMILE_MEAN = 32.93 # 직접 측정한 값
    SMILE_STD  = 17.72 
    
    if total_face_frames > 0:
        eye_ratio = eye_contact_frames / total_face_frames
        z_eye = (eye_ratio - EYE_CONTACT_MEAN_RATIO) / EYE_CONTACT_STD_RATIO
        percent_eye = 0.5 * (1.0 + math.erf(z_eye / math.sqrt(2))) * 100.0

        avg_smile_0_1 = smile_sum / total_face_frames
        avg_smile_0_100 = avg_smile_0_1 * 100.0
        z_smile = (avg_smile_0_1 - SMILE_MEAN) / SMILE_STD
        percent_smile = 0.5 * (1.0 + math.erf(z_smile / math.sqrt(2))) * 100.0

        print("\n=== INTERVIEW SUMMARY ===")
        print(f"Total face frames: {total_face_frames}")
        print(f"Eye-contact frames: {eye_contact_frames}")
        print(f"Eye-contact ratio: {eye_ratio*100:.1f}%")
        print(f"z-eye (mu=0.70, sigma=0.15): {z_eye:.2f} "
            f"(상위 {percent_eye:.1f}%)")
        print(f"Mean SmileIntensity: {avg_smile_0_100:.2f} / 100 "
            f"(z-smile={z_smile:.2f}, 상위 {percent_smile:.1f}%)")
        print(f"Nod Count: {nod_count}")
        
    else:
        print("세션 동안 얼굴이 한 번도 검출되지 않았습니다.")

# ============================================================
# 7. 메인 실행
# ============================================================

def main():
    detector = build_face_landmarker()
    
    global _frame_id
    _frame_id = 0

     # 1) 중앙 한 점 캘리브레이션
    mean_feat, std_feat = run_center_calibration(detector, num_samples=120)

    # 2) 인터뷰 세션에서 프레임당 아이컨택 + 비율 계산
    run_interview_session(detector, mean_feat, std_feat,
                          thr_h=2.0, thr_v=1.5)


if __name__ == "__main__":
    main()