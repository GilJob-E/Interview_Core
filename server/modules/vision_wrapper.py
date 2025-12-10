import numpy as np
import cv2
import os
import traceback
from collections import deque
from .vision.face import (
    build_face_landmarker, 
    detect_landmarks, 
    compute_gaze_features_ye,
    eye_contact_from_features_ye, 
    smile,
    detect_nod
)

class VisionWrapper: # 클래스 이름 유지 (혹은 RealTimeVisionWrapper로 변경 가능)
    def __init__(self):
        print("[Vision] Initializing Real-time Analyzer...")
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "vision", "face_landmarker.task")
        
        if not os.path.exists(model_path):
            print(f"[Vision Error] Model file missing at: {model_path}")
        
        self.detector = build_face_landmarker()
        
        # 1. 캘리브레이션 & 임계값 (제공해주신 파일 기준)
        self.mean_feat = np.zeros(8, dtype=np.float32)
        self.std_feat = np.full(8, 0.1, dtype=np.float32) 

        self.THR_EYE = 1.5
        self.THR_HEAD = 2.5
        
        self.EYE_CONTACT_MEAN = 0.70
        self.EYE_CONTACT_STD = 0.15
        self.SMILE_MEAN = 32.93
        self.SMILE_STD = 17.72

        # 2. 최적화: 프레임 스킵 설정
        self.frame_counter = 0
        self.SKIP_STEP = 3 

        # 3. 통계 변수 초기화
        self.reset_stats()

    def reset_stats(self):
        """누적된 통계 데이터를 초기화합니다."""
        self.total_processed_frames = 0
        self.eye_contact_frames = 0
        self.smile_scores = []
        self.nod_count = 0
        self.pitch_history = deque(maxlen=30) 

    def process_frame(self, frame):
        """
        [수정됨] 리스트가 아닌 프레임 1장을 받아 즉시 분석하고 누적
        """
        self.frame_counter += 1
        
        # 스킵 로직
        if self.frame_counter % self.SKIP_STEP != 0:
            return

        try:
            # 1. 랜드마크 추출
            landmarks, blendshapes, pitch, yaw = detect_landmarks(self.detector, frame)
            
            if landmarks:
                self.total_processed_frames += 1
                
                # 2. Eye Contact
                feat = compute_gaze_features_ye(landmarks)
                is_contact, _, _, _ = eye_contact_from_features_ye(
                    feat, self.mean_feat, self.std_feat, 
                    thr_eye=self.THR_EYE, thr_head=self.THR_HEAD
                )
                if is_contact:
                    self.eye_contact_frames += 1
                
                # 3. Smile
                if blendshapes:
                    score = smile(blendshapes) * 100
                    self.smile_scores.append(score)
                
                # 4. Nod
                if pitch is not None:
                    self.pitch_history.append(pitch)
                    if detect_nod(list(self.pitch_history)):
                        self.nod_count += 1
                        self.pitch_history.clear()

        except Exception as e:
            # 실시간 처리 중 에러는 출력만 하고 넘어감
            print(f"[Vision Process Error] {e}")

    def flush_stats(self):
        """
        [신규] 지금까지 누적된 통계를 계산하여 반환하고 리셋 (analyze 대체)
        """
        valid_frames = max(1, self.total_processed_frames)
        
        # 통계 계산
        eye_ratio = self.eye_contact_frames / valid_frames
        avg_smile = sum(self.smile_scores) / len(self.smile_scores) if self.smile_scores else 0
        
        # Z-Score
        z_eye = (eye_ratio - self.EYE_CONTACT_MEAN) / self.EYE_CONTACT_STD
        z_smile = (avg_smile - self.SMILE_MEAN) / self.SMILE_STD
        
        result = {
            "valid_frames": valid_frames,
            "features": {
                "eye_contact": {
                    "value": round(eye_ratio, 2),
                    "unit": "ratio",
                    "z_score": round(z_eye, 2)
                },
                "smile": {
                    "value": round(avg_smile, 1),
                    "unit": "intensity(0-100)",
                    "z_score": round(z_smile, 2)
                },
                "head_nod": {
                    "value": self.nod_count,
                    "unit": "count",
                    "z_score": None 
                }
            }
        }
        
        # 리셋
        self.reset_stats()
        
        return result