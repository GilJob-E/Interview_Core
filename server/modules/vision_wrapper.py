import numpy as np
import cv2
import os
import traceback
from .vision.face import (
    build_face_landmarker, 
    detect_landmarks, 
    compute_gaze_features_ye,
    eye_contact_from_features_ye, 
    smile,
    detect_nod
)

class VisionWrapper:
    def __init__(self):
        print("[Vision] Initializing FaceLandmarker...")
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "vision", "face_landmarker.task")
        
        # 모델 파일 체크
        if not os.path.exists(model_path):
            print(f"[Vision Error] Model file missing at: {model_path}")
        
        self.detector = build_face_landmarker()
        
        # Calibration Default
        self.mean_feat = np.zeros(8, dtype=np.float32)
        # [수정] 표준 편차를 줄여서 민감도 향상 (임시 하드코딩)
        self.std_feat = np.full(8, 0.1, dtype=np.float32) 

        self.THR_EYE = 1.5
        self.THR_HEAD = 2.5
        
        self.EYE_CONTACT_MEAN = 0.70
        self.EYE_CONTACT_STD = 0.15
        self.SMILE_MEAN = 32.93
        self.SMILE_STD = 17.72

    def analyze(self, video_frames: list):
        """
        Returns:
            dict: Raw Visual Features + Z-Scores
        """
        if not video_frames:
            return {}

        try:
            eye_contact_count = 0
            smile_scores = []
            pitch_history = []
            valid_frames = 0
            nod_count = 0
            
            # [수정 2] Latency 개선: 프레임 스킵 (Downsampling)
            # 88프레임 다 분석하면 느림 -> n프레임마다 1번만 분석해도 통계는 충분함
            SKIP_STEP = 3
            
            for i, frame in enumerate(video_frames):
                if i % SKIP_STEP != 0: continue # 스킵
                
                # 1. 랜드마크 추출
                landmarks, blendshapes, pitch, yaw = detect_landmarks(self.detector, frame)
                
                if landmarks:
                    valid_frames += 1
                    
                    # 2. Eye Contact
                    feat = compute_gaze_features_ye(landmarks)
                    
                    # [수정 3] 함수 인자 이름 수정 (thr_h -> thr_eye, thr_v -> thr_head)
                    is_contact, _, _, _ = eye_contact_from_features_ye(
                        feat, 
                        self.mean_feat, 
                        self.std_feat, 
                        thr_eye=self.THR_EYE,   # 수정됨
                        thr_head=self.THR_HEAD  # 수정됨
                    )
                    
                    if is_contact: eye_contact_count += 1
                    
                    # 3. Smile
                    if blendshapes:
                        smile_scores.append(smile(blendshapes) * 100)
                    
                    # 4. Nod
                    # if pitch is not None:
                    #     pitch_history.append(pitch)
                    #     if detect_nod(pitch_history):
                    #         nod_count += 1
                    #         pitch_history = [] 
                    if pitch is not None:
                        pitch_history.append(pitch)
                        # 히스토리 너무 길어지지 않게 관리 (최근 30개만 유지 등)
                        if len(pitch_history) > 30: pitch_history.pop(0)
                        
                        # 매번 검사하지 말고, 스킵 주기마다 검사
                        if detect_nod(pitch_history):
                            nod_count += 1
                            pitch_history = [] # 중복 카운트 방지

            if valid_frames > 0:
                eye_ratio = eye_contact_count / valid_frames
                avg_smile = sum(smile_scores) / len(smile_scores) if smile_scores else 0
                
                z_eye = (eye_ratio - self.EYE_CONTACT_MEAN) / self.EYE_CONTACT_STD
                z_smile = (avg_smile - self.SMILE_MEAN) / self.SMILE_STD

                return {
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
                            "value": nod_count,
                            "unit": "count",
                            "z_score": None 
                        }
                    }
                }
            else:
                return {"error": "No face detected"}

        except Exception as e:
            print(f"[Vision Error] {e}")
            traceback.print_exc()
            return {"error": str(e)}