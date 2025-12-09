# server/test_vision_standalone.py
import cv2
import time
import sys
import os

# 모듈 경로 잡기
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.vision_wrapper import VisionWrapper

def run_test():
    print("🎥 [Test] Initializing Vision Module Test...")
    wrapper = VisionWrapper()
    
    # 웹캠 열기 (0번 또는 1번)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ Camera not found.")
        return

    print("✅ Camera started. Press 'q' to exit.")
    print("---------------------------------------------------------------")
    print("  RAW DATA DEBUGGING (Real-time)")
    print("---------------------------------------------------------------")

    frame_buffer = []
    BATCH_SIZE = 15  # 0.5초 분량 (30fps 기준) 씩 모아서 분석 시뮬레이션

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            # 화면에 현재 상태 표시 (시각적 확인)
            cv2.imshow("Vision Debug (Press 'q')", frame)
            
            # 버퍼에 추가
            frame_buffer.append(frame)

            # 일정 프레임이 모이면 분석기 돌리기
            if len(frame_buffer) >= BATCH_SIZE:
                # 1. 분석 수행
                result = wrapper.analyze(frame_buffer)
                
                # 2. 결과가 있다면 상세 로그 출력
                if result and "features" in result:
                    feats = result["features"]
                    eye = feats.get("eye_contact", {})
                    nod = feats.get("head_nod", {})
                    
                    # ★ 여기가 핵심: Raw 값을 찍어봅니다.
                    print(f"[Analysis] Frames: {result.get('valid_frames')} | "
                          f"EyeRatio: {eye.get('value')} (Z: {eye.get('z_score')}) | "
                          f"Nods: {nod.get('value')}")
                    
                    # 만약 Eye Ratio가 계속 1.0이면 -> 임계값이 너무 널널한 것
                    # 만약 Nods가 계속 0이면 -> Pitch 변화폭이 임계값보다 작은 것
                
                # 버퍼 초기화 (다음 턴 준비)
                frame_buffer = []

            # 종료 키
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_test()