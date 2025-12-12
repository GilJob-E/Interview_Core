import cv2
import time
import sys
import os

# [필수] 모듈 경로 설정 (server 폴더 내부의 모듈을 가져오기 위함)
# 현재 파일이 server/ 폴더 안에 있다고 가정
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 만약 server 폴더 밖에서 실행한다면 아래 경로 로직이 필요할 수 있음
# sys.path.append(os.path.join(current_dir, 'server'))

from modules.vision_wrapper import VisionWrapper

def run_test():
    print("🎥 [Test] Initializing Real-time Vision Module...")
    
    try:
        wrapper = VisionWrapper()
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        print("Tip: 'face_landmarker.task' 파일 경로를 확인하세요.")
        return
    
    # 웹캠 열기
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ Camera not found.")
        return

    print("✅ Camera started. Press 'q' to exit.")
    print("---------------------------------------------------------------")
    print("  RAW DATA DEBUGGING (Accumulated every 15 frames)")
    print("---------------------------------------------------------------")

    # 배치 사이즈 (로그 출력 주기)
    BATCH_SIZE = 15  
    local_frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            # 1. [핵심] 실시간 프레임 처리 (리스트에 쌓지 않음)
            # 내부적으로 랜드마크를 추출하고 통계 변수에 누적합니다.
            wrapper.process_frame(frame)
            local_frame_count += 1

            # 2. 화면 표시 (비전 작동 여부 확인용)
            # (테스트를 위해 단순히 원본 영상을 보여줍니다. 
            #  실제 박스는 flush된 데이터를 기반으로 그려야 하므로 여기선 생략하거나 단순화)
            cv2.imshow("Vision Debug (Press 'q')", frame)

            # 3. 일정 주기로 통계 Flush 및 출력
            if local_frame_count >= BATCH_SIZE:
                # 지금까지 쌓인 통계 가져오기 & 리셋
                result = wrapper.flush_stats()
                local_frame_count = 0
                
                if result and "features" in result:
                    valid = result.get('valid_frames', 0)
                    feats = result["features"]
                    
                    eye = feats.get("eye_contact", {})
                    smile = feats.get("smile", {})
                    nod = feats.get("head_nod", {})
                    
                    # 로그 출력
                    print(f"[Analysis] Processed: {valid}/{BATCH_SIZE} | "
                          f"EyeRatio: {eye.get('value', 0):.2f} (Z: {eye.get('z_score', 0):.2f}) | "
                          f"Smile: {smile.get('value', 0):.1f} | "
                          f"Nods: {nod.get('value', 0)}")
                    
                    # [디버깅 팁]
                    # - EyeRatio가 0.00이면: 카메라를 안 보고 있거나 임계값이 너무 엄격함.
                    # - Nods가 안 오르면: 고개를 더 크게 끄덕여보세요 (Amplitude 임계값 확인).

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