import sys
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

# [필수] 모듈 경로 강제 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
modules_path = os.path.join(current_dir, 'modules')
lexical_path = os.path.join(modules_path, 'lexical')

if modules_path not in sys.path:
    sys.path.append(modules_path)
if lexical_path not in sys.path:
    sys.path.append(lexical_path)

# 래퍼 모듈 임포트
from modules.prosody_wrapper import ProsodyWrapper
from modules.vision_wrapper import VisionWrapper # (파일명은 그대로 유지)
from modules.lexical_wrapper import LexicalWrapper 

class MultimodalAnalyzer:
    def __init__(self):
        print("\n[System] Initializing Multimodal Analysis Engine...")
        
        # 1. 각 모듈 초기화
        self.prosody = ProsodyWrapper()
        self.vision = VisionWrapper() # 수정된 실시간 래퍼
        self.lexical = LexicalWrapper() 
        
        # 병렬 처리를 위한 쓰레드 풀
        self.executor = ThreadPoolExecutor(max_workers=3)
        print("[System] All Engines Ready.\n")

    def process_vision_frame(self, frame):
        """
        [New] 메인 루프에서 비디오 프레임이 들어올 때마다 호출
        """
        # 비전 처리는 백그라운드 스레드에서 수행 (메인 루프 차단 방지)
        self.executor.submit(self.vision.process_frame, frame)

    async def analyze_turn(self, audio_bytes: bytes, text_data: str, duration: float):
        """
        [Updated] 턴 종료 시 호출. 
        * video_frames 인자가 제거
        """
        loop = asyncio.get_running_loop()

        # 1. Vision 결과 가져오기 
        # flush_stats()를 호출하여 지금까지 쌓인 통계를 가져오고 리셋함
        v_res = self.vision.flush_stats()

        # 2. Audio & Text 비동기 병렬 실행
        tasks = [
            loop.run_in_executor(self.executor, self.prosody.analyze, audio_bytes),
            loop.run_in_executor(self.executor, self.lexical.analyze, text_data, duration)
        ]

        # 3. 결과 대기
        results = await asyncio.gather(*tasks)
        p_res, l_res = results

        # 4. 데이터 통합
        analysis_result = {
            "metadata": {
                "duration_sec": round(duration, 2),
                "text_length": len(text_data)
            },
            "multimodal_features": {
                "audio": p_res.get("features", {}),
                "video": v_res.get("features", {}), # 실시간 분석 결과
                "text": l_res.get("features", {})
            },
            "status": {
                "audio": "error" if "error" in p_res else "ok",
                "video": "error" if "error" in v_res else "ok",
                "text": "error" if "error" in l_res else "ok"
            }
        }
        
        # 로그 출력
        try:
            log_pitch = analysis_result['multimodal_features']['audio'].get('pitch', {}).get('value', 'N/A')
            log_eye = analysis_result['multimodal_features']['video'].get('eye_contact', {}).get('value', 'N/A')
            log_wpm = analysis_result['multimodal_features']['text'].get('wpsec', {}).get('value', 'N/A')
            print(f"[Analysis] Pitch:{log_pitch}Hz | Eye:{log_eye} | Speed:{log_wpm}wps")
        except:
            pass
        
        return analysis_result