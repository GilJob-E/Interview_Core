import sys
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

# [필수] 모듈 경로 강제 설정 (ko_liwc 등 내부 import 문제 해결용)
current_dir = os.path.dirname(os.path.abspath(__file__))
modules_path = os.path.join(current_dir, 'modules')
lexical_path = os.path.join(modules_path, 'lexical')

if modules_path not in sys.path:
    sys.path.append(modules_path)
if lexical_path not in sys.path:
    sys.path.append(lexical_path)

# 래퍼 모듈 임포트
from modules.prosody_wrapper import ProsodyWrapper
from modules.vision_wrapper import VisionWrapper
from modules.lexical_wrapper import LexicalWrapper 

class MultimodalAnalyzer:
    def __init__(self):
        print("\n[System] Initializing Multimodal Analysis Engine...")
        
        # 1. 각 모듈 초기화
        self.prosody = ProsodyWrapper()
        self.vision = VisionWrapper()
        self.lexical = LexicalWrapper() 
        
        # 병렬 처리를 위한 쓰레드 풀
        self.executor = ThreadPoolExecutor(max_workers=3)
        print("[System] All Engines Ready.\n")

    async def analyze_turn(self, audio_bytes: bytes, text_data: str, video_frames: list, duration: float):
        """
        한 턴(Turn)의 데이터를 받아 3개 모듈을 비동기 병렬로 분석하고 결과를 통합 반환.
        """
        loop = asyncio.get_running_loop()

        # 2. 비동기 병렬 실행 (Non-blocking)
        # 각 분석 작업은 CPU 연산이 주가 되므로 ThreadPoolExecutor에서 실행
        tasks = [
            loop.run_in_executor(self.executor, self.prosody.analyze, audio_bytes),
            loop.run_in_executor(self.executor, self.vision.analyze, video_frames),
            loop.run_in_executor(self.executor, self.lexical.analyze, text_data, duration)
        ]

        # 3. 결과 대기 (모든 분석이 끝날 때까지 Await)
        # 순서: Prosody, Vision, Lexical
        results = await asyncio.gather(*tasks)
        p_res, v_res, l_res = results

        # 4. 데이터 통합 (Aggregation)
        # 점수 계산 로직은 제거하고, Raw Feature + Z-Score 위주로 구성
        analysis_result = {
            "metadata": {
                "duration_sec": round(duration, 2),
                "text_length": len(text_data)
            },
            "multimodal_features": {
                "audio": p_res.get("features", {}),  # Pitch, Intensity...
                "video": v_res.get("features", {}),  # Eye Contact, Smile...
                "text": l_res.get("features", {})    # Wpsec, Fillers...
            },
            # 에러 로그 (디버깅용)
            "status": {
                "audio": "error" if "error" in p_res else "ok",
                "video": "error" if "error" in v_res else "ok",
                "text": "error" if "error" in l_res else "ok"
            }
        }
        
        # 5. 간단한 로그 출력
        try:
            log_pitch = analysis_result['multimodal_features']['audio'].get('pitch', {}).get('value', 'N/A')
            log_eye = analysis_result['multimodal_features']['video'].get('eye_contact', {}).get('value', 'N/A')
            log_wpm = analysis_result['multimodal_features']['text'].get('wpsec', {}).get('value', 'N/A')
            print(f"[Analysis] Pitch:{log_pitch}Hz | Eye:{log_eye} | Speed:{log_wpm}wps")
        except:
            pass
        
        return analysis_result