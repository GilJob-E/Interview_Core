import numpy as np
import traceback
from .prosody.prosody_analysis import ProsodyAnalyzerLight

class ProsodyWrapper:
    def __init__(self):
        print("[Prosody] Initializing Analyzer...")
        self.analyzer = ProsodyAnalyzerLight()

    def analyze(self, audio_bytes: bytes):
        """
        Returns:
            dict: Raw Features + Z-Scores (No aggregated scores)
        """
        try:
            if not audio_bytes:
                return {}

            # 1. Bytes -> Numpy (float32)
            audio_data = np.frombuffer(audio_bytes, dtype=np.float32)

            # 2. 분석 수행
            success = self.analyzer.analyze(audio_data, sampling_rate=16000)

            if success:
                # 3. 개별 피처 추출 (Raw Value & Z-Score)
                # prosody_analysis.py 내부 로직을 통해 이미 계산된 값 활용
                
                # 분포 내 위치(Z-Score) 계산을 위해 내부 baseline 참조
                baseline = self.analyzer.baseline_male if self.analyzer.gender == "Male" else self.analyzer.baseline_female
                
                def get_z(val, name):
                    if val is None: return 0.0
                    stat = baseline[name]
                    return (val - stat['mean']) / stat['std']

                result = {
                    "gender": self.analyzer.gender,
                    "features": {
                        "pitch": {
                            "value": round(self.analyzer.mean_pitch, 1),
                            "unit": "Hz",
                            "z_score": round(get_z(self.analyzer.mean_pitch, "mean pitch"), 2)
                        },
                        "intensity": {
                            "value": round(self.analyzer.intensity_mean, 1),
                            "unit": "dB",
                            "z_score": round(get_z(self.analyzer.intensity_mean, "intensityMean"), 2)
                        },
                        "pause_duration": {
                            "value": round(self.analyzer.avg_dur_pause, 2),
                            "unit": "sec",
                            "z_score": round(get_z(self.analyzer.avg_dur_pause, "avgDurPause"), 2)
                        },
                        "f1_bandwidth": {
                            "value": round(self.analyzer.avg_band1, 1),
                            "unit": "Hz",
                            "z_score": round(get_z(self.analyzer.avg_band1, "avgBand1"), 2)
                        },
                        "unvoiced_rate": {
                            "value": round(self.analyzer.percent_unvoiced, 2),
                            "unit": "ratio",
                            "z_score": round(get_z(self.analyzer.percent_unvoiced, "percentUnvoiced"), 2)
                        }
                    }
                }
                return result
            else:
                return {"error": "Analysis Failed"}

        except Exception as e:
            print(f"[Prosody Error] {e}")
            return {"error": str(e)}