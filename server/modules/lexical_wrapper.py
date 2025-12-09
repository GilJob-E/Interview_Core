import traceback
# ko_liwc 패키지 임포트
try:
    from .lexical.ko_liwc import InterviewAnalyzer
    from .lexical.ko_liwc.scoring import ZScoreNormalizer
    from .lexical.ko_liwc.scoring.normalizer import DEFAULT_FEATURE_STATS
except ImportError:
    print("[Lexical] Import Error - 경로를 확인해주세요.")
    pass

class LexicalWrapper:
    def __init__(self):
        print("[Lexical] Initializing Kiwi & LIWC Analyzer...")
        try:
            # 1. 분석기 초기화
            self.analyzer = InterviewAnalyzer()
            
            # 2. Z-Score 정규화기 초기화 (기본 통계 사용)
            self.normalizer = ZScoreNormalizer(preset_stats=DEFAULT_FEATURE_STATS)
            print("[Lexical] Initialization Complete.")
        except Exception as e:
            print(f"[Lexical Init Error] {e}")
            self.analyzer = None

    def analyze(self, text: str, duration: float):
        """
        Args:
            text: 사용자 발화 텍스트 (STT 결과)
            duration: 발화 길이 (초)
        Returns:
            dict: Raw Features + Z-Scores
        """
        if not self.analyzer:
            return {"error": "Module not initialized"}

        if not text or duration <= 0:
            return {"error": "Invalid input (Empty text or zero duration)"}

        try:
            # 1. 텍스트 분석 수행
            result = self.analyzer.analyze(text, duration=duration)
            
            # 2. Raw Feature 추출
            raw_feats = result.features
            
            # 3. Z-Score 계산 (Transform)
            z_scores = self.normalizer.transform(raw_feats)

            # 4. 결과 반환 (Tier 1 Feature 위주 구성)
            return {
                "features": {
                    "wpsec": {
                        "value": round(raw_feats.get('wpsec', 0), 2),
                        "unit": "words/sec",
                        "z_score": round(z_scores.get('wpsec', 0), 2),
                        "desc": "말하기 속도"
                    },
                    "upsec": {
                        "value": round(raw_feats.get('upsec', 0), 2),
                        "unit": "unique/sec",
                        "z_score": round(z_scores.get('upsec', 0), 2),
                        "desc": "어휘 다양성"
                    },
                    "fillers": {
                        "value": round(raw_feats.get('fpsec', 0), 2),
                        "unit": "fillers/sec",
                        "z_score": round(z_scores.get('fpsec', 0), 2),
                        "desc": "필러(음,어) 사용 빈도"
                    },
                    "quantifier": {
                        "value": round(raw_feats.get('quantifier_ratio', 0), 2),
                        "unit": "ratio",
                        "z_score": round(z_scores.get('quantifier_ratio', 0), 2),
                        "desc": "수량사 사용 비율"
                    },
                    # 추가 지표 (Emotion 등)
                    "positive_emotion": {
                        "value": round(raw_feats.get('pos_emotion_ratio', 0), 2),
                        "unit": "ratio",
                        "z_score": round(z_scores.get('pos_emotion_ratio', 0), 2)
                    }
                }
            }

        except Exception as e:
            print(f"[Lexical Runtime Error] {e}")
            traceback.print_exc()
            return {"error": str(e)}