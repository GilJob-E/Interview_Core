import os
import subprocess
import numpy as np
import io
import parselmouth
from parselmouth.praat import call
import matplotlib.pyplot as plt
from scipy import stats

try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except ImportError:
    # ffmpeg가 시스템 경로에 있다면 문제없음
    pass

class ProsodyAnalyzerLight:
    def __init__(self):
        # 1. 가중치 (Scoring Weights)
        self.weights = {
            "Overall": {
                "avgBand1": -0.120, "intensityMean": 0.065,
                "percentUnvoiced": -0.076, "avgDurPause": 0 #기존 weight는 -0.090
            },
            "RecommendedHiring": {
                "avgBand1": -0.132, "intensityMean": 0.086,
                "percentUnvoiced": -0.111, "avgDurPause": 0 #기존 weight는 -0.094
            }
        }

        # 2. 분석 결과를 저장할 멤버변수
        self._gender = None
        self._mean_pitch = None
        self._avg_band1 = None
        self._intensity_mean = None
        self._percent_unvoiced = None
        self._avg_dur_pause = None
        self._scores = None

        # 3. 기준 분포 (Baseline Statistics)
        self.baseline_male = {
            'mean pitch': {'mean': 130.1932, 'std': 15.3799},
            'avgBand1': {'mean': 323.3151, 'std': 58.7594},
            'intensityMean': {'mean': 45.4446, 'std': 9.0125},
            'percentUnvoiced': {'mean': 0.3476, 'std': 0.0623},
            'avgDurPause': {'mean': 0.9950, 'std': 0.1761},
        }
        self.baseline_female = {
            'mean pitch': {'mean': 218.5149, 'std': 21.1525},
            'avgBand1': {'mean': 314.1851, 'std': 42.5445},
            'intensityMean': {'mean': 50.3322, 'std': 5.2173},
            'percentUnvoiced': {'mean': 0.2815, 'std': 0.0440},
            'avgDurPause': {'mean': 1.0560, 'std': 0.3177},
        }
    
    # ==================== Properties ====================
    @property
    def gender(self): return self._gender
    
    @property
    def mean_pitch(self): return self._mean_pitch
    
    @property
    def avg_band1(self): return self._avg_band1
    
    @property
    def intensity_mean(self): return self._intensity_mean
    
    @property
    def percent_unvoiced(self): return self._percent_unvoiced
    
    @property
    def avg_dur_pause(self): return self._avg_dur_pause
    
    @property
    def scores(self): return self._scores
    
    def _convert_to_wav_ffmpeg(self, input_path):
        """FFmpeg를 이용해 미디어 파일을 16kHz Mono WAV로 변환"""
        wav_path = "temp_light_analysis.wav"
        if os.path.exists(wav_path): os.remove(wav_path)
        
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "16000", "-ac", "1", "-vn", "-f", "wav", wav_path
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return wav_path
        except:
            return None

    def _extract_features_light(self, sound):
        """핵심 4대 Feature 추출 (Normalization 완료된 Sound 객체 사용)"""
        duration = sound.get_total_duration()

        # [1] Pitch Analysis
        pitch = sound.to_pitch(time_step=0.02, pitch_floor=50.0, pitch_ceiling=500.0)
        pitch_vals = pitch.selected_array['frequency']
        pitch_vals_valid = pitch_vals[pitch_vals >= 50.0]
        mean_pitch = np.mean(pitch_vals_valid) if len(pitch_vals_valid) > 0 else 0
        
        # [2] Intensity Analysis
        intensity = sound.to_intensity()
        mean_int = np.mean(intensity.values[0])

        # [3] Formant Analysis (F1)
        formant = sound.to_formant_burg(time_step=0.02, max_number_of_formants=3, maximum_formant=3000)
        f1_bw_list = []
        times = np.arange(0, duration, 0.02)
        for t in times:
            bw = formant.get_bandwidth_at_time(1, t)
            if not np.isnan(bw):
                f1_bw_list.append(bw)
        avg_band1 = np.mean(f1_bw_list) if f1_bw_list else 0

        # [4] Pause Analysis (TextGrid)
        silence_tier = call(sound, "To TextGrid (silences)", 50.0, 0.0, -35.0, 0.5, 0.1, "silent", "sounding")
        num_intervals = call(silence_tier, "Get number of intervals", 1)
        
        pause_durs = []
        total_silence_dur = 0
        
        for i in range(1, num_intervals + 1):
            label = call(silence_tier, "Get label of interval", 1, i)
            start = call(silence_tier, "Get start time of interval", 1, i)
            end = call(silence_tier, "Get end time of interval", 1, i)
            dur = end - start
            
            if label == "silent":
                pause_durs.append(dur)
                total_silence_dur += dur
        
        avg_dur_pause = np.mean(pause_durs) if pause_durs else 0

        # [5] Unvoiced Rate Correction
        speaking_duration = duration - total_silence_dur
        voiced_duration = len(pitch_vals_valid) * 0.02
        
        if speaking_duration > 0:
            unvoiced_duration = max(0, speaking_duration - voiced_duration)
            percent_unvoiced = unvoiced_duration / speaking_duration
        else:
            percent_unvoiced = 0
            
        percent_unvoiced = min(1.0, max(0.0, percent_unvoiced))

        return {
            "mean pitch": mean_pitch,
            "avgBand1": avg_band1,
            "intensityMean": mean_int,
            "percentUnvoiced": percent_unvoiced,
            "avgDurPause": avg_dur_pause
        }

    def analyze(self, input_data, sampling_rate=16000):
        """
        분석 수행 및 결과를 멤버변수에 저장
        
        Args:
            input_data (str | np.ndarray): 파일 경로(str) 또는 오디오 데이터(np.ndarray)
            sampling_rate (int): 오디오 데이터일 경우 샘플링 레이트 (Default: 16000)
        """
        wav_path_temp = None
        sound = None

        try:
            # 1. 입력 타입 처리 (파일 vs 메모리)
            if isinstance(input_data, str):
                # 파일 경로인 경우 -> FFmpeg 변환
                wav_path_temp = self._convert_to_wav_ffmpeg(input_data)
                if not wav_path_temp: return False
                sound = parselmouth.Sound(wav_path_temp)
            
            elif isinstance(input_data, np.ndarray):
                # NumPy 배열인 경우 -> 메모리 직접 로드 (Efficient!)
                # parselmouth.Sound는 (data, sampling_frequency)를 지원함
                sound = parselmouth.Sound(input_data, sampling_frequency=sampling_rate)
            
            else:
                print("[Error] Unsupported input type. Use file path(str) or numpy array.")
                return False

            # 2. Normalization (-1dB Peak)
            sound.scale_peak(0.89125)
            
            # 3. 특징 추출
            raw_features = self._extract_features_light(sound)
            
        except Exception as e:
            print(f"[Analysis Error] {e}")
            return False
        finally:
            # 임시 파일이 생성되었다면 삭제
            if wav_path_temp and os.path.exists(wav_path_temp):
                os.remove(wav_path_temp)

        # 4. Gender Detection
        if raw_features["mean pitch"] < 175.0:
            baseline = self.baseline_male
            self._gender = "Male"
        else:
            baseline = self.baseline_female
            self._gender = "Female"

        # 5. 결과 저장
        self._mean_pitch = round(raw_features["mean pitch"], 2)
        self._avg_band1 = raw_features["avgBand1"]
        self._intensity_mean = raw_features["intensityMean"]
        self._percent_unvoiced = raw_features["percentUnvoiced"]
        self._avg_dur_pause = raw_features["avgDurPause"]

        # 6. Z-Score Normalization
        normalized = {}
        for key, val in raw_features.items():
            if key in baseline:
                stat = baseline[key]
                mu, sigma = stat['mean'], stat['std']
                if sigma == 0: sigma = 1
                normalized[key] = (val - mu) / sigma

        # 7. Scoring
        self._scores = {}
        for category, weights in self.weights.items():
            total = 0
            for feat, weight in weights.items():
                z_val = normalized.get(feat, 0)
                total += z_val * weight
            self._scores[category] = round(total, 4)

        return True

    def get_zscore_visualization(self, target_feature=None, save_path=None, return_bytes=False):
        """
        Z-score 정규분포 시각화 메서드.
        
        Args:
            target_feature (str, optional): 시각화할 특정 feature 이름. 
                                            None일 경우 4개 feature 전체를 2x2 그리드로 반환.
                                            (예: "intensityMean")
            save_path (str, optional): 이미지 저장 경로.
            return_bytes (bool): True일 경우 이미지 바이트 데이터 반환.
        
        Returns:
            matplotlib.figure.Figure or bytes: 그래프 객체 또는 이미지 바이트.
        """
        if self._scores is None:
            print("[Error] Analysis not performed. Call analyze() first.")
            return None
        
        baseline = self.baseline_male if self._gender == "Male" else self.baseline_female
        
        # 시각화할 feature 목록 결정
        valid_features = ["avgBand1", "intensityMean", "percentUnvoiced", "avgDurPause"]
        
        if target_feature:
            if target_feature not in valid_features:
                print(f"[Error] Invalid feature name. Choose from: {valid_features}")
                return None
            features_to_plot = [target_feature]
            figsize = (7, 5) # 단일 그래프 크기
            layout = (1, 1)
        else:
            features_to_plot = valid_features
            figsize = (14, 10) # 2x2 전체 그래프 크기
            layout = (2, 2)
        
        # 그래프 생성
        fig, axes = plt.subplots(*layout, figsize=figsize)
        if target_feature:
            axes = [axes] # 단일 축도 리스트로 감싸서 반복문 처리 통일
        else:
            axes = axes.flatten()
            fig.suptitle(f'Feature Z-Score Distribution Analysis ({self._gender})', 
                         fontsize=16, fontweight='bold')
        
        colors = {'avgBand1': '#FF6B6B', 'intensityMean': '#4ECDC4', 
                  'percentUnvoiced': '#45B7D1', 'avgDurPause': '#FFA07A'}
        
        feature_labels = {
            "avgBand1": "F1 Bandwidth (Hz)",
            "intensityMean": "Intensity Mean (dB)",
            "percentUnvoiced": "Unvoiced Ratio",
            "avgDurPause": "Pause Duration (s)"
        }

        # Plotting Loop
        for idx, feat in enumerate(features_to_plot):
            ax = axes[idx]
            
            # 값 가져오기
            if feat == "avgBand1": raw_val = self._avg_band1
            elif feat == "intensityMean": raw_val = self._intensity_mean
            elif feat == "percentUnvoiced": raw_val = self._percent_unvoiced
            elif feat == "avgDurPause": raw_val = self._avg_dur_pause
            
            # Z-Score 계산
            stat = baseline[feat]
            mu, sigma = stat['mean'], stat['std']
            z_score = (raw_val - mu) / sigma
            percentile = stats.norm.cdf(z_score) * 100
            
            # 정규분포 곡선 그리기
            x = np.linspace(-4, 4, 1000)
            y = stats.norm.pdf(x)
            
            ax.plot(x, y, 'k-', linewidth=2, label='Normal Distribution')
            ax.fill_between(x, y, alpha=0.1, color='gray')
            
            # 현재값 표시
            color = colors.get(feat, '#333333')
            if -4 <= z_score <= 4:
                y_val = stats.norm.pdf(z_score)
                ax.plot(z_score, y_val, 'o', markersize=12, color=color, 
                        label=f'Z-Score: {z_score:.2f}')
                ax.axvline(z_score, color=color, linestyle='--', linewidth=2, alpha=0.7)
                
                # 누적 확률 영역 음영
                x_fill = x[x <= z_score]
                y_fill = stats.norm.pdf(x_fill)
                ax.fill_between(x_fill, y_fill, alpha=0.3, color=color)
            
            # 꾸미기
            title_text = f"{feature_labels[feat]}\nPercentile: {percentile:.1f}%"
            ax.set_title(title_text, fontweight='bold', fontsize=12)
            ax.set_xlabel('Z-Score', fontsize=10)
            ax.set_ylabel('Probability Density', fontsize=10)
            ax.set_xlim(-4, 4)
            ax.set_ylim(0, max(y) * 1.1)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, loc='upper right')
        
        plt.tight_layout()
        
        # 파일 저장
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Info] Visualization saved to: {save_path}")

        # 바이트 반환 (웹 전송용)
        if return_bytes:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img_bytes = buf.getvalue()
            buf.close()
            plt.close(fig)
            return img_bytes

        return fig

if __name__ == "__main__":
    # Test Block
    analyzer = ProsodyAnalyzerLight()
    
    # 1. 파일 테스트
    test_files = [f for f in os.listdir('.') if f.endswith(('.wav', '.mp4'))]
    if test_files:
        test_file = test_files[0]
        print(f"Testing with File: {test_file}")
        if analyzer.analyze(test_file):
            print(f"Scores: {analyzer.scores}")
            
            # 특정 Feature 시각화 테스트 (예: Intensity)
            analyzer.get_zscore_visualization(target_feature="intensityMean", save_path="intensity_vis.png")
    else:
        print("No test file found.")

    # 2. NumPy 메모리 테스트 (가상 데이터)
    print("\nTesting with NumPy Array (Sine Wave)...")
    sample_rate = 16000
    t = np.linspace(0, 3, sample_rate * 3) # 3초짜리
    # 200Hz 톤 + 노이즈
    dummy_audio = 0.5 * np.sin(2 * np.pi * 200 * t) + 0.1 * np.random.normal(0, 1, t.shape)
    
    if analyzer.analyze(dummy_audio, sampling_rate=sample_rate):
        print(f"Memory Analysis Success!")
        print(f"Mean Pitch: {analyzer.mean_pitch} (Expected ~200Hz)")
        print(f"Scores: {analyzer.scores}")