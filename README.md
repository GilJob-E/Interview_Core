# Giljob-E (길잡이): Real-time Multimodal AI Interview Coach 

## Overview
**Giljob-E**는 단순한 텍스트 챗봇이 아닌, **음성(Prosody), 내용(Lexical), 표정(Facial)** 정보를 실시간으로 분석하여 실제 면접관과 같은 상호작용과 피드백을 제공하는 AI 면접 솔루션입니다.

### Features
* **Latency-Optimized Architecture:** Streaming Pipeline을 통해 발화 후 3초 이내 응답.
* **Multimodal Analysis:** * **Audio:** Pitch, Speaking Rate등 분석 (Confidence & Fluency 측정)
    * **Text:** 형태소 추출 및 어휘 다양성 분석 
    * **Vision:** Eye Tracking & Facial Expression
* **Academic Basis:**
    * Hoque, Mohammed, et al. "Mach: My automated conversation coach." Proceedings of the 2013 ACM international joint conference on Pervasive and ubiquitous computing. 2013: 자동화된 대화 코치 시스템의 실질적인 역량 향상 효과가 입증되었다는 점에 착안
    * Naim, Iftekhar, et al. "Automated analysis and prediction of job interview performance." IEEE Transactions on Affective Computing 9.2 (2018): 
멀티모달 Feature를 활용한 면접 성과 예측 모델링 방법론을 재현



## 🛠 Tech Stack
| Category | Technology |
| :--- | :--- |
| **STT** | Groq Whisper (v3 large) |
| **LLM** | OpenAI API (ChatGPT-4o) |
| **TTS** | ElevenLabs Turbo v2.5 (PCM Streaming) |
| **Server** | Python FastAPI, WebSocket |
| **Client** | Python PyQt6  |

## Installation & Setup

### 1. Prerequisites
* Python 3.10+

### 2. Install Dependencies
```bash
conda env create -f environment.yaml
```

### 3. Environment (수정필요)
Create a .env file in the server/ directory:
- https://elevenlabs.io/app/developers/api-keys
- https://console.groq.com/keys
```
GROQ_API_KEY=your_groq_key
ELEVENLABS_API_KEY=your_elevenlabs_key
```

### 4. FFMPEG (수정필요)
To enable GPU acceleration for Faster-Whisper on Windows:

Download cuDNN v9 and zlibwapi.dll.

Place the .dll files in the server/ directory.


##  Usage
### 1. Start Server
```bash
cd server
uvicorn main:app --reload
```

### 2. Start Client 
```bash
cd client
python client_gui.py
```
---
