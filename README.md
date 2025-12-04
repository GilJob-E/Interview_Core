# Giljob-E (길잡이): Real-time Multimodal AI Interview Coach 

## Overview
**Giljob-E**는 단순한 텍스트 챗봇이 아닌, **음성(Prosody), 내용(Lexical), 표정(Facial)** 정보를 실시간으로 분석하여 실제 면접관과 같은 피드백을 제공하는 AI 면접 솔루션입니다.

### Features
* **Latency-Optimized Architecture:** Streaming Pipeline을 통해 발화 후 1.5초 이내 응답.
* **Multimodal Analysis:** * **Audio:** Jitter, Pitch, Speaking Rate 분석 (Confidence & Fluency 측정)
    * **Text:** STAR 구조 및 직무 적합성 분석
    * **Vision:** Gaze Tracking & Head Pose Estimation (구현 예정)
* **Academic Basis:** Naim et al. (IEEE 2018)의 상관관계 분석에 근거한 가중치 평가 시스템.

## 🛠 Tech Stack
| Category | Technology |
| :--- | :--- |
| **STT** | Faster-Whisper (Local GPU Accelerated) + Silero VAD |
| **LLM** | Groq API (Llama-3-70b) |
| **TTS** | ElevenLabs Turbo v2.5 (PCM Streaming) |
| **Server** | Python FastAPI, WebSocket |
| **Client** | Python (SoundDevice, NumPy) |

## Installation & Setup

### 1. Prerequisites
* Python 3.10+
* NVIDIA GPU (CUDA 12.x Recommended)

### 2. Install Dependencies
```bash
pip install -r requirements.txt 
```

### 3. Environment 
Create a .env file in the server/ directory:
- https://elevenlabs.io/app/developers/api-keys
- https://console.groq.com/keys
```
GROQ_API_KEY=your_groq_key
ELEVENLABS_API_KEY=your_elevenlabs_key
```

### 4. DLL Configuration 
To enable GPU acceleration for Faster-Whisper on Windows:

Download cuDNN v9 and zlibwapi.dll.

Place the .dll files in the server/ directory.


##  Usage
### 1. Start Server
```bash
cd server
uvicorn main:app --reload
```

### 2. Start Client (Test)
```bash
cd client
python test_client.py
```
---
