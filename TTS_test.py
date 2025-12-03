from elevenlabs import ElevenLabs, VoiceSettings
from typing import IO
from io import BytesIO

print (" _____________ ELEVENLABS TTS TEST _____________ ")

class MakeTTSMain:
    def __init__(self):
        print(f" _____________ INIT _____________")
        self.client = ElevenLabs(api_key="sk_6cf3b9c7908ae81a9dfe55a54ff7e00ab642db33e0d165ed")

        # 텍스트를 음성으로 변환
        returnVal = self.TextToSpeech("제목을 입력하세요.")
        print(returnVal)

    def TextToSpeech(self, text):
        try:
            response = self.client.text_to_speech.convert(
                voice_id="AW5wrnG1jVizOYY7R1Oo",  # Adam pre-made voice
                output_format="mp3_22050_32",
                text=text,
                model_id="eleven_multilingual_v2",  # Use multilingual model for better results
                voice_settings=VoiceSettings(
                    stability=0.0,
                    similarity_boost=1.0,
                    style=0.0,
                    use_speaker_boost=True,
                ),
            )

            save_file_path = f"./{text}.mp3"
            with open(save_file_path, "wb") as f:
                for chunk in response:
                    if chunk:
                        f.write(chunk)
            print(f"{save_file_path}: A new audio file was saved successfully!")
            return save_file_path

        except Exception as e:
            print(f" ____________ ERROR TextToSpeech ______________ {e}")
            
if __name__ == "__main__":
    MakeTTSMain()