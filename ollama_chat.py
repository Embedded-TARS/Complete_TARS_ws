import requests
import json
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import whisper
import warnings
from openai import OpenAI

# 경고 메시지 무시
warnings.filterwarnings("ignore")

# Ollama API 설정
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="not-needed"  # Ollama는 API 키가 필요하지 않음
)

SAMPLE_RATE = 16000
CHANNELS = 1
OUTPUT_FILE = "recorded_audio.wav"
recording = []

# Whisper 모델을 전역 변수로 한 번만 로드 (가장 작은 모델 사용)
print("음성 인식 모델을 로드하는 중...")
whisper_model = whisper.load_model("tiny.en")

def audio_callback(indata, frames, time_info, status):
    recording.append(indata.copy())

def record_audio():
    global recording
    recording = []
    print("🎤 음성을 녹음 중입니다... (Enter를 눌러 녹음을 종료하세요)")
    
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback):
        input()  # Enter 키를 누를 때까지 대기
    
    audio_data = np.concatenate(recording, axis=0)
    wav.write(OUTPUT_FILE, SAMPLE_RATE, audio_data)

def transcribe_audio():
    result = whisper_model.transcribe(OUTPUT_FILE)
    return result["text"]

def chat_with_phi4(prompt):
    """
    Ollama를 통해 phi4-mini 모델과 대화하는 함수 (OpenAI 클라이언트 사용)
    
    Args:
        prompt (str): 사용자의 입력 메시지
        
    Returns:
        str: 모델의 응답
    """
    try:
        response = client.chat.completions.create(
            model="phi4-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise assistant. Keep your responses brief and to the point, under 2-3 sentences."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=10
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"에러 발생: {str(e)}"

def main():
    print("phi4-mini와 음성 대화를 시작합니다. 종료하려면 'quit' 또는 'exit'를 입력하세요.")
    
    while True:
        print("\n🔘 [Enter]를 눌러 음성 입력을 시작하세요 (또는 'quit'/'exit' 입력):")
        user_input = input().strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("대화를 종료합니다.")
            break
            
        if not user_input:
            record_audio()
            user_input = transcribe_audio()
            print(f"\n👤 음성 입력: {user_input}")
            
        response = chat_with_phi4(user_input)
        print(f"\n🤖 phi4-mini: {response}")

if __name__ == "__main__":
    main() 
