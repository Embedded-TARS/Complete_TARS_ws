import requests
import json
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import whisper
import warnings

# 경고 메시지 무시
warnings.filterwarnings("ignore")

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
    Ollama를 통해 phi4-mini 모델과 대화하는 함수
    
    Args:
        prompt (str): 사용자의 입력 메시지
        
    Returns:
        str: 모델의 응답
    """
    url = "http://localhost:11434/api/generate"
    
    # 시스템 프롬프트 추가하여 짧은 답변 유도
    system_prompt = "You are a concise assistant. Keep your responses brief and to the point, under 2-3 sentences."
    full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
    
    data = {
        "model": "phi4-mini",
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "num_predict": 10,  # 최대 토큰 수 제한
            "temperature": 0.7,  # 창의성과 일관성의 균형
            "top_p": 0.9,  # 더 결정적인 응답을 위해
            "top_k": 40  # 더 집중된 응답을 위해
        }
    }
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        
        result = response.json()
        return result.get('response', '응답을 받지 못했습니다.')
    
    except requests.exceptions.RequestException as e:
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
