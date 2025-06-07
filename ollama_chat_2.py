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
        tuple: (assistant_reply, command_json) - 대화 응답과 명령 JSON
    """
    try:
        response = client.chat.completions.create(
            model="phi4-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a robotic driving assistant that interprets Korean user commands "
                        "into structured JSON control instructions for a rover.\n\n"
                        "There are only two types of tasks:\n"
                        "1. navigate: Go to a specific destination with a speed setting.\n"
                        "   - Valid destinations: 'home', 'office', 'airport', 'school'\n"
                        "   - Valid speeds: 'fast', 'normal' (default), 'slow'\n\n"
                        "2. manual_command: Direct movement commands.\n"
                        "   - Valid commands (mapped to action field):\n"
                        "     - 'stop' → 'stop'\n"
                        "     - 'forward' → 'go_forward'\n"
                        "     - 'backward' → 'go_backward'\n"
                        "     - 'left_turn' → 'turn_left'\n"
                        "     - 'right_turn' → 'turn_right'\n"
                        "     - 'turn_around' → 'turn_around'\n\n"
                        "If the user's command is unclear or doesn't match any category, reply politely asking for clarification.\n\n"
                        "You must always respond with:\n"
                        "1. A short assistant-style reply in English that EXACTLY matches your JSON command.\n"
                        "   - For navigation: 'Okay, going to [destination] at [speed] speed.'\n"
                        "   - For manual commands: 'Okay, [action].'\n"
                        "   - For unclear commands: 'I'm not sure what you want me to do. Could you please clarify?'\n"
                        "2. A JSON block enclosed in triple backticks, like this:\n"
                        "\n"
                        "{\n"
                        '  "task_type": "navigate" | "manual_command" | "unknown",\n'
                        '  "action": "navigate_to" | "stop" | "go_forward" | "go_backward" | "turn_left" | "turn_right" | "turn_around" | "",\n'
                        '  "parameters": {\n'
                        '     "destination": "home" | "office" | "airport" | "school" | null,\n'
                        '     "speed": "fast" | "normal" | "slow" | null\n'
                        '  }\n'
                        "}\n"
                        "\n"
                        "IMPORTANT: Your verbal response MUST match your JSON command exactly. If you say 'going to school', your JSON must have destination: 'school'.\n"
                        "Only use the values listed above. If the input is unclear (e.g., 'go anywhere'), then return 'task_type': 'unknown', and ask the user to clarify."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            # max_tokens=150
        )
        
        response_text = response.choices[0].message.content
        
        # Extract JSON from response if it exists
        try:
            json_start = response_text.find('```json')
            json_end = response_text.find('```', json_start + 7)
            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start + 7:json_end].strip()
                command_json = json.loads(json_str)
                # Remove JSON part from response text
                assistant_reply = response_text[:json_start].strip()
            else:
                command_json = {
                    "task_type": "unknown",
                    "action": "",
                    "parameters": {
                        "destination": None,
                        "speed": None
                    }
                }
                assistant_reply = response_text
        except json.JSONDecodeError:
            command_json = {
                "task_type": "unknown",
                "action": "",
                "parameters": {
                    "destination": None,
                    "speed": None
                }
            }
            assistant_reply = response_text
            
        return assistant_reply, command_json
    
    except Exception as e:
        return f"에러 발생: {str(e)}", {
            "task_type": "unknown",
            "action": "",
            "parameters": {
                "destination": None,
                "speed": None
            }
        }

def main():
    print("TARS AI 드라이버와 음성 대화를 시작합니다. 종료하려면 'quit' 또는 'exit'를 입력하세요.")
    
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
            
        assistant_reply, command_json = chat_with_phi4(user_input)
        print(f"\n🤖 TARS: {assistant_reply}")
        
        # Print command analysis if it's not unknown
        if command_json["task_type"] != "unknown":
            print("\n📋 명령 분석:")
            print(json.dumps(command_json, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main() 
