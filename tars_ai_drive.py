import requests
import sounddevice as sd
import numpy as np
import wave
import whisper
import json
import re
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

AUDIO_FILE = "user_command.wav"
SAMPLE_RATE = 44100
DURATION = 5  # seconds

def record_audio():
    print("🎙️ 녹음 중... 말하세요")
    audio_data = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    wave_file = wave.open(AUDIO_FILE, 'wb')
    wave_file.setnchannels(1)
    wave_file.setsampwidth(2)
    wave_file.setframerate(SAMPLE_RATE)
    wave_file.writeframes(audio_data.tobytes())
    wave_file.close()
    print("✅ 녹음 완료")

def transcribe_audio():
    print("🔎 Whisper로 음성 텍스트 변환 중...")
    model = whisper.load_model("base")
    result = model.transcribe(AUDIO_FILE, language="ko")
    return result['text']

def get_structured_command_from_model(user_input):
    url = "http://localhost:11434/api/generate"
    system_prompt = """
You are a robotic driving assistant. Your job is to interpret voice commands in Korean or English and convert them into a structured JSON command. Use this format:

{
  "task_type": "navigate" | "manual_command" | "unknown",
  "action": "stop" | "go_forward" | "go_backward" | "turn_left" | "turn_right" | "turn_around" | null,
  "parameters": {
    "speed": "fast" | "normal" | "slow" | null,
    "destination": "school" | "home" | "work" | null
  }
}

Only return JSON between triple backticks.
"""
    full_prompt = f"{system_prompt}\n\nUser: {user_input}\nAssistant:"

    data = {
        "model": "phi4-mini",
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.4,
            "top_p": 0.9,
            "top_k": 40,
            "num_predict": 150
        }
    }

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        text = result.get("response", "")

        # Extract JSON
        match = re.search(r"```(.*?)```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        else:
            return {"task_type": "unknown"}

    except Exception as e:
        print(f"⚠️ Error in LLM call: {e}")
        return {"task_type": "unknown"}

def execute_command_from_json(driver, command):
    task = command.get("task_type")
    action = command.get("action")
    params = command.get("parameters", {})

    if task == "navigate":
        speed_map = {"fast": 0.5, "normal": 0.3, "slow": 0.15}
        linear = speed_map.get(params.get("speed", "normal"), 0.3)
        print(f"🧭 Navigating to {params.get('destination')} at speed '{params.get('speed')}' → {linear}")
        driver.set_velocity(linear, 0.0)

    elif task == "manual_command":
        cmd_map = {
            "stop": (0.0, 0.0),
            "go_forward": (0.3, 0.0),
            "go_backward": (-0.3, 0.0),
            "turn_left": (0.3, 0.3),
            "turn_right": (0.3, -0.3),
            "turn_around": (0.3, -0.5)
        }
        if action in cmd_map:
            linear, angular = cmd_map[action]
            print(f"🎮 Manual Command: {action} → 선속도 {linear}, 각속도 {angular}")
            driver.set_velocity(linear, angular)
        else:
            print("⚠️ Unknown manual command.")

    else:
        print("🤷‍♂️ 명령을 이해하지 못했습니다.")

def handle_user_voice_command(driver):
    record_audio()
    transcript = transcribe_audio()
    print(f"\n🗣️ 사용자: {transcript}")
    command_json = get_structured_command_from_model(transcript)
    print(f"\n📦 파싱된 명령: {json.dumps(command_json, indent=2, ensure_ascii=False)}")
    execute_command_from_json(driver, command_json)

class TarsAIDriver(Node):
    def __init__(self):
        super().__init__('tars_ai_driver')
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.running = True

    def set_velocity(self, linear, angular):
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.cmd_pub.publish(twist)

    def run(self):
        print("🚗 TARS AI Driver 실행 중 (Enter를 눌러 명령)")
        while self.running:
            key = input("\n🔘 [Enter]를 눌러 말하거나 'q'로 종료:")
            if key.strip().lower() == 'q':
                self.set_velocity(0.0, 0.0)
                self.running = False
                break
            handle_user_voice_command(self)

def main():
    rclpy.init()
    driver = TarsAIDriver()
    try:
        driver.run()
    finally:
        driver.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
