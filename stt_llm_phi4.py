import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import whisper
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from gtts import gTTS
import os
import json
import requests
import re
import socket
from tars_planning import EnhancedLanePlanner
import asyncio
import websockets
import time
from datetime import datetime

SAMPLE_RATE = 16000
CHANNELS = 1
OUTPUT_FILE = "recorded_audio.wav"
recording = []

# EnhancedLanePlanner 인스턴스 생성
planner = EnhancedLanePlanner()

def audio_callback(indata, frames, time_info, status):
    recording.append(indata.copy())

def record_audio():
    global recording
    recording = []
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback):
        input()

    audio_data = np.concatenate(recording, axis=0)
    wav.write(OUTPUT_FILE, SAMPLE_RATE, audio_data)

def transcribe_audio():
    model = whisper.load_model("tiny.en")
    result = model.transcribe(OUTPUT_FILE)
    return result["text"]

torch.manual_seed(0)
model_path = "microsoft/Phi-4-mini-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

messages = [
    {
        "role": "system",
        "content": (
            "You are a robotic driving assistant that interprets Korean user commands "
            "into structured JSON control instructions for a rover.\n\n"
            "There are only two types of tasks:\n"
            "1. `navigate`: Go to a specific destination with a speed setting.\n"
            "   - Valid destinations: 'home', 'office', 'airport', 'school'\n"
            "   - Valid speeds: 'fast', 'normal' (default), 'slow'\n\n"
            "2. `manual_command`: Direct movement commands.\n"
            "   - Valid commands (mapped to action field):\n"
            "     - 'stop' → 'stop'\n"
            "     - 'forward' → 'go_forward'\n"
            "     - 'backward' → 'go_backward'\n"
            "     - 'left_turn' → 'turn_left'\n"
            "     - 'right_turn' → 'turn_right'\n"
            "     - 'turn_around' → 'turn_around'\n\n"
            "If the user's command is unclear or doesn't match any category, reply politely asking for clarification.\n\n"
            "When receiving trip information, respond with a natural summary of the trip details in English.\n"
            "For example: 'Trip completed! We traveled {distance} meters in {duration} seconds. The fare is {fare} won.'\n\n"
            "You must always respond with:\n"
            "1. A short assistant-style reply in English (e.g., 'Okay, going to school at normal speed.')\n"
            "2. A JSON block enclosed in triple backticks, like this:\n"
            "```\n"
            "{\n"
            '  "task_type": "navigate" | "manual_command" | "unknown",\n'
            '  "action": "navigate_to" | "stop" | "go_forward" | "go_backward" | "turn_left" | "turn_right" | "turn_around" | "",\n'
            '  "parameters": {\n'
            '     "destination": "home" | "office" | "airport" | "school" | null,\n'
            '     "speed": "fast" | "normal" | "slow" | null\n'
            '  }\n'
            "}\n"
            "```\n"
            "Only use the values listed above. If the input is unclear (e.g., 'go anywhere'), then return 'task_type': 'unknown', and ask the user to clarify."
        )
    }
]

generation_args = {
    "max_new_tokens": 128,
    "return_full_text": False,
    #"temperature": 0.7,
    "do_sample": False,
}

print("[Enter] 키를 눌러 녹음을 시작/종료하세요.")

# 서버에 연결
SERVER_IP = "192.168.0.43"  # TCP 서버 IP
PORT = 5000

class STTLLMHandler:
    def __init__(self, host='localhost', port=5001):
        self.host = host
        self.port = port
        self.server = None
        self.clients = set()
        self.is_running = False

    async def start_server(self):
        try:
            self.server = await websockets.serve(self._handle_client, self.host, self.port)
            self.is_running = True
            print(f"STT LLM 서버가 시작되었습니다. 포트: {self.port}")
        except Exception as e:
            print(f"서버 시작 중 오류 발생: {e}")
            raise

    async def _handle_client(self, websocket, path):
        self.clients.add(websocket)
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    # STT 처리 및 LLM 응답 생성
                    response = await self.process_stt_command(data)
                    await websocket.send(json.dumps(response))
                except json.JSONDecodeError:
                    print("잘못된 JSON 형식")
        except websockets.exceptions.ConnectionClosed:
            print("클라이언트 연결 종료")
        finally:
            self.clients.remove(websocket)

    async def process_stt_command(self, data):
        # STT 명령 처리 로직
        try:
            # 여기에 STT 처리 및 LLM 응답 생성 로직 구현
            response = {
                "status": "success",
                "action": "process_command",
                "parameters": data.get("parameters", {})
            }
            return response
        except Exception as e:
            print(f"명령 처리 중 오류: {e}")
            return {"status": "error", "message": str(e)}

    async def stop_server(self):
        print("서버 종료 중...")
        self.is_running = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        print("서버가 종료되었습니다.")

async def main():
    stt_llm_handler = STTLLMHandler()
    await stt_llm_handler.start_server()
    
    try:
        # 서버가 실행 중인 동안 대기
        while stt_llm_handler.is_running:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n서버를 종료합니다...")
    finally:
        await stt_llm_handler.stop_server()

if __name__ == "__main__":
    asyncio.run(main())