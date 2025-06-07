import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import whisper
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from accelerate import init_empty_weights, load_checkpoint_in_model, disk_offload
from tars_planning import EnhancedLanePlanner  # Your custom module

# Recording settings
SAMPLE_RATE = 16000
CHANNELS = 1
OUTPUT_FILE = "recorded_audio.wav"
recording = []

def audio_callback(indata, frames, time_info, status):
    recording.append(indata.copy())

def record_audio():
    global recording
    recording = []
    print("녹음 시작... [Enter]로 종료")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback):
        input()
    audio_data = np.concatenate(recording, axis=0)
    wav.write(OUTPUT_FILE, SAMPLE_RATE, audio_data)
    print("녹음 완료.")

def transcribe_audio():
    model = whisper.load_model("tiny.en")  # or "base", "small"
    result = model.transcribe(OUTPUT_FILE)
    return result["text"]

# Load tokenizer
model_path = "microsoft/Phi-4-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model with disk_offload
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(model_path)

device_map = {"": "cpu"}
offload_folder = "offload"

model = load_checkpoint_in_model(
    model,
    checkpoint=model_path,
    device_map=device_map,
    offload_folder=offload_folder,
    offload_state_dict=True
)

# Create pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# EnhancedLanePlanner 인스턴스 생성
planner = EnhancedLanePlanner()

# Instruction template
messages = [
    {
        "role": "system",
        "content": (
            "You are a robotic driving assistant that interprets Korean user commands "
            "into structured JSON control instructions for a rover.\n\n"
            "There are only two types of tasks:\n"
            "1. `navigate`: Go to a specific destination with a speed setting.\n"
            "   - Valid destinations: 'home', 'office', 'airport', 'school'\n"
            "   - Valid speeds: 'fast', 'normal'(default), 'slow')\n\n"
            "2. `manual_command`: Direct movement commands.\n"
            "   - Valid commands: 'stop', 'forward', 'backward', 'left_turn', 'right_turn', 'turn_around'\n\n"
            "If the user's command is unclear or doesn't match any category, reply politely asking for clarification.\n\n"
            "Always respond with:\n"
            "1. A short assistant-style reply in English.\n"
            "2. A JSON output in this format:\n"
            "```\n"
            "{\n"
            '  "task_type": "navigate" | "manual_command" | "unknown",\n'
            '  "action": "navigate_to" | "stop" | "go_forward" | "go_backward" | "turn_left" | "turn_right" | "turn_around" | "",\n'
            '  "parameters": {\n'
            '     "destination": "home" | "office" | "airport" | "school" | null,\n'
            '     "speed": "fast" | "normal" | "slow" | null\n'
            "  }\n"
            "}\n"
            "Only use the above values. If the user says something unclear like 'go anywhere', then ask for clarification and set task_type to 'unknown'."
        )
    }
]

generation_args = {
    "max_new_tokens": 128,
    "return_full_text": False,
    "do_sample": False,
}

# Main loop
print("[Enter] 키를 눌러 녹음을 시작/종료하세요.")
while True:
    user_input = input("[Enter]로 녹음 / 'exit' 입력시 종료: ").strip().lower()
    if user_input == "exit":
        print("대화를 종료합니다.")
        break

    record_audio()
    stt_text = transcribe_audio()
    print(f"You (STT): {stt_text}")

    messages.append({"role": "user", "content": stt_text})
    output = pipe(messages, **generation_args)
    reply = output[0]["generated_text"].strip()
    print(f"TARS: {reply}")

    # LLM의 출력에서 JSON을 전달하여 계획 계산
    speed, steering, deviation = planner.plan(None, None, None, reply)
    print(f"[Planning 결과] 속도: {speed}, 조향: {steering}, 편차: {deviation}")

    messages.append({"role": "assistant", "content": reply})
