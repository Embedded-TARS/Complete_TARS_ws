# save_command_client.py
import asyncio, json, os, datetime, sys

SERVER_IP = "192.168.0.134"   # 서버 IP
PORT      = 5000
MSG_END   = b"\n"            # 서버와 동일한 구분자

# ──────────────────────────────────────────────────────────────────────
def make_filename(base: str = "latest_command") -> str:
    """
    base가 'latest_command'이면
      - 덮어쓰기는 latest_command.json 로,
      - 버전을 남기고 싶으면 latest_command_YYYYmmdd_HHMMSS.json 로 만들 수 있다.
    """
    # ① 덮어쓰기 버전
    return f"{base}.json"

    # ② 타임스탬프 버전으로 남기려면 아래 두 줄만 사용하고 위 return 삭제
    # ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # return f"{base}_{ts}.json"

# ──────────────────────────────────────────────────────────────────────
async def main() -> None:
    reader, writer = await asyncio.open_connection(SERVER_IP, PORT)
    print(f"🛰️  connected to {SERVER_IP}:{PORT}")

    try:
        while True:
            # 서버가 MSG_END(개행)까지 한 줄 JSON 을 보냄
            raw = await reader.readuntil(MSG_END)
            raw = raw.rstrip(MSG_END)      # 개행 제거

            try:
                data = json.loads(raw)     # bytes → dict
                print("📥  JSON received:", data)

                # 파일로 저장
                fname = make_filename()
                with open(fname, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                print(f"💾  saved to ./{fname}")

                # (선택) 서버에게 ACK 보내기
                # ack = json.dumps({"status": "ok"}).encode() + MSG_END
                # writer.write(ack); await writer.drain()

            except json.JSONDecodeError:
                # JSON 이 아니면 텍스트로 출력
                print("📝  text received:", raw.decode(errors="ignore"))

    except (asyncio.IncompleteReadError, ConnectionResetError):
        print("🔌  server closed connection")

    finally:
        writer.close()
        await writer.wait_closed()

# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
