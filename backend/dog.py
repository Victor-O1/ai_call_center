import asyncio
import os
import websockets
from dotenv import load_dotenv

load_dotenv()
DG_KEY = os.getenv("DEEPGRAM_API_KEY", "")

URL = "wss://api.deepgram.com/v1/listen?model=nova-3&language=en-US&encoding=linear16&sample_rate=16000&channels=1&interim_results=true&smart_format=true&punctuate=true"

def header_variants():
    token = f"Token {DG_KEY}"
    return [
        # variant A: websockets legacy
        {"extra_headers": {"Authorization": token}},
        # variant B: websockets modern
        {"additional_headers": [("Authorization", token)]},
        # variant C: sometimes supported
        {"headers": {"Authorization": token}},
    ]

async def try_connect(kwargs):
    print("\n[TEST] Trying connect args:", kwargs)
    async with websockets.connect(URL, **kwargs) as ws:
        print("✅ CONNECTED OK")
        await ws.send(b"\x00" * 3200)
        print("✅ Sent silence bytes")
        return True

async def main():
    print("\n==============================")
    print("DEEPGRAM WS HANDSHAKE TEST (FIXED)")
    print("==============================")
    print("API key set:", "YES ✅" if DG_KEY else "NO ❌")
    print("API key length:", len(DG_KEY))
    print("URL:", URL)
    print("==============================\n")

    if not DG_KEY:
        print("❌ No API key in env")
        return

    last_error = None

    for variant in header_variants():
        try:
            ok = await try_connect(variant)
            if ok:
                print("\n✅ SUCCESS: Deepgram STT websocket handshake works.")
                return
        except Exception as e:
            last_error = e
            print("❌ Failed:", repr(e))

    print("\n❌ ALL VARIANTS FAILED.")
    print("Last error:", repr(last_error))

asyncio.run(main())
