from config import MODE
from llm_module import get_llm_reply
from tts_module import speak
from voice_module import process_voice

if MODE in ["face", "fusion"]:
    from facial_module import process_face
if MODE == "fusion":
    from fusion_module import run_fusion

print(f"🎤 Assistant ready in {MODE.upper()} mode.")
try:
    while True:
        if MODE == "voice":
            voice_out = process_voice()
            transcript = voice_out["transcript"]
            final_emotion = voice_out["speech_emotion"]

        elif MODE == "fusion":
            result = run_fusion()
            transcript = result["transcript"]
            if not transcript:
                print("⚪ No valid transcript, skipping turn.")
                continue
            final_emotion = result["fusion"]["final_emotion"]

        # main just passes transcript + emotion to llm_module
        response = get_llm_reply(transcript, final_emotion)

        # TTS output
        # speak(response)

except KeyboardInterrupt:
    print("🛑 Stopped by user.")