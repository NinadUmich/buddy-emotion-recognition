# voice_module.py
# Handles audio capture, STT, SER, and scripted emotion-aware activities
# Output format: [SER], [Transcript], [LLM response]
import sounddevice as sd
import numpy as np
from config import SAMPLE_RATE, SER_API_URL, DURATION, CHANNELS, LANGUAGE, BEAM_SIZE, COMPUTE_TYPE
from llm_module import generate_response, speak   # import from llm_module
from faster_whisper import WhisperModel
from transformers import pipeline
import requests

# Load STT model once at import
stt_model = WhisperModel("small.en", device="cuda", compute_type=COMPUTE_TYPE)
# Conversation history buffer
conversation_history = []

# --- Core Functions ---
def capture_audio():
    input("\n🎤 Press Enter to start recording...")
    print("🎙️ Recording...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=CHANNELS, dtype='float32')
    sd.wait()
    print("✅ Recording complete")
    return np.squeeze(audio)

def run_stt(audio):
    try:
        segments, _ = stt_model.transcribe(
            audio,
            language=LANGUAGE,
            beam_size=max(1, BEAM_SIZE)
        )
        transcript = "".join(seg.text for seg in segments).strip()
        return transcript if transcript else "[Unintelligible or silent input]"
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("[STT] OOM during transcription. Retrying with Whisper Medium on GPU.")
            fallback_model = WhisperModel("medium.en", device="cuda", compute_type=COMPUTE_TYPE)
            segments, _ = fallback_model.transcribe(
                audio,
                language=LANGUAGE,
                beam_size=max(1, BEAM_SIZE)
            )
            transcript = "".join(seg.text for seg in segments).strip()
            return transcript if transcript else "[Unintelligible or silent input]"
        else:
            print(f"[STT] Error during transcription: {e}")
            return "[STT error]"

def run_ser(audio):
    """Send audio to external SER server and return (label, confidence)."""
    import soundfile as sf
    import io

    # Save audio to a buffer in WAV format
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    buf.seek(0)

    files = {"file": ("audio.wav", buf, "audio/wav")}
    try:
        resp = requests.post(SER_API_URL, files=files, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["emotion"], data["confidence"]
    except Exception as e:
        print(f"[SER API Error] {e}")
        return "neutral", 0.0

def log_and_listen(prompt_text: str = ""):
    """Capture audio, run STT + SER, log results."""
    if prompt_text:
        speak(prompt_text)

    audio = capture_audio()
    transcript = run_stt(audio)
    emotion = run_ser(audio)

    label, conf = emotion[0], emotion[1]

    # --- Standardized logging ---
    print(f"\n[SER] {label} ({conf:.2f})")
    print(f"\n[Transcript] {transcript}")

    conversation_history.append({"role": "user", "content": transcript})
    return transcript, (label, conf)

def directional_response(direction: str, emotion_label: str = "neutral", role="empathetic companion"):
    """Generate LLM response with history, emotion grounding, and directional prompt."""
    history_text = "\n".join([f"{t['role'].upper()}: {t['content']}" for t in conversation_history])

    prompt = f"""
    You are an {role} guiding a structured emotional interaction.

    The user's detected emotion is: {emotion_label}.
    Acknowledge this emotion directly in your reply.
    Then respond empathetically and naturally to what they said.
    Finally, guide them smoothly into the next activity: {direction}.
    Keep your reply short (2–3 sentences), conversational, and human‑like.
    Do not output lists, numbered steps, or meta‑instructions.
    Do not restart the conversation or re‑introduce yourself.
    Avoid repeating the same question twice.
    Always continue naturally from the conversation so far.

    Conversation so far:
    {history_text}
    """

    response = generate_response(prompt)
    print(f"\n[LLM response] {response}")
    speak(response)
    conversation_history.append({"role": "assistant", "content": response})
    return response

# --- Research-Inspired Activities ---
def activity_emotion_recognition():
    directional_response("Ask the user to say a neutral sentence in different emotions (happy, sad, angry, neutral). Then guess the emotion and state confidence.")

def activity_adaptive_dialogue(emotion_label):
    directional_response("Ask the user how their day is going. Respond based on their emotion.", emotion_label)

def activity_role_swapping():
    directional_response("Tell the user you feel nervous or sad, and ask them to cheer you up with their voice.")

def activity_scenario_based():
    directional_response("Present a short scenario, like 'Imagine you lost your keys,' and ask the user to respond emotionally. Then comment on their tone.")

def activity_task_with_feedback():
    directional_response("Give the user a simple task, like a trivia question. If they sound frustrated, slow down and encourage them.")

# --- Entry Point for main.py ---
#def process_voice():


    """10-minute scripted HRI session with emotion-aware activities."""

    # Greeting
    directional_response("Greet the user warmly and ask their name.")
    _, (label, _) = log_and_listen()

    # Emotion recognition activity
    activity_emotion_recognition()
    log_and_listen()

    # Adptive dialogue
    activity_adaptive_dialogue(label)
    log_and_listen()

    # Role-swapping
    activity_role_swapping()
    log_and_listen()

    # Scenario-based
    activity_scenario_based()
    log_and_listen()

    # Task with feedback
    activity_task_with_feedback()
    log_and_listen()

    # Closing
    directional_response("Thank the user, reflect on how emotions shaped the interaction, and close warmly.")

    return {
        "transcript": transcript,
        "speech_emotion": label,
        "speech_conf": conf,
        "llm_response": response
    }


def process_voice():
    """10-minute scripted HRI session with emotion-aware activities."""
    last_transcript = ""
    last_emotion = "neutral"
    last_conf = 1.0
    last_response = ""
    # Greeting
    last_response = directional_response("Greet the user warmly and ask their name.", last_emotion)
    last_transcript, (last_emotion, last_conf) = log_and_listen()
    # Emotion recognition activity
    last_response = activity_emotion_recognition()
    last_transcript, (last_emotion, last_conf) = log_and_listen()
    # Adaptive dialogue
    last_response = activity_adaptive_dialogue(last_emotion)
    last_transcript, (last_emotion, last_conf) = log_and_listen()
    # Role-swapping
    last_response = activity_role_swapping()
    last_transcript, (last_emotion, last_conf) = log_and_listen()
    # Scenario-based
    last_response = activity_scenario_based()
    last_transcript, (last_emotion, last_conf) = log_and_listen()
    # Task with feedback
    last_response = activity_task_with_feedback()
    last_transcript, (last_emotion, last_conf) = log_and_listen()
    # Closing
    last_response = directional_response("Thank the user, reflect on how emotions shaped the interaction, "
    "and end the session with a clear goodbye. Do not continue the conversation after this.", last_emotion)
    
    return {
        "transcript": last_transcript,
        "speech_emotion": last_emotion,
        "speech_conf": last_conf,
        "llm_response": last_response
    }