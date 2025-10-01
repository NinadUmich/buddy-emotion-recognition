# llm_module.py
import requests
from config import EXT_API_URL

def get_llm_reply(messages: list) -> str:
    """
    Send the structured chat messages to the LLM server and return its reply.
    messages: a list of dicts with {"role": "system"|"user"|"assistant", "content": str}
    """
    payload = {"messages": messages}
    try:
        response = requests.post(EXT_API_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get("reply", "[No reply]")
    except Exception as e:
        print(f"[LLM API Error] {e}")
        return "[LLM unavailable]"

# --- Convenience wrappers for voice_module ---

def generate_response(prompt: str) -> str:
    """
    Wrap a plain string prompt into the chat format expected by get_llm_reply.
    """
    messages = [{"role": "system", "content": "You are an empathetic conversational agent."},
                {"role": "user", "content": prompt}]
    return get_llm_reply(messages)

def speak(text: str):
    """
    Stub for TTS. Replace with your actual TTS engine.
    """
    #print(f"[TTS OUTPUT] {text}")
