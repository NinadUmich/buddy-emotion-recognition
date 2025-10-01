FUSED_EMOTIONS = ["neutral", "happy", "sad", "surprise", "anger"]
SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 5   # seconds
EMOTION_BLOB_PATH = r"D:\thesis\oak-examples\neural-networks\face-detection\emotion-recognition\emotion_recognition.blob"
LANGUAGE = "en"
BEAM_SIZE = 1
COMPUTE_TYPE = "float16"
MODE = "voice"   # options: "voice", "face", "fusion"

# Local endpoints
LOCAL_LLM_URL = "http://localhost:8000/completion"   # your local LLM server
SER_API_URL   = "http://localhost:8001/ser"