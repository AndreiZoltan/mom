import os
import ctranslate2

# Set to False when you want to load real Whisper & LLM models
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() in ("true", "1", "yes")

# Paths & Hardware config
CUDA_AVAILABLE = ctranslate2.get_cuda_device_count() > 0
DEVICE = os.getenv("DEVICE", "cuda" if CUDA_AVAILABLE else "cpu")
# DEVICE = "cuda" if os.getenv("USE_CUDA", "true").lower() == "true" else "cpu"
WHISPER_MODEL_PATH = os.getenv("WHISPER_MODEL_PATH", "/app/models/whisper-large-v3")
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "/app/models/qwen2.5-7b-instruct-q4_k_m.gguf")

# Predefined Hospital Distribution Lists
DISTRIBUTION_LISTS = {
    "Medical": [
        "consiliu.medical@medpark.local",
        "sef.chirurgie@medpark.local",
        "ati.board@medpark.local",
        "v.cebotari@medpark.local",
    ],
    "Executive": [
        "board.executiv@medpark.local",
        "director.general@medpark.local",
        "cfo@medpark.local",
    ],
    "Administrative": [
        "administratie@medpark.local",
        "hr@medpark.local",
        "mentenanta@medpark.local",
    ],
}

# Medpark Doctor Directory for email matching
HOSPITAL_STAFF_DIRECTORY = {
    "Vasile Cebotari": "v.cebotari@medpark.local",
    "Elena Morari": "e.morari@medpark.local",
    "Angela Rusu": "a.rusu@medpark.local",
    "Mihai Grosu": "m.grosu@medpark.local",
    "Alexandr Popov": "a.popov@medpark.local",
}
