import os

# Set to False when you want to run real inference
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() in ("true", "1", "yes")

# ASR Configuration (Whisper runs on CPU by default to keep hardware agnostic)
DEVICE = os.getenv("DEVICE", "cpu")
WHISPER_MODEL_PATH = os.getenv("WHISPER_MODEL_PATH", "large-v3")

# Ollama LLM Configuration (Connects to host Ollama running with AMD/NPU acceleration)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")

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