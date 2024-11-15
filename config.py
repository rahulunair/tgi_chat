# Config management
CONFIG_FILE = "chat_config.json"
HISTORY_DIR = "chat_history"

DEFAULT_CONFIG = {
    "endpoints": {
        "TGI Server 1": "https://span-mitsubishi-dependence-opportunities.trycloudflare.com",
    },
    "current_endpoint": "https://span-mitsubishi-dependence-opportunities.trycloudflare.com",
    "system_message": "You are a helpful assistant.",
    "parameters": {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 1024,
    },
}
