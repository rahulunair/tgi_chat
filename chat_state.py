from huggingface_hub import InferenceClient
import time
from config import CONFIG_FILE, DEFAULT_CONFIG
import json
import os

from history_manager import build_message_history, save_chat_history

class ChatState:
    def __init__(self):
        self.config = self.load_config()
        self.client = InferenceClient(base_url=self.config["current_endpoint"])
        self.is_stopped = False
        self.current_chat_id = None
        self.last_save_time = time.time()
        self.save_interval = 30  # Auto-save every 30 seconds
        self.pending_save = False
        self.last_save_attempt = 0
        self.max_save_retries = 3

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
                # Ensure all default parameters exist
                if "parameters" not in config:
                    config["parameters"] = DEFAULT_CONFIG["parameters"]
                for key in DEFAULT_CONFIG["parameters"]:
                    if key not in config["parameters"]:
                        config["parameters"][key] = DEFAULT_CONFIG["parameters"][key]
                return config
        return DEFAULT_CONFIG.copy()

    def save_config(self):
        with open(CONFIG_FILE, "w") as f:
            json.dump(self.config, f, indent=4)

    def update_endpoint(self, endpoint: str) -> None:
        try:
            self.client = InferenceClient(base_url=endpoint)
            self.config["current_endpoint"] = endpoint
            self.save_config()
        except Exception as e:
            print(f"Error updating endpoint: {e}")
            # Revert to previous endpoint if update fails
            self.client = InferenceClient(base_url=self.config["current_endpoint"])

    def update_parameter(self, param_name, value):
        try:
            # Validate parameter ranges
            if param_name == "temperature" and not (0 <= value <= 2):
                raise ValueError("Temperature must be between 0 and 2")
            elif param_name == "top_p" and not (0 <= value <= 1):
                raise ValueError("Top P must be between 0 and 1")
            elif param_name == "max_tokens" and not (64 <= value <= 4096):
                raise ValueError("Max tokens must be between 64 and 4096")
            
            self.config["parameters"][param_name] = value
            self.save_config()
        except Exception as e:
            print(f"Error updating parameter {param_name}: {e}")

    def stop(self):
        self.is_stopped = True

    def reset(self):
        self.is_stopped = False

    def schedule_save(self):
        self.pending_save = True
        
    def check_pending_saves(self, history):
        current_time = time.time()
        if (self.pending_save and 
            current_time - self.last_save_attempt >= 5):  # 5 second cooldown
            success = save_chat_history(history, self, force=True)
            if success:
                self.pending_save = False
            self.last_save_attempt = current_time

    def validate_message(self, message: str) -> tuple[bool, str]:
        if not message or message.isspace():
            return False, "Message cannot be empty"
        if len(message) > 3999:
            return False, "Message too long (max 4096 characters)"
        return True, ""

def inference(message, history, state, max_retries=3):
    for attempt in range(max_retries):
        try:
            messages = build_message_history(history)
            messages.append({"role": "user", "content": message})
            
            print(f"\n=== Starting inference ===")
            print(f"Time: {time.strftime('%H:%M:%S')}")
            print("Sending request to TGI server...")
            
            start_time = time.time()
            output = state.client.chat.completions.create(
                messages=messages, 
                stream=True,  # Keep streaming enabled
                **state.config["parameters"]
            )
            
            print(f"Got response stream after {time.time() - start_time:.2f} seconds")
            
            partial_message = ""
            for chunk in output:
                if state.is_stopped:
                    break
                if chunk.choices[0].delta.content:
                    partial_message += chunk.choices[0].delta.content
                    yield partial_message
            break
            
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error in inference after {max_retries} attempts: {e}")
                yield f"Error: Failed to generate response after {max_retries} attempts. {str(e)}"
            else:
                print(f"Attempt {attempt + 1} failed, retrying...")
                time.sleep(1)
