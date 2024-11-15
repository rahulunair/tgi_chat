import json
import os
from datetime import datetime
from config import HISTORY_DIR
import time


def ensure_history_dir():
    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR)


def generate_chat_title(messages, client):
    # Create a prompt to generate a title
    conversation = "\n".join(
        [
            f"{'User' if i%2==0 else 'Assistant'}: {msg}"
            for i, msg in enumerate(messages)
        ]
    )
    prompt = f"Based on this conversation, generate a very brief (3-5 words) title that captures the main topic:\n\n{conversation}"

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.7,
        )
        title = response.choices[0].message.content.strip()
        # Remove quotes if present
        title = title.strip("\"'")
        return title
    except:
        # Fallback to timestamp if title generation fails
        return f"Chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def save_chat_history(history, state, title=None, force=False):
    if not history:
        return None

    current_time = time.time()
    if (
        not force
        and state.current_chat_id
        and current_time - state.last_save_time < state.save_interval
    ):
        return state.current_chat_id

    ensure_history_dir()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not state.current_chat_id:
        messages = [msg for pair in history[:3] for msg in pair if msg]
        if not title:
            title = generate_chat_title(messages, state.client)
        chat_id = f"{timestamp}_{title.replace(' ', '_')}"
        state.current_chat_id = chat_id
    else:
        chat_id = state.current_chat_id

    filename = f"{HISTORY_DIR}/{chat_id}.json"

    chat_data = {
        "chat_id": chat_id,
        "title": title or chat_id,
        "history": history,
        "timestamp": timestamp,
        "last_modified": datetime.now().isoformat(),
    }

    # Use atomic write operation
    temp_filename = f"{filename}.tmp"
    try:
        with open(temp_filename, "w") as f:
            json.dump(chat_data, f, indent=4)
        os.replace(temp_filename, filename)  # Atomic operation
    except Exception as e:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        print(f"Error saving chat history: {e}")
        return None

    state.last_save_time = current_time
    return chat_id


def load_chat_history(chat_id):
    filename = f"{HISTORY_DIR}/{chat_id}.json"
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                data = json.load(f)
                # Ensure all required fields exist
                if "history" not in data:
                    return None
                if "chat_id" not in data:
                    data["chat_id"] = chat_id
                if "title" not in data:
                    data["title"] = chat_id
                if "last_modified" not in data:
                    data["last_modified"] = datetime.now().isoformat()
                return data
        except Exception as e:
            print(f"Error loading chat {chat_id}: {e}")
            return None
    return None


def list_chat_histories():
    ensure_history_dir()
    histories = []
    for file in os.listdir(HISTORY_DIR):
        if file.endswith(".json"):
            try:
                with open(f"{HISTORY_DIR}/{file}", "r") as f:
                    data = json.load(f)
                    # Handle legacy files or missing fields
                    chat_id = data.get("chat_id", file.replace(".json", ""))
                    title = data.get("title", chat_id)
                    last_modified = data.get(
                        "last_modified",
                        data.get("timestamp", datetime.now().isoformat()),
                    )
                    histories.append(
                        {
                            "chat_id": chat_id,
                            "title": title,
                            "last_modified": last_modified,
                        }
                    )
            except Exception as e:
                print(f"Error loading chat history {file}: {e}")
                continue

    # Sort by last modified time, newest first
    histories.sort(key=lambda x: x["last_modified"], reverse=True)
    return histories


def delete_chat_history(chat_id):
    filename = f"{HISTORY_DIR}/{chat_id}.json"
    if os.path.exists(filename):
        os.remove(filename)


def build_message_history(history):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    return messages

