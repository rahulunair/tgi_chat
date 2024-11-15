import gradio as gr
from huggingface_hub import InferenceClient
import json
import os
from datetime import datetime
import time
import threading

# Config management
CONFIG_FILE = "chat_config.json"
HISTORY_DIR = "chat_history"

DEFAULT_CONFIG = {
    "endpoints": {
        "TGI Server 1": "https://span-mitsubishi-dependence-opportunities.trycloudflare.com",
    },
    "current_endpoint": "http://127.0.0.1:8081",
    "parameters": {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 1024,
    },
}


class ChatState:
    def __init__(self):
        self.config = self.load_config()
        self.client = InferenceClient(base_url=self.config["current_endpoint"])
        self.is_stopped = False
        self.current_chat_id = None
        self.last_save_time = time.time()
        self.save_interval = 30  # Auto-save every 30 seconds

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

    def update_endpoint(self, endpoint):
        self.config["current_endpoint"] = endpoint
        self.client = InferenceClient(base_url=endpoint)
        self.save_config()

    def update_parameter(self, param_name, value):
        self.config["parameters"][param_name] = value
        self.save_config()

    def stop(self):
        self.is_stopped = True

    def reset(self):
        self.is_stopped = False


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


def inference(message, history, state):
    messages = build_message_history(history)
    messages.append({"role": "user", "content": message})

    partial_message = ""
    output = state.client.chat.completions.create(
        messages=messages, stream=False, **state.config["parameters"]
    )

    for chunk in output:
        if state.is_stopped:
            break
        if chunk.choices[0].delta.content:
            partial_message += chunk.choices[0].delta.content
            yield partial_message


def create_demo():
    state = ChatState()

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        with gr.Tabs() as tabs:
            with gr.Tab("Chat"):
                with gr.Row():
                    chat_list = gr.Dropdown(
                        choices=[
                            f"{h['title']} ({h['chat_id']})"
                            for h in list_chat_histories()
                        ],
                        label="Recent Chats",
                        interactive=True,
                        allow_custom_value=False,
                    )
                    new_chat_btn = gr.Button("New Chat")
                    delete_chat_btn = gr.Button("Delete Chat", variant="stop")

                chatbot = gr.Chatbot(
                    height=600,
                    show_label=False,
                    container=True,
                    show_copy_button=True,
                )

                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Type your message here...",
                        container=False,
                        scale=8,
                        show_label=False,
                    )
                    submit = gr.Button("Send", scale=1, variant="primary")
                    stop = gr.Button("Stop", scale=1)

                examples = gr.Examples(
                    examples=[
                        "Are tomatoes vegetables?",
                        "Explain quantum computing in simple terms",
                        "Write a short poem about AI",
                    ],
                    inputs=msg,
                )

            with gr.Tab("Config"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("## Server Settings")
                        with gr.Row():
                            endpoint_name = gr.Textbox(
                                label="Server Name", placeholder="Enter server name"
                            )
                            endpoint_url = gr.Textbox(
                                label="Server URL", placeholder="Enter server URL"
                            )
                            add_endpoint_btn = gr.Button("Add Server")

                        endpoint_dropdown = gr.Dropdown(
                            choices=list(state.config["endpoints"].items()),
                            label="Active Server",
                            value=state.config["current_endpoint"],
                            type="value",
                        )
                        remove_endpoint_btn = gr.Button("Remove Server")

                    with gr.Column(scale=1):
                        gr.Markdown("## Generation Parameters")
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=state.config["parameters"]["temperature"],
                            step=0.1,
                            label="Temperature",
                        )
                        top_p = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=state.config["parameters"]["top_p"],
                            step=0.05,
                            label="Top P",
                        )
                        max_tokens = gr.Slider(
                            minimum=64,
                            maximum=4096,
                            value=state.config["parameters"]["max_tokens"],
                            step=64,
                            label="Max New Tokens",
                        )

        def user(message, history):
            state.reset()
            return "", history + [[message, None]]

        def bot(history):
            if not history:
                return history

            message = history[-1][0]
            last_save_time = time.time()

            for partial_message in inference(message, history[:-1], state):
                history[-1][1] = partial_message

                # Check if it's time to auto-save
                current_time = time.time()
                if current_time - last_save_time >= state.save_interval:
                    save_chat_history(history, state)
                    last_save_time = current_time

                yield history

            # Final save after completion
            save_chat_history(history, state, force=True)

        def load_selected_chat(selection):
            if not selection:
                return None
            chat_id = selection.split("(")[-1].rstrip(")")
            data = load_chat_history(chat_id)
            if data:
                state.current_chat_id = data["chat_id"]
                return data["history"]
            return None

        def new_chat():
            state.current_chat_id = None
            return None

        def delete_current_chat(selection):
            if not selection:
                return gr.Dropdown(choices=[]), None
            chat_id = selection.split("(")[-1].rstrip(")")
            delete_chat_history(chat_id)
            choices = [f"{h['title']} ({h['chat_id']})" for h in list_chat_histories()]
            if not choices:  # If no chats remain after deletion
                return gr.Dropdown(choices=[]), None
            return gr.Dropdown(choices=choices), None

        # Parameter update functions
        def update_parameter(param_name, value):
            state.update_parameter(param_name, value)

        # Event handlers
        msg_submit = msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        submit.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        stop.click(lambda: state.stop(), None, None, queue=False)
        new_chat_btn.click(new_chat, None, chatbot)
        chat_list.change(load_selected_chat, chat_list, chatbot)
        delete_chat_btn.click(delete_current_chat, chat_list, [chat_list, chatbot])

        # Server config handlers
        add_endpoint_btn.click(
            lambda name, url: (
                state.config["endpoints"].update({name: url}),
                state.save_config(),
                gr.Dropdown(choices=list(state.config["endpoints"].items())),
            )[2],
            [endpoint_name, endpoint_url],
            endpoint_dropdown,
        )
        remove_endpoint_btn.click(
            lambda url: (
                state.config["endpoints"].pop(
                    next(k for k, v in state.config["endpoints"].items() if v == url),
                    None,
                ),
                state.save_config(),
                gr.Dropdown(choices=list(state.config["endpoints"].items())),
            )[2],
            endpoint_dropdown,
            endpoint_dropdown,
        )
        endpoint_dropdown.change(
            lambda x: state.update_endpoint(x), endpoint_dropdown, None
        )

        # Parameter update handlers
        temperature.change(
            lambda x: update_parameter("temperature", x), temperature, None
        )
        top_p.change(lambda x: update_parameter("top_p", x), top_p, None)
        max_tokens.change(lambda x: update_parameter("max_tokens", x), max_tokens, None)

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.queue().launch(server_name="0.0.0.0", share=True, height=800)