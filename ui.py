import gradio as gr
from chat_state import ChatState, inference
from history_manager import (
    save_chat_history, load_chat_history, 
    list_chat_histories, delete_chat_history
)
import time



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

            try:
                message = history[-1][0]
                last_save_time = time.time()
                print("\n=== Starting bot response ===")
                
                for partial_response in inference(message, history[:-1], state):
                    print(f"Received partial response: {len(partial_response)} chars")  # Debug print
                    history[-1][1] = partial_response
                    
                    # Only save periodically to reduce overhead
                    current_time = time.time()
                    if current_time - last_save_time >= state.save_interval:
                        save_chat_history(history, state)
                        last_save_time = current_time
                        
                    yield history

                # Final save after completion
                save_chat_history(history, state, force=True)

            except Exception as e:
                print(f"Error in bot response: {e}")
                history[-1][1] = f"Error: Failed to generate response. {str(e)}"
                yield history

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
        msg_submit = msg.submit(
            user, [msg, chatbot], [msg, chatbot], queue=False
        ).then(
            bot, chatbot, chatbot,
            api_name=False
        )
        submit.click(
            user, [msg, chatbot], [msg, chatbot], queue=False
        ).then(
            bot, chatbot, chatbot,
            api_name=False
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
