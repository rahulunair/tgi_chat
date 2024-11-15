import gradio as gr
from huggingface_hub import InferenceClient
import time

# Initialize client
client = InferenceClient(
    base_url="https://span-mitsubishi-dependence-opportunities.trycloudflare.com",
    timeout=30
)

def generate_response(message):
    print("\n=== Starting new request ===")
    print(f"Time: {time.strftime('%H:%M:%S')}")
    print(f"Message: {message}")
    
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message}
        ]
        
        print("Sending request to TGI server...")
        start_time = time.time()
        
        response_stream = client.chat.completions.create(
            messages=messages,
            stream=True,
            temperature=0.7,
            max_tokens=256
        )
        
        partial_message = ""
        for chunk in response_stream:
            if chunk.choices[0].delta.content:
                partial_message += chunk.choices[0].delta.content
                print(f"Partial response received: {len(partial_message)} chars")
                yield partial_message
        
        print(f"Total response time: {time.time() - start_time:.2f} seconds")
        
    except Exception as e:
        error_msg = f"Error: {type(e).__name__} - {str(e)}"
        print(error_msg)
        yield error_msg

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## TGI Server Test")
    
    with gr.Row():
        input_text = gr.Textbox(
            placeholder="Enter your message here...",
            label="Input"
        )
        output_text = gr.Textbox(label="Output")
    
    status = gr.Markdown("Status: Ready")
    
    def on_submit(message):
        try:
            yield "Status: Generating response...", ""  # Initial status
            for response in generate_response(message):
                yield "Status: Generating...", response
            yield "Status: Ready", response  # Final status
        except Exception as e:
            yield f"Status: Error - {str(e)}", "Error occurred"
    
    submit_btn = gr.Button("Submit")
    submit_btn.click(
        on_submit,
        inputs=input_text,
        outputs=[status, output_text],  # Note: status comes first now
        queue=False
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        share=True,
        debug=True
    )