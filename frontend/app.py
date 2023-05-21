import time

import gradio as gr
import requests


def chat_with_bot(message: str) -> str:
    response = requests.post("http://localhost:8000/chat/", json={"query": message})
    if response.status_code == 200:
        return response.json().get("answer")
    else:
        return f"Error: Unabled to get response from the bot. (status code {response.status_code})"


def start_gradio():
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        msg = gr.Textbox(
            show_label=False,
            placeholder="Enter text and press enter",
        )
        clear = gr.Button("Clear")

        def user(user_message, history):
            return "", history + [[user_message, None]]

        def bot(history):
            bot_message = chat_with_bot(history[-1][0])
            history[-1][1] = ""
            for character in bot_message:
                history[-1][1] += character
                time.sleep(0.05)
                yield history

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.queue()
    url = demo.launch()
    print(f"Gradio app running at {url}")
