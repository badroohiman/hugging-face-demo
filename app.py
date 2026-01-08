from transformers import pipeline
import gradio as gr
import torch

summarizer = pipeline(
    "summarization",
    model="google-t5/t5-small",
    framework="pt",   # <- force PyTorch
    device=-1         # CPU
)

def predict(text):
    out = summarizer(text, max_length=120, min_length=30, do_sample=False)
    return out[0]["summary_text"]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(placeholder="Enter text to summarize...", lines=6),
    outputs="text",
)

demo.launch()
