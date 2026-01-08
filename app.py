from transformers import pipeline
import gradio as gr

# Force TensorFlow + CPU
summarizer = pipeline(
    "summarization",
    model="google-t5/t5-small",
    framework="tf"   # <- TensorFlow
)

def predict(text):
    result = summarizer(
        text,
        max_length=120,
        min_length=30,
        do_sample=False,
    )
    return result[0]["summary_text"]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        placeholder="Enter text to summarize...",
        lines=6,
        label="Input text",
    ),
    outputs=gr.Textbox(label="Summary"),
    title="TensorFlow Text Summarizer",
)

if __name__ == "__main__":
    demo.launch()
