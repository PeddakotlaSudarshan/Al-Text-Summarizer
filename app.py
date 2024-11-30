from transformers import BartTokenizer, BartForConditionalGeneration
import gradio as gr

# Load the BART model
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def summarize(text, num_points):
    # Encode input and generate summary
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Split summary into sentences and format as points
    points = summary.split('. ')
    limited_points = points[:num_points]
    formatted_summary = '\n'.join([f"{i + 1}. {point.strip()}" for i, point in enumerate(limited_points) if point.strip()])
    return formatted_summary

iface = gr.Interface(
    fn=summarize,
    inputs=[
        "text", 
        gr.Number(label="Number of points", value=5, precision=0)
    ],
    outputs="text",
    title="Text Summarization with BART",
    description="Enter text and specify the number of points for summarization."
)

iface.launch()
