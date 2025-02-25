import gradio as gr
import re

# Function to generate responses
def generate_response(question):
    matched_words = [word for word in restricted_words if re.search(rf"\b{word}\b", question, re.IGNORECASE)]
    if matched_words:
        return "I'm sorry, I can only respond to medical-related questions."
    
    inputs = tokenizer([question], return_tensors="pt", padding=True, truncation=True).to("cuda")
    outputs = model_lora.generate(**inputs, max_new_tokens=1200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Gradio interface
iface = gr.Interface(
    fn=generate_response, 
    inputs=gr.Textbox(lines=2, placeholder="Enter your medical question here..."), 
    outputs="text",
    title="Medical Chatbot",
    description="Ask any medical-related question and get a detailed response from the fine-tuned DeepSeek R1 model.",
)

iface.launch()