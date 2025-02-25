import re
import gradio as gr
from unsloth import FastLanguageModel  # Import FastLanguageModel for inference mode

# List of restricted words to ensure the chatbot remains domain-specific
restricted_words = [
    "bake", "cake", "cooking", "recipe", "joke", "capital", "France", "weather",
    "sports", "movie", "music", "travel", "politics", "celebrity", "investment",
    "football", "basketball", "technology", "AI", "art", "fashion", "economy"
]

# Define the model_lora as a global variable
# Ensure this is loaded or defined before running the Gradio interface
# Example:
# model_lora = FastLanguageModel.from_pretrained(...)
# model_lora = FastLanguageModel.get_peft_model(...)

# Function to generate a response using the fine-tuned model
def generate_response(question):
    """
    Generates a response to a medical question using the fine-tuned DeepSeek R1 model.
    If the question contains restricted words, the chatbot will respond with a domain-specific message.

    Args:
        question (str): The user's input question.

    Returns:
        str: The chatbot's response.
    """
    global model_lora  # Access the global model_lora variable

    # Check if the question contains any restricted words
    matched_words = [word for word in restricted_words if re.search(rf"\b{word}\b", question, re.IGNORECASE)]
    
    # If restricted words are found, return a domain-specific message
    if matched_words:
        return "I'm sorry, I can only respond to medical-related questions."
    
    # Format the question for the chatbot
    formatted_question = train_prompt_style.format(question, "", "")
    
    # Tokenize the input question
    inputs = tokenizer([formatted_question], return_tensors="pt", padding=True, truncation=True).to("cuda")
    
    # Set the model to inference mode
    model_lora = FastLanguageModel.for_inference(model_lora)
    
    # Generate a response using the fine-tuned model
    outputs = model_lora.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=1200,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        use_cache=True
    )
    
    # Decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the model's response (after "### Response:")
    _, _, response_text = response.partition("### Response:")
    response = response_text.strip() if response_text else response
    
    return response

# Create the Gradio interface
iface = gr.Interface(
    fn=generate_response,  # Function to generate responses
    inputs=gr.Textbox(lines=2, placeholder="Enter your medical question here..."),  # Input textbox
    outputs="text",  # Output textbox
    title="Medical Chatbot",  # Title of the interface
    description="Ask any medical-related question and get a detailed response from the fine-tuned DeepSeek R1 model.",  # Description
    examples=[  # Example questions for user guidance
        ["What are the symptoms of diabetes?"],
        ["How is hypertension treated?"],
        ["What is the best way to manage asthma?"]
    ],
    theme="default"  # Interface theme
)

# Launch the Gradio interface
iface.launch()