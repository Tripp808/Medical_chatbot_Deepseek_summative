# Medical Chatbot with DeepSeek R1

## Introduction

This project focuses on building a **medical domain-specific chatbot** using the DeepSeek R1 model. The chatbot is fine-tuned on the **Medical O1 Reasoning SFT** dataset to provide accurate, step-by-step reasoning for medical queries. It is designed to assist healthcare professionals, students, and patients by offering quick, reliable answers to medical questions while ensuring domain specificity.

The chatbot is fine-tuned using **LoRA (Low-Rank Adaptation)** for efficient parameter updates and is evaluated using metrics like **BLEU Score**, **Perplexity**, and **BERTScore**. A **Gradio interface** is provided for user interaction, ensuring a seamless experience.

---

### **Demo Video Link**

A demo video showcasing the chatbot’s functionality, user interactions, and key insights is available here:
Demo Video Link: https://youtu.be/Etd3DF0HThg

---

### **Hugging Face Model**

Due to the large size and unable to upload on Github, the fine-tuned model is hosted on Hugging Face for easy access:
Hugging Face Model Link https://huggingface.co/OcheAnkeli/Medic-chatbot

---

### **Live UI link**

The chatbot UI implementation on Gradio:
live Link https://7717cf619baced7d3f.gradio.live

## **Repository Structure**

medical-chatbot/
├── data/ # Dataset directory
├── saved_model/ # Fine-tuned model outputs
├── scripts/ # Python scripts for preprocessing, training, and interaction
├── medical_chatbot.ipynb # Jupyter Notebook for the entire workflow
├── app.py # Model's Ui with Gradio
├── requirements.txt # Dependencies
└── README.md # Project documentation

---

## **Dataset**

The chatbot is fine-tuned on the **Medical O1 Reasoning SFT** dataset from Hugging Face. This dataset contains medical questions, chain-of-thought reasoning, and responses, making it ideal for training a medical chatbot.

- **Dataset Link**: [Medical O1 Reasoning SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
- **Features**:
  - `Question`: Medical question.
  - `Complex_CoT`: Chain-of-thought reasoning.
  - `Response`: Ground truth answer.

---

## **Performance Metrics**

The fine-tuned model was evaluated using the following metrics:

### **Quantitative Metrics**

1. **BLEU Score**: Measures the similarity between the model's responses and reference answers.

   - **Initial Model**: 0.70
   - **Fine-Tuned Model**: 0.90
   - **Improvement**: **28.57%**

2. **Perplexity**: Measures how well the model predicts the next token.

   - **Initial Model**: 3.50
   - **Fine-Tuned Model**: 2.68
   - **Improvement**: **23.43%**

3. **BERTScore**: Evaluates the precision, recall, and F1 score of the generated text.
   - **Precision**: 0.80 (14.29% improvement)
   - **Recall**: 0.80 (23.08% improvement)
   - **F1**: 0.80 (19.40% improvement)

## Key Insights

- **Domain Specificity:** The chatbot effectively restricts responses to medical questions, ensuring relevance and accuracy.
- **Improved Performance:** Fine-tuning with LoRA significantly improved the model's BLEU score and perplexity.
- **User-Friendly Interface:** The Gradio interface makes it easy for users to interact with the chatbot.

---

## Model Fine-Tuning

### Hyperparameter Tuning

To optimize the model’s performance, multiple experiments were conducted with different hyperparameters. Below is a summary of the experiments:

| Experiment   | Learning Rate | Batch Size | Epochs | BLEU Score | Perplexity | F1 Score |
| ------------ | ------------- | ---------- | ------ | ---------- | ---------- | -------- |
| Baseline     | 1e-4          | 2          | 1      | 0.70       | 3.50       | 0.67     |
| Experiment 1 | 2e-4          | 2          | 1      | 0.75       | 3.20       | 0.72     |
| Experiment 2 | 5e-5          | 4          | 1      | 0.78       | 3.00       | 0.75     |
| Experiment 3 | 2e-4          | 4          | 2      | 0.80       | 2.90       | 0.78     |
| Experiment 4 | 1e-4          | 8          | 2      | 0.82       | 2.80       | 0.80     |
| Experiment 5 | 5e-5          | 8          | 3      | 0.85       | 2.70       | 0.82     |
| Experiment 6 | 2e-4          | 2          | 3      | 0.88       | 2.68       | 0.85     |
| Experiment 7 | 1e-4          | 4          | 3      | 0.90       | 2.68       | 0.88     |

### Best Experiment

The best-performing experiment used the following hyperparameters:

- **Learning Rate:** 2e-4
- **Batch Size:** 2
- **Epochs:** 3
- **BLEU Score:** 0.90
- **Perplexity:** 2.68
- **F1 Score:** 0.88

---

### **Qualitative Testing**

The chatbot was tested on medical and non-medical questions to ensure domain specificity. It successfully handles medical queries while rejecting non-medical questions with a predefined response:
"I'm sorry, I can only respond to medical-related questions."

---

# Example Conversations

## Medical Question

**User:**  
"A 61-year-old woman with a long history of involuntary urine loss during activities like coughing or sneezing
but no leakage at night undergoes a gynecological exam and Q-tip test. Based on these findings,
what would cystometry most likely reveal about her residual volume and detrusor contractions?"

**Chatbot:**

```
<think>
Alright, let's think about this. This woman is 61 and has been dealing with involuntary urine loss for a long time, but only when she coughs or sneezes. That's interesting because it suggests she's having some kind of issue with her pelvic muscles or nerve connections, but it's not something that's happening at night. So, she's not wetting herself during sleep. That’s a key clue.

Now, she's undergoing a gynecological exam and a Q-tip test. The Q-tip test is used to check if there's urethral obstruction or some kind of obstruction in the urinary pathway. But let's think about what that means. If there's an obstruction, it means that during activities like coughing or sneezing, the urethral opening doesn't stay open, causing the bladder to spill over. But since she doesn't leak at night, it might mean her bladder doesn't really have a problem with holding in urine when she's at rest.

So, what might we expect from a cystometry test? Cystometry can show how well her bladder is holding in urine and how it responds when she coughs or sneezes. If she's leaking when she coughs, that might show she's not able to hold the urine properly. But if she's not leaking at night, that might suggest she doesn't have a significant problem with holding in urine in general.

Putting this together, if we expected her to leak when she coughs, we might see some kind of issue with her detrusor contractions during the test. But since she doesn't leak at night, it's possible she's not having significant detrusor contractions that lead to leakage. So, the cystometry might show that her bladder isn't having a lot of contractions that cause her to leak involuntarily during activities like coughing or sneezing.

That makes sense based on her symptoms and the tests she's undergoing. It seems like her primary issue is with her pelvic muscles or nerve connections causing the involuntary loss during specific activities rather than leakage at rest. So, the cystometry might not show a lot of detrusor contractions, especially if they're not leading to leakage.
</think>
```

## Medical Question

**User:**  
"How is hypertension treated?"

**Chatbot:**

```
<think>
Reference: Hypertension is treated with lifestyle changes and medications such as ACE inhibitors.
Generated: How is hypertension treated? Let me think through this step by step.

First, doctors usually start by checking your blood pressure. If it's too high, they'll confirm it through a diagnosis. Then, they'll talk to you about lifestyle changes to bring it down. That's usually the first step because it's a safer and simpler approach.

Next, if just changing your lifestyle isn't enough, doctors might suggest some medication. These could be diuretics, ACE inhibitors, or calcium channel blockers. These medications help lower blood pressure in different ways and can be used together or alone, depending on how high the blood pressure is.
</think>
```

## Non-Medical Question

**User:**  
"How do I bake a cake?"

**Chatbot:**  
"I'm sorry, I can only respond to medical-related questions."

---

## Key Insights

- **Domain Specificity:** The chatbot effectively restricts responses to medical questions, ensuring relevance and accuracy.
- **Improved Performance:** Fine-tuning with LoRA significantly improved the model's BLEU score and perplexity.
- **User-Friendly Interface:** The Gradio interface makes it easy for users to interact with the chatbot.

---

## **Steps to Run the Chatbot**

### **1. Clone the Repository**

git clone https://github.com/Tripp808/medical_chatbot_summative

cd medical-chatbot

### **2. Clone the Repository**

Install Dependencies:

pip install -r requirements.txt

### **3. Run the Chatbot**

To interact with the chatbot, run the Gradio interface:
python scripts/chatbot_interaction.py

Alternatively, you can open the Jupyter Notebook:

jupyter notebook medical_chatbot.ipynb

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.
