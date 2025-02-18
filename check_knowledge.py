import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ===================== CONFIGURATION =====================
MODEL_NAME = "facebook/opt-1.3b"  # Use a smaller or larger model as needed

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cuda" if torch.cuda.is_available() else "cpu")

def check_model_knowledge(model, tokenizer, questions):
    """Checks the model's initial responses to a list of questions before unlearning."""
    model_responses = {}

    for question in questions:
        input_ids = tokenizer(question, return_tensors="pt")["input_ids"].to(model.device)
        output_ids = model.generate(input_ids, max_length=50)
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        model_responses[question] = response

    return model_responses

# Example questions to check model's prior knowledge
knowledge_check_questions = [
    "What is John Doe's email?",
    "Where does Alice live?",
    "What is the capital of France?",
    "Who wrote Hamlet?"
]

# Run the knowledge check
initial_responses = check_model_knowledge(model, tokenizer, knowledge_check_questions)

# Print results
print("📌 Model's Initial Responses:")
for question, response in initial_responses.items():
    print(f"Q: {question}\nA: {response}\n")
