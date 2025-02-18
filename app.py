import logging
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ===================== CONFIGURATION CLASS =====================

import toml
import os
import logging

# ===================== CONFIGURATION CLASS =====================

class Config:
    """Configuration class that loads from `config.toml` if available, otherwise uses defaults."""

    DEFAULTS = {
        "model": {
            "MODEL_NAME": "facebook/opt-1.3b",
            "USE_QUANTIZATION": True
        },
        "api": {
            "API_HOST": "0.0.0.0",
            "API_PORT": 8000
        },
        "evaluation": {
            "MAX_GENERATION_LENGTH": 50,
            "FORGET_THRESHOLD": 0.7
        },
        "logging": {
            "LOG_FILE": "app.log",
            "LOG_LEVEL": "DEBUG"
        }
    }

    @classmethod
    def load_config(cls, filename="config.toml"):
        """Loads configuration from a TOML file or falls back to defaults."""
        if os.path.exists(filename):
            try:
                loaded_config = toml.load(filename)
                return {**cls.DEFAULTS, **loaded_config}  # Merge defaults with loaded values
            except Exception as e:
                print(f"⚠️ Error loading TOML file: {e}. Using defaults.")
        return cls.DEFAULTS  # If file is missing or fails to load

# Load configuration
CONFIG = Config.load_config()

# ===================== SETUP LOGGING =====================

logging.basicConfig(
    filename=CONFIG["logging"]["LOG_FILE"],
    level=getattr(logging, CONFIG["logging"]["LOG_LEVEL"].upper(), logging.DEBUG),
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w"  # Overwrites log file each run; change to "a" to append logs
)

logger = logging.getLogger(__name__)
logger.info("🚀 Application started with loaded configuration!")

# ===================== LOAD MODEL =====================

logger.info(f"📌 Loading model: {Config.MODEL_NAME}")
quantization_config = BitsAndBytesConfig(load_in_8bit=Config.USE_QUANTIZATION)

tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    Config.MODEL_NAME, quantization_config=quantization_config
).to("cuda" if torch.cuda.is_available() else "cpu")

logger.info("✅ Model loaded successfully!")

# ===================== FASTAPI SERVER =====================

app = FastAPI()

class EvaluationRequest(BaseModel):
    forget_data: List[Dict[str, str]]
    retain_data: List[Dict[str, str]]

# ===================== UNLEARNING EVALUATION CLASS =====================

class UnlearningEvaluator:
    """Evaluates Knowledge Forgetting Rate (KFR), Knowledge Retention Rate (KRR),
    and Linguistic Score (LS) after unlearning."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def compute_kfr(self, forget_data):
        """Computes Knowledge Forgetting Rate (KFR)."""
        total, forgotten = len(forget_data), 0
        for q in forget_data:
            input_ids = self.tokenizer(q["question"], return_tensors="pt").to(self.model.device)
            output = self.model.generate(input_ids.input_ids, max_length=Config.MAX_GENERATION_LENGTH)
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)

            if "private" in response or "not available" in response:
                forgotten += 1

        logger.info(f"KFR Computed: {forgotten}/{total} forgotten")
        return forgotten / total

    def compute_krr(self, retain_data):
        """Computes Knowledge Retention Rate (KRR)."""
        total, retained = len(retain_data), 0
        for q in retain_data:
            input_ids = self.tokenizer(q["question"], return_tensors="pt").to(self.model.device)
            output = self.model.generate(input_ids.input_ids, max_length=Config.MAX_GENERATION_LENGTH)
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)

            if q["answer"].lower() in response.lower():
                retained += 1

        logger.info(f"KRR Computed: {retained}/{total} retained")
        return retained / total

    def compute_linguistic_score(self, test_data):
        """Computes Linguistic Score (LS) using Perplexity and Vocabulary Diversity."""
        total_ppl, total_bi, total_hs = 0, 0, 0
        num_samples = len(test_data)

        for q in test_data:
            input_ids = self.tokenizer(q["question"], return_tensors="pt").to(self.model.device)
            output = self.model.generate(input_ids.input_ids, max_length=Config.MAX_GENERATION_LENGTH)
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)

            # Compute Perplexity
            with torch.no_grad():
                outputs = self.model(input_ids.input_ids, labels=input_ids.input_ids)
                loss = outputs.loss
                ppl = torch.exp(loss).item()

            total_ppl += ppl

            # Compute linguistic complexity (Brunet’s Index, Honore’s Statistic)
            words = response.split()
            unique_words = set(words)
            bi = len(unique_words) ** -0.165
            hs = 100 * torch.log(torch.tensor(len(words))) / (1 - len(unique_words) / len(words))

            total_bi += bi
            total_hs += hs.item()

        logger.info(f"Linguistic Score Computed: PPL={total_ppl / num_samples}, BI={total_bi / num_samples}, HS={total_hs / num_samples}")
        return {
            "Perplexity": total_ppl / num_samples,
            "Brunet’s Index": total_bi / num_samples,
            "Honore’s Statistic": total_hs / num_samples
        }

# Initialize evaluator
evaluator = UnlearningEvaluator(model, tokenizer)

# ===================== API ENDPOINTS =====================

@app.post("/evaluate/")
def evaluate_unlearning(request: EvaluationRequest):
    """Evaluate Knowledge Forgetting, Retention, and Linguistic Score."""
    forget_data = request.forget_data
    retain_data = request.retain_data
    test_data = forget_data + retain_data

    logger.info("🔍 Received evaluation request")
    kfr = evaluator.compute_kfr(forget_data)
    krr = evaluator.compute_krr(retain_data)
    ls = evaluator.compute_linguistic_score(test_data)

    result = {
        "Knowledge Forgetting Rate (KFR)": kfr,
        "Knowledge Retention Rate (KRR)": krr,
        "Linguistic Score (LS)": ls
    }

    logger.info(f"✅ Evaluation Complete: {result}")
    return result

# ===================== RUNNING THE SERVER =====================
# Run the server using: `uvicorn app:app --host 0.0.0.0 --port 8000`
