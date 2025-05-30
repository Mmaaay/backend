from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "silma-ai/SILMA-9B-Instruct-v1.0"
SAVE_PATH = f"/home/mazen/coding/Quran-back/backend/models/text-generation/"

def download_model():
    # Download and save model
    model = AutoModel.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Save both model and tokenizer
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)

if __name__ == "__main__":
    download_model()