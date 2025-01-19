from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "sentence-transformers/paraphrase-albert-small-v2"
SAVE_PATH = "/home/mazen/coding/Quran-back/backend/models/embeddings"

def download_model():
    # Download and save model
    model = AutoModel.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Save both model and tokenizer
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)

if __name__ == "__main__":
    download_model()