from transformers import pipeline


def classify_text(texts, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    """Return sentiment predictions for a list of texts."""
    classifier = pipeline("sentiment-analysis", model=model_name)
    return classifier(texts)


def generate_text(prompt, model_name="gpt2", max_length=50):
    """Generate text continuation from a prompt."""
    generator = pipeline("text-generation", model=model_name)
    output = generator(prompt, max_length=max_length, num_return_sequences=1)[0]
    return output["generated_text"]


if __name__ == "__main__":
    sample_texts = ["J'adore les modèles Transformers", "Je n'aime pas la pluie."]
    print(classify_text(sample_texts))

    prompt = "C'était une belle journée,"
    print(generate_text(prompt))
