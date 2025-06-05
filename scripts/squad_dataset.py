import json
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from tqdm import tqdm
import random

# Create data directory
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

print("Loading SQuAD dataset...")
# Load the SQuAD dataset (we'll need more than 1000 to find 1000 unique contexts)
squad = load_dataset("squad", split="train[:20000]")
print(f"Loaded {len(squad)} examples from SQuAD dataset.")

# Extract unique contexts
print("Extracting unique contexts...")
unique_contexts = {}  # Use dict to track context -> [questions]
context_to_questions = {}

# First pass: collect all unique contexts and their questions
for example in tqdm(squad, desc="Collecting unique contexts"):
    context = example["context"]
    question = example["question"]
    answer = example["answers"]["text"][0] if example["answers"]["text"] else ""
    
    if context not in context_to_questions:
        context_to_questions[context] = []
    
    context_to_questions[context].append({"question": question, "answer": answer})

print(f"Found {len(context_to_questions)} unique contexts in dataset.")

# If we have more than 1000 contexts, select 1000 randomly
contexts = list(context_to_questions.keys())
if len(contexts) > 1000:
    random.seed(42)  # For reproducibility
    contexts = random.sample(contexts, 1000)
    print(f"Randomly selected 1000 contexts.")
else:
    print(f"Using all {len(contexts)} available unique contexts.")

# For each context, pick one question (the first one)
squad_pairs = []
selected_questions = []
selected_contexts = []

for context in contexts:
    # Pick the first question for this context
    question = context_to_questions[context][0]["question"]
    
    # Add to our lists
    squad_pairs.append({"fact": context, "query": question})
    selected_contexts.append(context)
    selected_questions.append(question)

# Save contexts
contexts_file = os.path.join(data_dir, "squad_contexts.json")
with open(contexts_file, 'w', encoding='utf-8') as f:
    json.dump(selected_contexts, f, ensure_ascii=False, indent=2)

# Save queries
queries_file = os.path.join(data_dir, "squad_queries.json")
with open(queries_file, 'w', encoding='utf-8') as f:
    json.dump(selected_questions, f, ensure_ascii=False, indent=2)

# Save the full dataset with pairs
pairs_file = os.path.join(data_dir, "squad_pairs.json")
with open(pairs_file, 'w', encoding='utf-8') as f:
    json.dump(squad_pairs, f, ensure_ascii=False, indent=2)

print(f"Dataset created successfully:")
print(f"- {len(selected_contexts)} unique contexts saved to {contexts_file}")
print(f"- {len(selected_questions)} unique questions saved to {queries_file}")
print(f"- {len(squad_pairs)} context-question pairs saved to {pairs_file}")

# Print first 5 examples
print("\nSample SQuAD pairs:")
for i in range(min(5, len(squad_pairs))):
    print(f"\nFact (truncated): {selected_contexts[i][:100]}...")
    print(f"Query: {selected_questions[i]}")

# ----- EMBEDDING GENERATION -----
print("\n----- GENERATING EMBEDDINGS -----")

# Configuration for embeddings
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
OUTPUT_CONTEXT_EMBEDDINGS_FILE = os.path.join(data_dir, "squad_embeddings.npy")

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Embedding Model
print(f"Loading embedding model: {EMBED_MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(device).eval()
print("Embedding model loaded.")

# Embedding Function
def get_passage_embeddings(texts, batch_size=32):
    """Compute embeddings for passages/contexts in batches."""
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i:i + batch_size]
        inputs = ["passage: " + text for text in batch_texts]
        encoded_input = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
        
        with torch.no_grad():
            model_output = model(**encoded_input)
            embeddings = model_output.last_hidden_state.mean(dim=1)  # Average pool
        
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)

# Generate Context Embeddings (in batches due to potential memory constraints)
print("Generating embeddings for contexts...")
context_embeddings = get_passage_embeddings(selected_contexts)
print(f"Context embeddings shape: {context_embeddings.shape}")

# Save Context Embeddings
print(f"Saving context embeddings to {OUTPUT_CONTEXT_EMBEDDINGS_FILE}...")
np.save(OUTPUT_CONTEXT_EMBEDDINGS_FILE, context_embeddings)
print("Context embeddings saved successfully!")

# Print statistics
print("\nEmbedding statistics:")
print(f"- Dimensions: {context_embeddings.shape[1]}")
print(f"- Number of contexts: {context_embeddings.shape[0]}")
print(f"- Embedding model: {EMBED_MODEL_NAME}")
