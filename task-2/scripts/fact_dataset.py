### This dataset was made using Claude 3.7 Sonnet using the query: """Give me 100 facts (short facts) and 100 queries, so they are query context pairs where the fact helps answer the query"

import json
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Create data directory
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# 100 short fact-query pairs
fact_query_pairs = [
    # Science
    {"fact": "Water boils at 100 degrees Celsius at sea level.", "query": "What is the boiling point of water?"},
    {"fact": "The human body has 206 bones.", "query": "How many bones are in the human body?"},
    {"fact": "DNA stands for deoxyribonucleic acid.", "query": "What does DNA stand for?"},
    {"fact": "Jupiter is the largest planet in our solar system.", "query": "Which planet is the largest in our solar system?"},
    {"fact": "Sound cannot travel through a vacuum.", "query": "Can sound travel through empty space?"},
    {"fact": "Diamond is the hardest natural material.", "query": "What is the hardest natural substance on Earth?"},
    {"fact": "The speed of light is approximately 299,792,458 meters per second.", "query": "How fast does light travel?"},
    {"fact": "Photosynthesis converts carbon dioxide into oxygen.", "query": "What process do plants use to make oxygen?"},
    {"fact": "The Earth's core is primarily made of iron and nickel.", "query": "What elements make up Earth's core?"},
    {"fact": "Penicillin was discovered by Alexander Fleming in 1928.", "query": "Who discovered penicillin?"},
    
    # History
    {"fact": "The United States declared independence in 1776.", "query": "When did the American colonies declare independence?"},
    {"fact": "The first Moon landing occurred on July 20, 1969.", "query": "When did humans first walk on the Moon?"},
    {"fact": "The Roman Empire fell in 476 CE.", "query": "When did the Western Roman Empire collapse?"},
    {"fact": "The Great Wall of China is over 13,000 miles long.", "query": "How long is the Great Wall of China?"},
    {"fact": "The Magna Carta was signed in 1215.", "query": "When was the Magna Carta signed?"},
    {"fact": "World War II ended in 1945.", "query": "What year did World War II end?"},
    {"fact": "Cleopatra was the last pharaoh of ancient Egypt.", "query": "Who was the final ruler of ancient Egypt?"},
    {"fact": "The Berlin Wall fell in 1989.", "query": "When did the Berlin Wall come down?"},
    {"fact": "The printing press was invented by Johannes Gutenberg around 1440.", "query": "Who invented the printing press?"},
    {"fact": "Christopher Columbus reached the Americas in 1492.", "query": "When did Columbus first reach the New World?"},
    
    # Geography
    {"fact": "Mount Everest is the tallest mountain on Earth.", "query": "What is the highest mountain in the world?"},
    {"fact": "The Amazon is the largest rainforest on Earth.", "query": "What is the biggest rainforest in the world?"},
    {"fact": "The Nile is the longest river in the world.", "query": "Which river has the greatest length?"},
    {"fact": "Russia is the largest country by land area.", "query": "What is the biggest country by area?"},
    {"fact": "Vatican City is the smallest country in the world.", "query": "What is the world's smallest independent state?"},
    {"fact": "The Pacific Ocean is the largest ocean on Earth.", "query": "Which is the biggest ocean?"},
    {"fact": "The Sahara is the largest hot desert on Earth.", "query": "What is the biggest non-polar desert?"},
    {"fact": "Tokyo is the most populous city in the world.", "query": "Which city has the most people?"},
    {"fact": "Lake Baikal is the deepest lake in the world.", "query": "What is the world's deepest lake?"},
    {"fact": "The Dead Sea is the lowest point on land.", "query": "What is the lowest elevation on Earth's surface?"},
    
    # Literature & Arts
    {"fact": "Shakespeare wrote 37 plays.", "query": "How many plays did Shakespeare write?"},
    {"fact": "The Mona Lisa was painted by Leonardo da Vinci.", "query": "Who painted the Mona Lisa?"},
    {"fact": "Don Quixote is considered the first modern novel.", "query": "What is often called the first modern novel?"},
    {"fact": "The Odyssey was written by Homer.", "query": "Who wrote The Odyssey?"},
    {"fact": "Vincent van Gogh cut off his own ear.", "query": "Which famous painter cut off his ear?"},
    {"fact": "To Kill a Mockingbird was written by Harper Lee.", "query": "Who is the author of To Kill a Mockingbird?"},
    {"fact": "The Great Pyramid of Giza is the oldest Seven Wonder still standing.", "query": "Which of the Seven Wonders of the Ancient World still exists?"},
    {"fact": "Pablo Picasso co-founded the Cubist movement.", "query": "Which artist helped create Cubism?"},
    {"fact": "Jane Austen wrote Pride and Prejudice.", "query": "Who wrote Pride and Prejudice?"},
    {"fact": "Beethoven composed nine symphonies.", "query": "How many symphonies did Beethoven compose?"},
    
    # Technology
    {"fact": "ARPANET was the predecessor to the internet.", "query": "What was the early network that became the internet?"},
    {"fact": "The first iPhone was released in 2007.", "query": "When did Apple launch the first iPhone?"},
    {"fact": "Alan Turing is considered the father of computer science.", "query": "Who is often called the father of computer science?"},
    {"fact": "HTML stands for Hypertext Markup Language.", "query": "What does HTML stand for?"},
    {"fact": "Bitcoin was created by Satoshi Nakamoto in 2009.", "query": "Who created Bitcoin and when?"},
    {"fact": "The first successful airplane was built by the Wright brothers.", "query": "Who built the first working airplane?"},
    {"fact": "Tim Berners-Lee invented the World Wide Web.", "query": "Who created the World Wide Web?"},
    {"fact": "The first computer mouse was invented by Douglas Engelbart.", "query": "Who invented the computer mouse?"},
    {"fact": "The first video game console was the Magnavox Odyssey.", "query": "What was the first home video game console?"},
    {"fact": "Python was created by Guido van Rossum.", "query": "Who developed the Python programming language?"},
    
    # Sports
    {"fact": "The modern Olympic Games began in 1896.", "query": "When did the modern Olympics start?"},
    {"fact": "A marathon is 26.2 miles long.", "query": "How long is a marathon race?"},
    {"fact": "The World Cup is held every four years.", "query": "How often does the FIFA World Cup occur?"},
    {"fact": "Cricket is the national sport of England.", "query": "What is the national sport of England?"},
    {"fact": "Michael Phelps won 23 Olympic gold medals.", "query": "Who has the most Olympic gold medals?"},
    {"fact": "Golf originated in Scotland in the 15th century.", "query": "Where and when did golf begin?"},
    {"fact": "A standard chess board has 64 squares.", "query": "How many squares are on a chess board?"},
    {"fact": "Basketball was invented by James Naismith in 1891.", "query": "Who invented basketball and when?"},
    {"fact": "The Stanley Cup is awarded in hockey.", "query": "Which sport awards the Stanley Cup?"},
    {"fact": "The Tour de France began in 1903.", "query": "When was the first Tour de France cycling race?"},
    
    # Food & Drink
    {"fact": "Sushi originated in Japan.", "query": "Where did sushi come from?"},
    {"fact": "Champagne can only come from the Champagne region of France.", "query": "Where must true champagne be produced?"},
    {"fact": "Coffee beans are the seeds of a fruit called a coffee cherry.", "query": "What are coffee beans actually?"},
    {"fact": "Chocolate is made from the seeds of the cacao tree.", "query": "What plant does chocolate come from?"},
    {"fact": "Honey never spoils.", "query": "Which food can last forever without going bad?"},
    {"fact": "Pizza originated in Naples, Italy.", "query": "Where was pizza invented?"},
    {"fact": "Vanilla comes from orchids.", "query": "What flower produces vanilla?"},
    {"fact": "Tea is the second most consumed beverage after water.", "query": "What is the world's second most popular drink?"},
    {"fact": "Ketchup was sold as medicine in the 1830s.", "query": "What condiment was once marketed as a health product?"},
    {"fact": "Potatoes are native to South America.", "query": "Where did potatoes originally come from?"},
    
    # Health & Medicine
    {"fact": "The average adult human body contains about 5 liters of blood.", "query": "How much blood does the average person have?"},
    {"fact": "The smallest bone in the human body is the stapes.", "query": "What is the tiniest bone in humans?"},
    {"fact": "The human heart beats about 100,000 times per day.", "query": "How many times does a heart beat daily?"},
    {"fact": "Insulin was discovered in 1921.", "query": "When was insulin discovered?"},
    {"fact": "Humans have 23 pairs of chromosomes.", "query": "How many chromosome pairs do humans have?"},
    {"fact": "Vitamin C prevents scurvy.", "query": "Which vitamin deficiency causes scurvy?"},
    {"fact": "The appendix has no known essential function in humans.", "query": "What is the purpose of the appendix?"},
    {"fact": "The human brain is about 60% fat.", "query": "What percentage of the brain is fat?"},
    {"fact": "Regular handwashing can prevent the spread of diseases.", "query": "What simple hygiene practice prevents most infections?"},
    {"fact": "The cornea is the only part of the body with no blood supply.", "query": "Which body part has no blood vessels?"},
    
    # Animals
    {"fact": "Octopuses have three hearts.", "query": "How many hearts does an octopus have?"},
    {"fact": "A group of lions is called a pride.", "query": "What is a group of lions called?"},
    {"fact": "Bats are the only mammals capable of sustained flight.", "query": "Which mammals can truly fly?"},
    {"fact": "The blue whale is the largest animal to ever exist.", "query": "What is the biggest animal of all time?"},
    {"fact": "Ants can lift up to 50 times their body weight.", "query": "How strong are ants relative to their size?"},
    {"fact": "Cows have four stomach chambers.", "query": "How many stomach compartments do cows have?"},
    {"fact": "A rhinoceros horn is made of keratin.", "query": "What substance forms rhino horns?"},
    {"fact": "Honeybees communicate through dancing.", "query": "How do bees share information with each other?"},
    {"fact": "Cheetahs are the fastest land animals.", "query": "Which animal runs the fastest on land?"},
    {"fact": "Jellyfish have existed for over 650 million years.", "query": "What animal has survived without evolving for hundreds of millions of years?"},
    
    # Miscellaneous
    {"fact": "The Great Barrier Reef is the world's largest coral reef system.", "query": "What is the biggest coral reef on Earth?"},
    {"fact": "A group of crows is called a murder.", "query": "What is the collective noun for crows?"},
    {"fact": "A jiffy is an actual unit of time, equal to 1/100th of a second.", "query": "How long is a jiffy in scientific terms?"},
    {"fact": "LEGO is the largest tire manufacturer in the world by number.", "query": "Which company produces the most tires annually?"},
    {"fact": "The shortest war in history was between Britain and Zanzibar in 1896, lasting 38 minutes.", "query": "What was the briefest war ever fought?"},
    {"fact": "The word 'OK' originated in Boston in the 1830s.", "query": "Where did the term 'OK' come from?"},
    {"fact": "A single lightning bolt can heat the air around it to 30,000°C.", "query": "How hot can lightning make the surrounding air?"},
    {"fact": "Bananas are berries, but strawberries are not.", "query": "Are bananas technically berries?"},
    {"fact": "The Great Pyramid of Giza was the tallest structure until the Eiffel Tower.", "query": "What was the tallest human-made structure for almost 4,000 years?"},
    {"fact": "The first email was sent in 1971.", "query": "When was the first email sent?"}
]

# Extract contexts and queries
contexts = [pair["fact"] for pair in fact_query_pairs]
queries = [pair["query"] for pair in fact_query_pairs]

# Save contexts
contexts_file = os.path.join(data_dir, "short_facts_contexts.json")
with open(contexts_file, 'w', encoding='utf-8') as f:
    json.dump(contexts, f, ensure_ascii=False, indent=2)

# Save queries
queries_file = os.path.join(data_dir, "short_facts_queries.json")
with open(queries_file, 'w', encoding='utf-8') as f:
    json.dump(queries, f, ensure_ascii=False, indent=2)

# Save the full dataset with pairs
pairs_file = os.path.join(data_dir, "short_facts_pairs.json")
with open(pairs_file, 'w', encoding='utf-8') as f:
    json.dump(fact_query_pairs, f, ensure_ascii=False, indent=2)

print(f"Dataset created successfully:")
print(f"- {len(contexts)} contexts saved to {contexts_file}")
print(f"- {len(queries)} queries saved to {queries_file}")
print(f"- Full dataset saved to {pairs_file}")

# Print first 5 examples
print("\nSample fact-query pairs:")
for i in range(5):
    print(f"\nFact: {contexts[i]}")
    print(f"Query: {queries[i]}")

# ----- EMBEDDING GENERATION -----
print("\n----- GENERATING EMBEDDINGS -----")

# Configuration for embeddings
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
OUTPUT_CONTEXT_EMBEDDINGS_FILE = os.path.join(data_dir, "short_facts_embeddings.npy")

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Embedding Model
print(f"Loading embedding model: {EMBED_MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(device).eval()
print("Embedding model loaded.")

# Embedding Function
def get_passage_embeddings(texts):
    """Compute embeddings for passages/contexts."""
    
    inputs = ["passage: " + text for text in texts]
    encoded_input = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
    
    with torch.no_grad():
        model_output = model(**encoded_input)
        embeddings = model_output.last_hidden_state.mean(dim=1)  # Average pool
    
    # Normalize embeddings
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    return embeddings.cpu().numpy()

# Generate Context Embeddings
print("Generating embeddings for contexts...")
context_embeddings = get_passage_embeddings(contexts)
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
print(f"- Average context length (words): {sum(len(c.split()) for c in contexts)/len(contexts):.1f}")
print(f"- Shortest context: {min(contexts, key=lambda x: len(x.split()))}")
print(f"- Longest context: {max(contexts, key=lambda x: len(x.split()))}")