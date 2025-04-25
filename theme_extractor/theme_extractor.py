import pandas as pd
import spacy
from gensim import corpora, models
import re
from collections import Counter

# Load the CSV file
file_path = r"C:\Users\shann\PycharmProjects\capstone2025V2\outputs\csv_merger\refined_combined_for_predictive_modelling.csv"
df = pd.read_csv(file_path)

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")


def preprocess_text(text):
    if not isinstance(text, str) or text.strip() == "":
        return []

    # Remove URLs, mentions, hashtags, and special characters
    text = re.sub(r'http\S+|@\w+|#\w+|[^\w\s]', '', text)

    # Process text with spaCy
    doc = nlp(text.lower())

    # Extract lemmatized tokens (nouns, adjectives, verbs)
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop
           and not token.is_punct
           and token.pos_ in ["NOUN", "ADJ", "VERB", "PROPN"]  # Include proper nouns
           and len(token.lemma_) > 2
    ]
    return tokens


# Apply preprocessing and filter empty entries
df['Processed'] = df['Caption'].apply(preprocess_text)
df = df[df['Processed'].apply(len) > 0]  # Remove empty entries

# Create a dictionary and corpus
dictionary = corpora.Dictionary(df['Processed'])
corpus = [dictionary.doc2bow(tokens) for tokens in df['Processed']]

# Train the LDA model with more topics and passes
lda_model = models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=10,  # Increased from 5 to 10 for more diversity
    random_state=42,
    passes=20,  # More iterations for better convergence
    alpha='auto',  # Let the model optimize topic distributions
    eta='auto',  # Let the model optimize word distributions
)

# Print topics with top 5 keywords
for idx, topic in lda_model.print_topics(-1, num_words=5):
    print(f"Topic {idx}: {topic}\n")


# Dynamically assign topic names based on top keywords
def get_topic_name(lda_model, topic_id, n_words=3):
    words = lda_model.show_topic(topic_id, topn=n_words)
    return "_".join([word[0] for word in words])  # e.g., "workout_endorphin_health"


# Assign dominant topic and label
df['Dominant_Topic'] = [max(lda_model[doc], key=lambda x: x[1])[0] for doc in corpus]
df['Topic_Label'] = df['Dominant_Topic'].apply(lambda x: get_topic_name(lda_model, x))

# Save results
output_path = r"C:\Users\shann\PycharmProjects\capstone2025V2\outputs\topic_modeling_results.csv"
df.to_csv(output_path, index=False)