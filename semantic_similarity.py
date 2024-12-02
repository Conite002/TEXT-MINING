import re
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from difflib import SequenceMatcher

# Prétraitement du texte
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Algorithmes de similarité
def LCS(str1, str2):
    seq_match = SequenceMatcher(None, str1, str2)
    return seq_match.find_longest_match(0, len(str1), 0, len(str2)).size

def jaro_winkler_similarity(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()

def ngram_similarity(str1, str2, n=2):
    str1 = str1.split()
    str2 = str2.split()
    ngrams1 = {tuple(str1[i:i+n]) for i in range(len(str1) - n + 1)}
    ngrams2 = {tuple(str2[i:i+n]) for i in range(len(str2) - n + 1)}
    intersection = ngrams1.intersection(ngrams2)
    return len(intersection) / float(len(ngrams1.union(ngrams2)))

def set_features_similarity(str1, str2):
    set1 = set(str1.split())
    set2 = set(str2.split())
    intersection = set1.intersection(set2)
    return len(intersection) / float(len(set1.union(set2)))

def word_order_similarity(str1, str2):
    words1 = str1.split()
    words2 = str2.split()
    matching_words = sum(1 for word in words1 if word in words2)
    return matching_words / float(max(len(words1), len(words2)))

def apply_syntaxic_similarity(df, algorithms):
    # Pour chaque algorithme, ajouter une colonne vide dans le DataFrame
    for algo_name in algorithms.keys():
        df[algo_name] = 0.0  # Initialiser chaque colonne avec des scores à 0.0

    # Appliquer chaque algorithme de similarité aux lignes du DataFrame
    for index, row in df.iterrows():
        response = row['UResponse']
        original_response = row['OResponse']
        
        for algo_name, algo in algorithms.items():
            similarity_score = algo(response, original_response)
            df.at[index, algo_name] = similarity_score  # Ajouter le score dans la colonne correspondante

    return df

# Définir les algorithmes de similarité
algorithms = {
    'Longest Common Sequence (LCS)': LCS,
    'Jaro-Winkler': jaro_winkler_similarity,
    'N-GRAMM': ngram_similarity,
    'Set features': set_features_similarity,
    'Word Order Similarity': word_order_similarity,
}

# Appliquer la fonction sur votre DataFrame
df_syntaxic = apply_syntaxic_similarity(merged_df, algorithms)
df_syntaxic

# ------------------------------------------------------------------------------------------------------------------
from sklearn.metrics import jaccard_score
import numpy as np
from scipy.spatial.distance import directed_hausdorff

# Calculer l'indice de Jaccard
def ngrams(text, n=2):
    words = text.split()
    return set([' '.join(words[i:i+n]) for i in range(len(words)-n+1)])

def jaccard_index(doc1, doc2, n=3):
    ngrams_doc1 = ngrams(doc1, n)
    ngrams_doc2 = ngrams(doc2, n)
    intersection = ngrams_doc1.intersection(ngrams_doc2)
    union = ngrams_doc1.union(ngrams_doc2)
    return len(intersection) / len(union)

# Classification des similarités
def classify_similarity(similarity):
    if similarity >= 0.75:
        return 'cut'
    elif similarity >= 0.50:
        return 'heavy'
    elif similarity >= 0.25:
        return 'light'
    else:
        return 'Non'
    

def compute_semantic_similarity(df):
    results = []
    for i in range(len(df)):
        doc1 = df['OResponse'][i]
        doc2 = df['UResponse'][i]
        
        # Calculer l'indice de Jaccard
        jaccard = SemanticDistanceDocs(doc1, doc2)
        
 
        # Classer la similarité
        similarity_class = classify_similarity(jaccard)
        
        results.append({
            'User': df['Username'][i],
            'Task': df['Task'][i],
            'WUP': jaccard,
            'Similarity Class': similarity_class
        })

    return pd.DataFrame(results)

# ------------------------------------------------------------------------------------------------------------------
# Calculer les similarités sémantiques
df_semantic = compute_semantic_similarity(merged_df)
import pandas as pd
excel_path = 'c:/Users/dscon/Documents/COURS UM6P/S3/TEXT-MINING/Corpus_Plagiat/Corpus_Plagiat/corpus-final09.xls'
sheet_name = 'File list'
file_list_df = pd.read_excel(excel_path, sheet_name=sheet_name)


import pandas as pd


# Fonction pour calculer la précision
def calculate_precision(similarities, true_categories):
    predicted_categories = [classify_similarity(sim) for sim in similarities]
    
    results_df = pd.DataFrame({
        'predicted': predicted_categories,
        'true': true_categories
    })
    
    correct_predictions = (results_df['predicted'] == results_df['true']).sum()
    precision = correct_predictions / len(true_categories) if len(true_categories) > 0 else 0
    return precision

true_categories = file_list_df['Category'].tolist()
precision = calculate_precision(df_semantic['WUP'], true_categories) * 100

print(f"Précision - SIMILARITES SEMANTIQUE")
print(f"La précision des classifications est: {precision:.2f} %")

# ------------------------------------------------------------------------------------------------------------------
print(f"Précision - SIMILARITES SYNTAXIQUE")

for algo in algorithms.keys():
    precision = calculate_precision(df_syntaxic[algo], true_categories) * 100
    print(f"La précision : {algo} est: {precision:.2f} %")

# 
