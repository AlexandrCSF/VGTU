import nltk
import re
import pandas as pd
from razdel import tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pymorphy2

nltk.download('stopwords')
nltk.download('punkt')

docs = [
    "Машинное обучение — это увлекательно.",
    "Глубокое обучение развивает машинное обучение.",
    "Обработка естественного языка — часть искусственного интеллекта."
]

morph = pymorphy2.MorphAnalyzer()
stop_words = set(stopwords.words('russian'))

def preprocess(text):
    text = text.lower()
    tokens = [t.text for t in tokenize(text)]
    tokens = [re.sub(r'[^а-яё]', '', t) for t in tokens]
    tokens = [t for t in tokens if t and t not in stop_words]
    tokens = [morph.parse(t)[0].normal_form for t in tokens if t]
    return " ".join(tokens)

clean_docs = [preprocess(d) for d in docs]

print("Исходные документы:")
for d in docs:
    print("-", d)

print("\nПосле предобработки:")
for d in clean_docs:
    print("-", d)

bow_vectorizer = CountVectorizer(token_pattern=r'[а-яё]+')
bow_matrix = bow_vectorizer.fit_transform(clean_docs)

print("\nФича-слова (BoW):")
print(bow_vectorizer.get_feature_names_out())

print("\nМатрица BoW (document-term):")
bow_df = pd.DataFrame(bow_matrix.toarray(), columns=bow_vectorizer.get_feature_names_out())
print(bow_df)

tfidf_vectorizer = TfidfVectorizer(token_pattern=r'[а-яё]+')
tfidf_matrix = tfidf_vectorizer.fit_transform(clean_docs)

print("\nФича-слова (TF-IDF):")
print(tfidf_vectorizer.get_feature_names_out())

print("\nМатрица TF-IDF (document-term):")
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
print(tfidf_df.round(3))

bow_sim = cosine_similarity(bow_matrix)
tfidf_sim = cosine_similarity(tfidf_matrix)

print("\nКосинусное сходство между документами (BoW):")
print(pd.DataFrame(bow_sim))

print("\nКосинусное сходство между документами (TF-IDF):")
print(pd.DataFrame(tfidf_sim.round(3)))
