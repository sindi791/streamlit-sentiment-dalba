
import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Hardcoded stopwords (tanpa NLTK)
stop_words = set([
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'untuk', 'dengan', 'pada', 'karena',
    'juga', 'akan', 'itu', 'sudah', 'saja', 'lagi', 'kalau', 'bisa', 'kami', 'kita',
    'mereka', 'ada', 'tidak', 'ya', 'kok', 'gak', 'ga', 'jadi', 'terus', 'dapat', 'lah',
    'apa', 'kenapa', 'nah', 'oh', 'iya', 'terima', 'kasih', 'thanks', 'thank'
])

# Stemmer
stemmer = StemmerFactory().create_stemmer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|@\w+|#\w+|\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()  # Gantikan word_tokenize
    filtered = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(filtered)
