from flask import Flask, render_template, request
import re
from collections import defaultdict
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)

def preprocess_document(document):
    # Convert to lowercase
    document = document.lower()
    # Remove punctuation and non-alphanumeric characters
    document = re.sub(r"[^\w\s]", "", document)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    document = " ".join(word for word in document.split() if word not in stop_words)
    # Perform word stemming using NLTK's PorterStemmer
    stemmer = PorterStemmer()
    document = " ".join(stemmer.stem(word) for word in document.split())

    return document

def build_index(documents):
    index = defaultdict(list)
    for doc_id, document in enumerate(documents):
        preprocessed_doc = preprocess_document(document)
        words = preprocessed_doc.split()
        for position, word in enumerate(words):
            index[word].append((doc_id, position))
    return index

def search(query, index, documents):
    preprocessed_query = preprocess_document(query)
    query_words = preprocessed_query.split()
    search_results = defaultdict(list)

    for query_word in query_words:
        if query_word in index:
            postings = index[query_word]
            for doc_id, position in postings:
                search_results[doc_id].append(position)

    # Update to store and retrieve matching lines
    for doc_id, positions in search_results.items():
        document = documents[doc_id]
        lines = document.split("\n")
        matching_lines = [lines[position] for position in positions if position < len(lines)]
        search_results[doc_id] = matching_lines

    return search_results






@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        documents = request.form.get('documents')
        query = request.form.get('query')

        # Preprocess the user-provided documents
        preprocessed_documents = [preprocess_document(doc) for doc in documents.split('\n')]

        # Build the index
        index = build_index(preprocessed_documents)

        # Perform the search
        search_results = search(query, index, preprocessed_documents)

        return render_template('results.html', results=search_results, documents=preprocessed_documents)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run()
