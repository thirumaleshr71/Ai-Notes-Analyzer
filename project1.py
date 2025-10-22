import os, string, random
from typing import List, Tuple, Dict
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
import PyPDF2
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

HF_SUMMARIZER = None
STOPWORDS = set(stopwords.words("english"))
MODEL = SentenceTransformer("all-mpnet-base-v2")

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

def load_abstractive_model():
    global HF_SUMMARIZER
    if HF_SUMMARIZER is None:
        HF_SUMMARIZER = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1
        )

def chunk_text(text: str, max_tokens: int = 400) -> List[str]:
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for s in sentences:
        if len(current_chunk.split()) + len(s.split()) <= max_tokens:
            current_chunk += " " + s
        else:
            chunks.append(current_chunk.strip())
            current_chunk = s
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def abstractive_summarize(text: str) -> str:
    if HF_SUMMARIZER is None:
        load_abstractive_model()
    chunks = chunk_text(text)
    summaries = []
    for chunk in chunks:
        s = HF_SUMMARIZER(chunk, max_length=120, min_length=30, do_sample=False)[0]['summary_text']
        summaries.append(s)
    if len(summaries) > 1:
        merged_summary = " ".join(summaries)
        final_summary = HF_SUMMARIZER(merged_summary, max_length=180, min_length=40, do_sample=False)[0]['summary_text']
        return final_summary
    return summaries[0]

def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_pdf(path: str) -> str:
    text = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(" ".join(page_text.split()))
    return "\n".join(text)

def load_notes(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".md"):
        return load_txt(path)
    elif ext == ".pdf":
        return load_pdf(path)
    else:
        raise ValueError("Unsupported file type. Use .txt, .md, or .pdf")

def clean_tokens(text: str) -> List[str]:
    return [t for t in word_tokenize(text.lower()) if t.isalpha() and t not in STOPWORDS]

def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 3]

def summarize(text: str, num_sentences: int = 3) -> str:
    sentences = split_sentences(text)
    if len(sentences) <= num_sentences:
        return " ".join(sentences)
    word_freq = Counter(clean_tokens(text))
    max_freq = max(word_freq.values(), default=1)
    for w in word_freq: word_freq[w] /= max_freq
    sentence_scores = {s: sum(word_freq.get(w,0) for w in clean_tokens(s)) for s in sentences}
    top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    return " ".join(top_sentences[:num_sentences]) or "No significant content found."

def answer_semantic_topk(question: str, text: str, top_k: int = 3, threshold: float = 0.4) -> List[Tuple[str, float]]:
    sentences = split_sentences(text)
    if not sentences: return []
    sent_embeddings = MODEL.encode(sentences, convert_to_tensor=True, normalize_embeddings=True)
    q_embedding = MODEL.encode(question, convert_to_tensor=True, normalize_embeddings=True)
    similarities = util.cos_sim(q_embedding, sent_embeddings)[0]
    top_indices = torch.topk(similarities, k=min(top_k,len(sentences))).indices
    results = [(sentences[idx], float(similarities[idx])) for idx in top_indices if float(similarities[idx])>=threshold]
    return results

def answer_question(question: str, text: str, top_k: int=3, threshold: float=0.5) -> str:
    top_answers = answer_semantic_topk(question, text, top_k, threshold)
    if not top_answers: return "Sorry — I couldn't find a relevant answer in your notes."
    combined = " ".join(ans for ans,_ in top_answers)
    return summarize(combined, num_sentences=min(3,len(top_answers)))

def extract_terms(sentence: str, num_terms: int=2) -> List[str]:
    tagged = pos_tag(word_tokenize(sentence))
    return [word for word, tag in tagged if tag.startswith(("NN","VB","JJ")) and word.lower() not in STOPWORDS and word.isalpha()][:num_terms]

def create_blank(sentence: str, term: str) -> str:
    return sentence.replace(term, "_____", 1)

def generate_quiz(text: str, num_questions: int=5) -> List[Dict[str,str]]:
    sentences = [s for s in split_sentences(text) if len(s.split())>4]
    random.shuffle(sentences)
    quizzes, used_terms = [], set()
    for s in sentences:
        terms = extract_terms(s)
        if not terms: continue
        term = terms[0].lower()
        if term in used_terms or len(term)<=2: continue
        used_terms.add(term)
        quizzes.append({"question": create_blank(s,term),"answer":term})
        if len(quizzes)>=num_questions: break
    return quizzes
