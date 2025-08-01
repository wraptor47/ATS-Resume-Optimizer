import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import ast
import re
import spacy
from collections import Counter
import nltk
from nltk.corpus import wordnet
import string
import pdfplumber

# Download required NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def flatten_list(nested):
    flat = []
    if isinstance(nested, list):
        for item in nested:
            flat.extend(flatten_list(item))
    elif isinstance(nested, str):
        flat.append(nested)
    return flat

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using pdfplumber"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def extract_skills_from_text(text, known_skills=None):
    """Extract skills from text using regex and spaCy"""
    text = text.lower().strip()

    skill_patterns = [
        r'\b(?:sql|python|java|javascript|excel|power\s*bi|dax|microstrategy|postgresql|'
        r'database\s*(?:design|management|optimization)|data\s*(?:analysis|analytics|cleaning|modeling|warehousing|'
        r'engineering|visualization)|etl\s*processes|hadoop|mapreduce|spark|hive|linux|'
        r'ms\s*(?:word|excel|powerpoint)|problem\s*solving|communication|collaboration|'
        r'advanced\s*excel|power\s*(?:query|pivot)|vlookup|pivot\s*tables|macros|statistical\s*analysis)\b'
    ]

    regex_skills = set()
    for pattern in skill_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        regex_skills.update(matches)

    doc = nlp(text)
    spacy_skills = set()
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"] and token.text.lower() in text:
            spacy_skills.add(token.text.lower())

    combined_skills = regex_skills | spacy_skills
    if known_skills:
        combined_skills = {skill for skill in combined_skills if any(ks.lower() in skill.lower() for ks in known_skills)}

    combined_skills = [skill.strip().replace('\s+', ' ') for skill in combined_skills if skill.strip()]
    return list(set(combined_skills))

class ATSOptimizer:
    def __init__(self):
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.skill_embeddings = None
        self.skill_clusters = None
        self.cluster_labels = None
        self.resume_vectors = None
        self.unique_skills = None

    def normalize_skill(self, skill):
        skill = skill.lower().translate(str.maketrans('', '', string.punctuation))

        synonym_map = {
            'machine learning': ['ml', 'machinelearning'],
            'deep learning': ['deeplearning', 'dl'],
            'data analysis': ['data analytics', 'dataanalysis'],
            'business intelligence': ['bi'],
            'javascript': ['js'],
            'python': ['py'],
            'sql': ['structured query language'],
            'natural language processing': ['nlp'],
            'excel': ['advanced excel', 'advance excel', 'ms excel'],
            'power bi': ['powerbi'],
            'java': ['java core', 'core java']
        }

        for standard, synonyms in synonym_map.items():
            if skill in synonyms:
                return standard
        return skill

    def prepare_data(self, resume_data):
        cols_to_clean = ['skills', 'related_skills_in_job']

        for col in cols_to_clean:
            if col in resume_data.columns:
                resume_data[col] = resume_data[col].apply(
                    lambda x: ', '.join(flatten_list(ast.literal_eval(x) if isinstance(x, str) else x))
                    if isinstance(x, (list, str)) and x
                    else str(x).replace('[','').replace(']','').replace("'", '')
                )
            else:
                print(f"⚠️ Column '{col}' not found. Skipping.")

        skills_col = 'skills' if 'skills' in resume_data.columns else 'resume_skills' if 'resume_skills' in resume_data.columns else None
        related_skills_col = 'related_skills_in_job' if 'related_skills_in_job' in resume_data.columns else 'related_skills' if 'related_skills' in resume_data.columns else None

        if not skills_col and not related_skills_col:
            raise ValueError("No skills columns found in dataset")

        all_skills = set()
        for _, row in resume_data.iterrows():
            if skills_col and pd.notna(row[skills_col]):
                skills = row[skills_col].split(', ') if row[skills_col] else []
                all_skills.update(skills)
            if related_skills_col and pd.notna(row[related_skills_col]):
                related_skills = row[related_skills_col].split(', ') if row[related_skills_col] else []
                all_skills.update(related_skills)

        self.unique_skills = [self.normalize_skill(skill) for skill in all_skills if skill]
        self.unique_skills = list(set(self.unique_skills))

        self.skill_embeddings = self.sbert_model.encode(self.unique_skills)

        return self.unique_skills

    def cluster_skills(self, n_clusters=10):
        clustering = KMeans(n_clusters=n_clusters, random_state=42)
        self.skill_clusters = clustering.fit_predict(self.skill_embeddings)

        cluster_skills = {i: [] for i in range(n_clusters)}
        for skill, cluster in zip(self.unique_skills, self.skill_clusters):
            cluster_skills[cluster].append(skill)

        self.cluster_labels = {}
        for cluster_id, skills in cluster_skills.items():
            words = ' '.join(skills).split()
            common_words = Counter(words).most_common(3)
            label = ' '.join(word for word, _ in common_words)
            self.cluster_labels[cluster_id] = label if label else f"Cluster {cluster_id}"

        return self.cluster_labels

    def create_resume_vectors(self, resume_data):
        skills_col = 'skills' if 'skills' in resume_data.columns else 'resume_skills' if 'resume_skills' in resume_data.columns else None
        related_skills_col = 'related_skills_in_job' if 'related_skills_in_job' in resume_data.columns else 'related_skills' if 'related_skills' in resume_data.columns else None

        self.resume_vectors = []

        for _, row in resume_data.iterrows():
            resume_skills = set()
            if skills_col and pd.notna(row[skills_col]):
                resume_skills.update(row[skills_col].split(', '))
            if related_skills_col and pd.notna(row[related_skills_col]):
                resume_skills.update(row[related_skills_col].split(', '))

            resume_skills = [self.normalize_skill(skill) for skill in resume_skills if skill]

            valid_skills = [skill for skill in resume_skills if skill in self.unique_skills]
            if not valid_skills:
                self.resume_vectors.append(np.zeros(self.skill_embeddings.shape[1]))
                continue

            skill_indices = [self.unique_skills.index(skill) for skill in valid_skills]
            resume_vector = np.mean(self.skill_embeddings[skill_indices], axis=0)
            self.resume_vectors.append(resume_vector)

        self.resume_vectors = np.array(self.resume_vectors)
        return self.resume_vectors

    def assign_resume_to_cluster(self):
        resume_clusters = []
        for resume_vector in self.resume_vectors:
            similarities = []
            for cluster_id in set(self.skill_clusters):
                cluster_skills = [i for i, c in enumerate(self.skill_clusters) if c == cluster_id]
                if cluster_skills:
                    cluster_centroid = np.mean(self.skill_embeddings[cluster_skills], axis=0)
                    similarity = cosine_similarity([resume_vector], [cluster_centroid])[0][0]
                    similarities.append((cluster_id, similarity))
                else:
                    similarities.append((cluster_id, 0.0))

            best_cluster = max(similarities, key=lambda x: x[1])[0]
            resume_clusters.append(best_cluster)

        return resume_clusters

    def match_job_description(self, job_skills, job_domain=None):
        job_skills = [self.normalize_skill(skill) for skill in job_skills if skill]

        valid_job_skills = [skill for skill in job_skills if skill in self.unique_skills]
        if not valid_job_skills:
            return [], [], "No valid skills found in job description"

        skill_indices = [self.unique_skills.index(skill) for skill in valid_job_skills]
        job_vector = np.mean(self.skill_embeddings[skill_indices], axis=0)

        similarities = cosine_similarity([job_vector], self.resume_vectors)[0]

        resume_clusters = self.assign_resume_to_cluster()

        results = []
        for i, (similarity, cluster_id) in enumerate(zip(similarities, resume_clusters)):
            results.append({
                'resume_index': i,
                'confidence_score': float(similarity),
                'predicted_domain': self.cluster_labels[cluster_id]
            })

        results = sorted(results, key=lambda x: x['confidence_score'], reverse=True)

        return results, valid_job_skills, "Success"

    def process_single_resume(self, resume_skills, job_skills):
        resume_skills = [self.normalize_skill(skill) for skill in resume_skills if skill]
        valid_skills = [skill for skill in resume_skills if skill in self.unique_skills]

        if not valid_skills:
            return {"domain": "Unknown", "confidence_score": 0.0, "message": "No valid skills found in resume"}

        skill_indices = [self.unique_skills.index(skill) for skill in valid_skills]
        resume_vector = np.mean(self.skill_embeddings[skill_indices], axis=0)

        similarities = []
        for cluster_id in set(self.skill_clusters):
            cluster_skills = [i for i, c in enumerate(self.skill_clusters) if c == cluster_id]
            if cluster_skills:
                cluster_centroid = np.mean(self.skill_embeddings[cluster_skills], axis=0)
                similarity = cosine_similarity([resume_vector], [cluster_centroid])[0][0]
                similarities.append((cluster_id, similarity))
            else:
                similarities.append((cluster_id, 0.0))

        best_cluster_id = max(similarities, key=lambda x: x[1])[0]
        predicted_domain = self.cluster_labels[best_cluster_id]

        job_skills = [self.normalize_skill(skill) for skill in job_skills if skill]
        valid_job_skills = [skill for skill in job_skills if skill in self.unique_skills]

        if not valid_job_skills:
            return {"domain": predicted_domain, "confidence_score": 0.0, "message": "No valid skills found in job description"}

        job_vector = np.mean(self.skill_embeddings[[self.unique_skills.index(skill) for skill in valid_job_skills]], axis=0)
        confidence_score = float(cosine_similarity([job_vector], [resume_vector])[0][0])

        return {"domain": predicted_domain, "confidence_score": confidence_score, "message": "Success"}

# Example usage
if __name__ == "__main__":
    pass