import os
import jsonlines
import random
import argparse

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rank_bm25 import BM25Okapi

argparser = argparse.ArgumentParser()
# Parameters for context collection strategy
argparser.add_argument("--stage", type=str, default="practice", help="Stage of the project")
argparser.add_argument("--lang", type=str, default="python", help="Language")
argparser.add_argument("--strategy", type=str, default="random", help="Context collection strategy")

# Parameters for context trimming
argparser.add_argument("--trim-prefix", action="store_true", help="Trim the prefix to 10 lines")
argparser.add_argument("--trim-suffix", action="store_true", help="Trim the suffix to 10 lines")

args = argparser.parse_args()

stage = args.stage
language = args.lang
strategy = args.strategy

if language == "python":
    extension = ".py"
elif language == "kotlin":
    extension = ".kt"
else:
    raise ValueError(f"Unsupported language: {language}")

print(f"Running the {strategy} baseline for stage '{stage}'")

# token used to separate different files in the context
FILE_SEP_SYMBOL = "<|file_sep|>"
# format to compose context from a file
FILE_COMPOSE_FORMAT = "{file_sep}{file_name}\n{file_content}"


def find_random_file(root_dir: str, min_lines: int = 10) -> str:
    """
    Select a random file:
        - in the given language
        - in the given directory and its subdirectories
        - meeting length requirements

    :param root_dir: Directory to search for files with given extension.
    :param min_lines: Minimum number of lines required in the file.
    :return: Selected random file or None if no files were found.
    """
    code_files = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(extension):
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        if len(lines) >= min_lines:
                            code_files.append(file_path)
                except Exception as e:
                    # Optional: handle unreadable files
                    # print(f"Could not read {file_path}: {e}")
                    pass

    return random.choice(code_files) if code_files else None


def find_bm25_file(root_dir: str, prefix: str, suffix: str, min_lines: int = 10) -> str:
    """
    Select the file:
        - in the given language
        - with the highest BM25 score with the completion file
        - in the given directory and its subdirectories
        - meeting length requirements

    :param root_dir: Directory to search for files.
    :param prefix: Prefix of the completion file.
    :param suffix: Suffix of the completion file.
    :param min_lines: Minimum number of lines required in the file.
    :return:
    """

    def prepare_bm25_str(s: str) -> list[str]:
        return "".join(c if c.isalnum() else " " for c in s.lower()).split()

    corpus = []
    file_names = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(extension):
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        if len(lines) >= min_lines:
                            content = "\n".join(lines)
                            content = prepare_bm25_str(content)
                            corpus.append(content)
                            file_names.append(file_path)
                except Exception as e:
                    # Optional: handle unreadable files
                    # print(f"Could not read {file_path}: {e}")
                    pass

    query = (prefix + " " + suffix).lower()
    query = prepare_bm25_str(query)

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query)
    best_idx = scores.argmax()

    return file_names[best_idx] if file_names else None


def find_random_recent_file(root_dir: str, recent_filenames: list[str], min_lines: int = 10) -> str:
    """
    Select the most recent file:
        - in the given language
        - in the given directory and its subdirectories
        - meeting length requirements

    :param root_dir: Directory to search for files.
    :param recent_filenames: List of recent files filenames.
    :param min_lines: Minimum number of lines required in the file.
    :return: Selected random file or None if no files were found.
    """
    code_files = []
    for filename in recent_filenames:
        if filename.endswith(extension):
            file_path = os.path.join(root_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if len(lines) >= min_lines:
                        code_files.append(file_path)
            except Exception as e:
                # Optional: handle unreadable files
                # print(f"Could not read {file_path}: {e}")
                pass
    return random.choice(code_files) if code_files else None


def trim_prefix(prefix: str):
    prefix_lines = prefix.split("\n")
    if len(prefix_lines) > 10:
        prefix = "\n".join(prefix_lines[-10:])
    return prefix

def trim_suffix(suffix: str):
    suffix_lines = suffix.split("\n")
    if len(suffix_lines) > 10:
        suffix = "\n".join(suffix_lines[:10])
    return suffix


class VectorContextManager:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Inicjalizacja modelu do embeddingów - wybierz model zoptymalizowany pod kod
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.root_directory = None

    def index_repository(self, root_dir, extension=".py"):
        """Skanuje repozytorium, dzieli pliki na fragmenty i buduje indeks FAISS."""
        self.root_directory = root_dir
        
        all_texts = []
        self.chunks = []
        
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith(extension):
                    path = os.path.join(dirpath, filename)
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        # Prosty chunking: dzieli plik na funkcje lub stałe bloki
                        # Warto tu użyć biblioteki 'ast' dla lepszej precyzji w Pythonie
                        chunks = [c for c in content.split("\n\n") if len(c.strip()) > 20]
                        for chunk in chunks:
                            self.chunks.append({"file": path, "content": chunk})
                            all_texts.append(chunk)

        if not all_texts:
            return

        # Generowanie wektorów i budowa indeksu
        embeddings = self.model.encode(all_texts)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))

    def get_context(self, query, top_k=5):
        """Wyszukuje najbardziej podobne fragmenty kodu."""
        if self.index is None:
            return ""

        query_vector = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), top_k)

        context_parts = []

        for idx in indices[0]:
            if idx == -1: continue
            chunk = self.chunks[idx]
            rel_path = chunk["file"] # Ścieżka relatywna do repozytorium
            clean_file_name = rel_path[len(self.root_directory) + 1:]
            
            # Formatowanie
            formatted_chunk = FILE_COMPOSE_FORMAT.format(
                file_sep=FILE_SEP_SYMBOL, 
                file_name=clean_file_name, 
                file_content=chunk["content"]
            )
            context_parts.append(formatted_chunk)
            
        return "".join(context_parts)


def vector_strategy(datapoint, root_directory):
    manager = VectorContextManager()
    manager.index_repository(root_directory)
    
    # Query to połączenie prefixu i suffixu dla lepszego kontekstu
    query = f"{datapoint['prefix']} {datapoint['suffix']}"
    return manager.get_context(query, top_k=3)


# Path to the file with completion points
completion_points_file = os.path.join("data", f"{language}-{stage}.jsonl")

# Path to the file to store predictions
prediction_file_name = f"{language}-{stage}-{strategy}"
if args.trim_prefix:
    prediction_file_name += "-short-prefix"
if args.trim_suffix:
    prediction_file_name += "-short-suffix"
predictions_file = os.path.join("predictions", f"{prediction_file_name}.jsonl")

current_repo = None
manager = VectorContextManager()

with jsonlines.open(completion_points_file, 'r') as reader:
    with jsonlines.open(predictions_file, 'w') as writer:
        for datapoint in reader:
            # Identify the repository storage for the datapoint
            repo_path = datapoint['repo'].replace("/", "__")
            repo_revision = datapoint['revision']
            root_directory = os.path.join("data", f"repositories-{language}-{stage}", f"{repo_path}-{repo_revision}")

            if repo_path != current_repo:
                manager.index_repository(root_directory)
                current_repo = repo_path

            # Run the baseline strategy
            if strategy == "vector":
                query = f"{datapoint['prefix']} {datapoint['suffix']}"
                # This returns the full formatted string with <|file_sep|> tokens
                context = manager.get_context(query, top_k=5) 
            else:
                if strategy == "random":
                    file_name = find_random_file(root_directory)
                elif strategy == "bm25":
                    file_name = find_bm25_file(root_directory, datapoint['prefix'], datapoint['suffix'])
                elif strategy == "recent":
                    recent_filenames = datapoint['modified']
                    file_name = find_random_recent_file(root_directory, recent_filenames)
                    # If no recent files match our filtering criteria, select a random file instead
                    if file_name is None:
                        file_name = find_random_file(root_directory)
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")

                # Compose the context from the selected file
                file_content = open(file_name, 'r', encoding='utf-8').read()
                clean_file_name = file_name[len(root_directory) + 1:]
                context = FILE_COMPOSE_FORMAT.format(file_sep=FILE_SEP_SYMBOL, file_name=clean_file_name,
                                                    file_content=file_content)

            submission = {"context": context}
            # Write the result to the prediction file
            if args.trim_prefix:
                submission["prefix"] = trim_prefix(datapoint["prefix"])
            if args.trim_suffix:
                submission["suffix"] = trim_suffix(datapoint["suffix"])
            writer.write(submission)
