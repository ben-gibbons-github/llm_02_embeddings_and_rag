import os
from typing import List
from sentence_transformers import SentenceTransformer, util
import numpy as np
import nltk

class TextFileLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.path = path
        self.encoding = encoding

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path) and self.path.endswith(".txt"):
            self.load_file()
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a .txt file."
            )

    def load_file(self):
        with open(self.path, "r", encoding=self.encoding) as f:
            self.documents.append(f.read())

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".txt"):
                    with open(
                        os.path.join(root, file), "r", encoding=self.encoding
                    ) as f:
                        self.documents.append(f.read())

    def load_documents(self):
        self.load()
        return self.documents


class CharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks

class SemanticTextSplitter:
    def __init__(
        self,
        similarity_threshold: float = 0.75,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_chunk_size: int = 1000,
    ):
        """
        :param similarity_threshold: The minimum average similarity 
                                     to keep sentences in the same chunk.
        :param model_name: SentenceTransformers model name.
        :param max_chunk_size: Maximum chunk size (in characters) 
                               before starting a new chunk (optional).
        """
        self.similarity_threshold = similarity_threshold
        self.model = SentenceTransformer(model_name)
        self.max_chunk_size = max_chunk_size

    def tokenize_text(self, text: str) -> List[str]:
        """
        Splits text into sentences or paragraphs. 
        For simplicity, we use sentence tokenization here.
        """
        # You could also split by paragraphs or any custom rules
        sentences = nltk.sent_tokenize(text)
        return sentences

    def embed_sentences(self, sentences: List[str]) -> np.ndarray:
        """
        Returns the embeddings for a list of sentences.
        """
        return self.model.encode(sentences, convert_to_tensor=True)

    def split(self, text: str) -> List[str]:
        """
        Splits a single text into semantically coherent chunks.
        """
        # 1. Sentence-level split
        sentences = self.tokenize_text(text)
        # 2. Embed each sentence
        sentence_embeddings = self.embed_sentences(sentences)

        chunks = []
        current_chunk = []
        current_chunk_embeddings = []

        current_chunk_size = 0  # track character length of chunk

        for idx, sentence in enumerate(sentences):
            sentence_embedding = sentence_embeddings[idx]

            if not current_chunk:
                # start the first chunk
                current_chunk.append(sentence)
                current_chunk_embeddings.append(sentence_embedding)
                current_chunk_size += len(sentence)
            else:
                # check similarity with the current chunk's centroid
                centroid = np.mean(current_chunk_embeddings, axis=0)
                similarity = float(util.cos_sim(centroid, sentence_embedding))

                new_size = current_chunk_size + len(sentence)

                # If similarity is high enough (and not exceeding max chunk size),
                # add to current chunk
                if similarity >= self.similarity_threshold and new_size <= self.max_chunk_size:
                    current_chunk.append(sentence)
                    current_chunk_embeddings.append(sentence_embedding)
                    current_chunk_size = new_size
                else:
                    # Otherwise, start a new chunk
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_chunk_embeddings = [sentence_embedding]
                    current_chunk_size = len(sentence)

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        """
        Splits multiple texts into semantically coherent chunks.
        """
        all_chunks = []
        for text in texts:
            all_chunks.extend(self.split(text))
        return all_chunks


if __name__ == "__main__":
    loader = TextFileLoader("data/KingLear.txt")
    loader.load()
    splitter = CharacterTextSplitter()
    chunks = splitter.split_texts(loader.documents)
    print(len(chunks))
    print(chunks[0])
    print("--------")
    print(chunks[1])
    print("--------")
    print(chunks[-2])
    print("--------")
    print(chunks[-1])
