import numpy as np
from collections import defaultdict
from typing import List, Tuple, Callable
import asyncio

try:
    import faiss
except ImportError:
    raise ImportError(
        "faiss not found. Install via `pip install faiss-cpu` or `pip install faiss-gpu`"
    )

from aimakerspace.openai_utils.embedding import EmbeddingModel


def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)


class VectorDatabase:
    def __init__(self, embedding_model: EmbeddingModel = None):
        self.vectors = defaultdict(np.array)
        self.embedding_model = embedding_model or EmbeddingModel()

        # FAISS-related attributes
        self.index = None             # The FAISS index for ANN
        self._id_to_key = []          # Maps integer IDs in FAISS to keys in self.vectors
        self._key_to_id = {}          # Maps keys to integer IDs

    def insert(self, key: str, vector: np.array) -> None:
        self.vectors[key] = vector

    def search(
        self,
        query_vector: np.array,
        k: int,
        distance_measure: Callable = cosine_similarity,
    ) -> List[Tuple[str, float]]:
        """Brute-force search using a distance measure (e.g., cosine similarity)."""
        scores = [
            (key, distance_measure(query_vector, vector))
            for key, vector in self.vectors.items()
        ]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]

    def search_by_text(
        self,
        query_text: str,
        k: int,
        distance_measure: Callable = cosine_similarity,
        return_as_text: bool = False,
    ) -> List[Tuple[str, float]]:
        """Brute-force search by text input using a distance measure (e.g., cosine similarity)."""
        query_vector = self.embedding_model.get_embedding(query_text)
        results = self.search(query_vector, k, distance_measure)
        return [result[0] for result in results] if return_as_text else results

    def retrieve_from_key(self, key: str) -> np.array:
        return self.vectors.get(key, None)

    async def abuild_from_list(self, list_of_text: List[str]) -> "VectorDatabase":
        """
        Asynchronously build the vector database from a list of text.
        This populates `self.vectors` with text keys and their embeddings.
        """
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        for text, embedding in zip(list_of_text, embeddings):
            self.insert(text, np.array(embedding))
        return self

    # ---------------------- FAISS Approximate Nearest Neighbors ----------------------

    def build_approx_index(self, use_cosine: bool = False) -> None:
        """
        Build a FAISS index for Approximate Nearest Neighbors (ANN).
        
        Args:
            use_cosine (bool): If True, normalize vectors before building 
                               an inner product index to approximate cosine similarity.
        """
        # Ensure we have some vectors to index
        if not self.vectors:
            raise ValueError("No vectors to build an index from. Insert vectors first.")

        # Extract vectors and keys
        all_keys = list(self.vectors.keys())
        all_vectors = [self.vectors[key] for key in all_keys]
        all_vectors = np.array(all_vectors).astype(np.float32)

        d = all_vectors.shape[1]  # Dimension of the vectors

        if use_cosine:
            # Normalize vectors for approximate cosine similarity
            faiss.normalize_L2(all_vectors)
            # We use an IndexFlatIP (Inner Product) for approximate cosine
            self.index = faiss.IndexFlatIP(d)
        else:
            # Use L2 distance
            self.index = faiss.IndexFlatL2(d)

        # Add vectors to the index
        self.index.add(all_vectors)

        # Keep track of the ID/key mappings
        self._id_to_key = all_keys
        self._key_to_id = {key: i for i, key in enumerate(all_keys)}

    def approx_search(
        self,
        query_vector: np.array,
        k: int,
        use_cosine: bool = False,
    ) -> List[Tuple[str, float]]:
        """
        Perform an approximate nearest neighbors search using the FAISS index.

        Args:
            query_vector (np.array): The query vector.
            k (int): Number of nearest neighbors to retrieve.
            use_cosine (bool): If True, normalizes the query vector before searching 
                               with an inner product index to approximate cosine similarity.

        Returns:
            List[Tuple[str, float]]: A list of (key, distance) pairs. 
                                     Distance is either L2 or inner product.
        """
        if self.index is None:
            raise ValueError("FAISS index not built. Call `build_approx_index` first.")

        # Make a copy so we don't change the original
        query = query_vector.astype(np.float32).reshape(1, -1)

        if use_cosine:
            # Normalize the query vector for approximate cosine
            faiss.normalize_L2(query)

        # Search the index
        distances, indices = self.index.search(query, k)

        # distances shape: (1, k), indices shape: (1, k)
        result = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                # FAISS returns -1 for empty results if fewer than k found
                continue

            # For cosine similarity approximation with an inner-product index:
            #   - 'dist' is the approximate inner product (closer to 1 is more similar).
            # For L2 index:
            #   - 'dist' is the L2 distance (smaller is more similar).
            key = self._id_to_key[idx]
            result.append((key, float(dist)))

        return result

    def approx_search_by_text(
        self,
        query_text: str,
        k: int,
        use_cosine: bool = False,
        return_as_text: bool = False,
    ) -> List[Tuple[str, float]]:
        """
        Approximate nearest neighbors search by text input.

        Args:
            query_text (str): The query text.
            k (int): Number of nearest neighbors to retrieve.
            use_cosine (bool): If True, normalizes the query vector before 
                               searching with an inner product index for approx cosine.
            return_as_text (bool): If True, only returns the keys (text). Otherwise, 
                                   returns (key, distance) pairs.

        Returns:
            List[Tuple[str, float]] or List[str]: The approximate nearest neighbors.
        """
        query_vector = self.embedding_model.get_embedding(query_text)
        results = self.approx_search(np.array(query_vector), k, use_cosine)
        return [r[0] for r in results] if return_as_text else results

