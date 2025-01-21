import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Union, Optional

class EmbeddingManager:
    def __init__(self, storage_path: str = 'embeddings.pkl'):
        """
        Initialize the EmbeddingManager with a storage file path.
        
        Args:
            storage_path (str): Path to store the embeddings pickle file
        """
        self.storage_path = Path(storage_path)
        self.embeddings: Dict[str, List[np.ndarray]] = self._load_embeddings()
    
    def _load_embeddings(self) -> Dict[str, List[np.ndarray]]:
        """Load existing embeddings from file or create new dictionary."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading embeddings: {e}")
                return {}
        return {}
    
    def _save_embeddings(self) -> None:
        """Save embeddings to file."""
        with open(self.storage_path, 'wb') as f:
            pickle.dump(self.embeddings, f)
    
    def add_embedding(self, word: str, embedding: np.ndarray) -> None:
        """
        Add a new embedding for a word. If the word exists, append to its list.
        
        Args:
            word (str): The token/word
            embedding (np.ndarray): The embedding vector
        """
        if word not in self.embeddings:
            self.embeddings[word] = []
        self.embeddings[word].append(embedding)
        self._save_embeddings()
    
    def add_embeddings_batch(self, word: str, embeddings: List[np.ndarray]) -> None:
        """
        Add multiple embeddings for a word at once.
        
        Args:
            word (str): The token/word
            embeddings (List[np.ndarray]): List of embedding vectors
        """
        if word not in self.embeddings:
            self.embeddings[word] = []
        self.embeddings[word].extend(embeddings)
        self._save_embeddings()
    
    def get_embeddings(self, word: str) -> Optional[List[np.ndarray]]:
        """
        Retrieve all embeddings for a word.
        
        Args:
            word (str): The token/word
        
        Returns:
            Optional[List[np.ndarray]]: List of embeddings or None if word not found
        """
        return self.embeddings.get(word)
    
    def get_average_embedding(self, word: str) -> Optional[np.ndarray]:
        """
        Get the average embedding for a word.
        
        Args:
            word (str): The token/word
        
        Returns:
            Optional[np.ndarray]: Average embedding or None if word not found
        """
        embeddings = self.get_embeddings(word)
        if embeddings:
            return np.mean(embeddings, axis=0)
        return None
    
    def get_vocabulary(self) -> List[str]:
        """Get list of all stored words."""
        return list(self.embeddings.keys())
    
    def get_embedding_count(self, word: str) -> int:
        """Get number of stored embeddings for a word."""
        return len(self.embeddings.get(word, []))
    
    def clear_embeddings(self, word: Optional[str] = None) -> None:
        """
        Clear embeddings for a specific word or all embeddings.
        
        Args:
            word (Optional[str]): Word to clear. If None, clears all embeddings.
        """
        if word:
            self.embeddings.pop(word, None)
        else:
            self.embeddings.clear()
        self._save_embeddings()

# Example usage
if __name__ == "__main__":
    # Initialize manager
    manager = EmbeddingManager("activations/test_embeddings.pkl")

    
    # Add some test embeddings
    test_embedding = np.random.rand(768)  # Example dimension
    manager.add_embedding("test_word", test_embedding)
    
    # Add batch of embeddings
    batch_embeddings = [np.random.rand(768) for _ in range(3)]
    manager.add_embeddings_batch("test_word", batch_embeddings)
    
    # Retrieve embeddings
    embeddings = manager.get_embeddings("test_word")
    print(embeddings)
    print(f"Number of embeddings for 'test_word': {len(embeddings)}")
    
    # Get average embedding
    avg_embedding = manager.get_average_embedding("test_word")
    print(f"Average embedding shape: {avg_embedding.shape}")
    
    # Show vocabulary
    print(f"Stored words: {manager.get_vocabulary()}")