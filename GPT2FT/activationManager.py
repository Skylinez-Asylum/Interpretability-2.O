import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional

class ActivationManager:
    def __init__(self, storage_path: str = 'activations/activations.pkl'):
        self.storage_path = Path(storage_path)
        self.activations: Dict[str, List[np.ndarray]] = self._load_activations()
    
    def _load_activations(self) -> Dict[str, List[np.ndarray]]:
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'rb') as f:
                    loaded_data = pickle.load(f)
                    # Ensure each category has a list of activations
                    return {category: list(acts) if not isinstance(acts, list) else acts 
                           for category, acts in loaded_data.items()}
            except Exception as e:
                print(f"Error loading activations: {e}")
                return {}
        return {}
    
    def _save_activations(self) -> None:
        with open(self.storage_path, 'wb') as f:
            pickle.dump(self.activations, f)
    
    def add_activation(
        self,
        activation: np.ndarray,
        category: str
    ) -> None:
        # Initialize the category with an empty list if it doesn't exist
        if category not in self.activations:
            self.activations[category] = []
        
        # Now we can safely append to the list
        self.activations[category].append(activation)
        # self._save_activations()
    
    def add_activations_batch(
        self,
        activations: List[np.ndarray],
        category: str
    ) -> None:
        # Initialize the category with an empty list if it doesn't exist
        if category not in self.activations:
            self.activations[category] = []
        
        # Extend the list with the batch of activations
        self.activations[category].extend(activations)
        # self._save_activations()
    
    def get_activations(self, category: str) -> Optional[List[np.ndarray]]:
        return self.activations.get(category, [])
    
    def get_stats(self, show_vocabulary: bool = False) -> Dict[str, int]:
        total_count = 0
        activation_counts = {}
        
        for category, acts in self.activations.items():
            count = len(acts)
            activation_counts[category] = count
            total_count += count
        
        if show_vocabulary:
            return {"total": total_count, "Vocabulary": activation_counts}
        return {"total": total_count}
    
    def clear_activations(self, category: Optional[str] = None) -> None:
        if category:
            self.activations.pop(category, None)
        else:
            self.activations.clear()
        self._save_activations()

if __name__ == "__main__":
    # Example usage
    manager = ActivationManager("activations/GPT2FT/activations.pkl")
    
    # Add single activation for "dog" category
    activation = np.random.rand(756)
    manager.add_activation(activation, category="dog")
    
    # Add batch of activations for "cat" category
    batch_activations = [np.random.rand(756) for _ in range(10)]
    manager.add_activations_batch(batch_activations, category="cat")
    
    # Retrieve and display stats
    stats = manager.get_stats(show_vocabulary=True)
    print(f"Stats: {stats}")

    dog_activations = manager.get_activations('is')
    print('Dog shape:', dog_activations[0].shape)