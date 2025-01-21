import numpy as np
import json
import os
from datetime import datetime
import glob

class ActivationRetriever:
    def __init__(self, activations_dir="activations"):
        self.activations_dir = activations_dir
    
    def list_available_files(self):
        """List all available activation files with their metadata"""
        meta_files = glob.glob(os.path.join(self.activations_dir, "*_meta.json"))
        available_files = []
        
        for meta_file in meta_files:
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            available_files.append(metadata)
        
        return available_files
    
    def get_word_embeddings(self, word, file_timestamp=None):
        """
        Retrieve embeddings for a specific word.
        If file_timestamp is None, gets from all files.
        """
        results = []
        print('cow1')
        # Get list of files to process
        if file_timestamp:
            meta_files = [f"activations_{file_timestamp}_meta.json"]
        else:
            meta_files = glob.glob(os.path.join(self.activations_dir, "*_meta.json"))
        
        print('cow2')
        for meta_file in meta_files:
            # with open(os.path.join(self.activations_dir, meta_file), 'r') as f:
            #     metadata = json.load(f)

            with open('/home/arjun/Desktop/GitHub/Interpretability-2.O/activations/activations_20250121_193044_meta.json', 'r') as f:
                activation_file = json.load(f)[0]['activation_file']
            
            # activation_file = metadata["activation_file"]
            if not os.path.exists(activation_file):
                print(f"Warning: Activation file {activation_file} not found")
                continue
            
            # Load activations
            activations = np.load(activation_file)
            if word in activations:
                results.append({
                    "timestamp": metadata["timestamp"],
                    "prompt": metadata["prompt"],
                    "generated_text": metadata["generated_text"],
                    "embeddings": activations[word]
                })
        
        return results
    
    def get_average_embedding(self, word, file_timestamp=None):
        """Get the average embedding for a word across all instances"""
        embeddings_data = self.get_word_embeddings(word, file_timestamp)
        if not embeddings_data:
            return None
        
        all_embeddings = []
        for data in embeddings_data:
            all_embeddings.extend(data["embeddings"])
        
        return np.mean(all_embeddings, axis=0) if all_embeddings else None

if __name__ == "__main__":
    # Example usage
    retriever = ActivationRetriever()
    
    # List available files
    print("Available activation files:")
    for metadata in retriever.list_available_files():
        print(f"\nTimestamp: {metadata['timestamp']}")
        print(f"Prompt: {metadata['prompt'][:50]}...")
        print(f"Word counts: {metadata['word_counts']}")
    
    # Get embeddings for a specific word
    word = "RAM"
    embeddings_data = retriever.get_word_embeddings(word)
    print(embeddings_data)
    
    if embeddings_data:
        print(f"\nFound embeddings for '{word}':")
        for data in embeddings_data:
            print(f"\nTimestamp: {data['timestamp']}")
            print(f"Number of embeddings: {len(data['embeddings'])}")
            print(f"Embedding shape: {data['embeddings'][0].shape}")
    
    # Get average embedding
    avg_embedding = retriever.get_average_embedding(word)
    if avg_embedding is not None:
        print(f"\nAverage embedding shape for '{word}': {avg_embedding.shape}")