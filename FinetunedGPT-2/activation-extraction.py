import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
import numpy as np
from embeddingManager import EmbeddingManager

class GPT2TokenEmbeddingExtractor:
    def __init__(self, model_name: str = "Arjun-G-Ravi/chat-GPT2"):
        """
        Initialize the token-level embedding extractor with a GPT2 model.
        
        Args:
            model_name (str): Name or path of the pre-trained model
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
        self.model.eval()
    
    def get_token_embeddings(self, text: str) -> Tuple[Dict[str, np.ndarray], List[str]]:
        """
        Extract embeddings for each token in the given text.
        
        Args:
            text (str): Input text to extract embeddings from
            
        Returns:
            Tuple[Dict[str, np.ndarray], List[str]]: 
                - Dictionary mapping tokens to their embeddings
                - List of tokens in sequence order
        """
        # Tokenize the input text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=200
        )
        
        # Get the actual tokens for reference
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        
        # Forward pass with output_hidden_states=True
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Get the hidden states from the last layer
        last_hidden_state = outputs.hidden_states[-1]
        
        # Convert to numpy and remove batch dimension
        token_embeddings = last_hidden_state.squeeze(0).numpy()
        
        # Create dictionary mapping tokens to their embeddings
        token_embedding_dict = {
            token: embedding
            for token, embedding in zip(tokens, token_embeddings)
        }
        
        return token_embedding_dict, tokens
    
    def extract_and_store_token_embeddings(
        self,
        texts: List[str],
        embedding_manager: EmbeddingManager
    ) -> None:
        """
        Extract and store embeddings for all tokens in the given texts.
        
        Args:
            texts (List[str]): List of input texts
            embedding_manager (EmbeddingManager): Instance of EmbeddingManager
        """
        for text in texts:
            token_embeddings, _ = self.get_token_embeddings(text)
            
            # Store embedding for each unique token
            for token, embedding in token_embeddings.items():
                embedding_manager.add_embedding(token, embedding)

# Example usage
if __name__ == "__main__":
    # Initialize the extractor and embedding manager
    extractor = GPT2TokenEmbeddingExtractor()
    manager = EmbeddingManager("GPT2FTEmbeddings/gpt2_token_embeddings.pkl")
    
    # Example texts
    texts = [
        "This is a sample text",
        "This is sparta",
    ]
    
    # Extract and store token embeddings
    extractor.extract_and_store_token_embeddings(texts, manager)
    
    # Example of retrieving embeddings for specific tokens
    vocabulary = manager.get_vocabulary()
    print(f"Total unique tokens: {len(vocabulary)}")
    
    # Print some example tokens and their embedding counts
    for token in list(vocabulary)[:5]:
        count = manager.get_embedding_count(token)
        print(f"Token: {token}, Number of embeddings: {count}")
        
        # Get average embedding for this token
        avg_embedding = manager.get_average_embedding(token)
        if avg_embedding is not None:
            print(f"Average embedding shape: {avg_embedding.shape}\n")