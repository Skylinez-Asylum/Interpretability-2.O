from huggingface_hub import HfApi, create_repo
import torch
import os
from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
from model import GPT, GPTConfig, load_model
from transformers import GPT2Tokenizer


class CustomGPTConfig(PretrainedConfig):
    model_type = "custom_gpt"
    
    def __init__(
        self,
        block_size=1024,
        vocab_size=50304,
        n_layer=12,
        n_head=6,
        n_embd=768,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.architectures = ["CustomGPTPreTrainedModel"]
        self.tokenizer_class = "GPT2Tokenizer"


class CustomGPTPreTrainedModel(PreTrainedModel, GenerationMixin):  # Fix: Add GenerationMixin
    config_class = CustomGPTConfig
    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__(config)
        self.model = GPT(
            GPTConfig(
                block_size=config.block_size,
                vocab_size=config.vocab_size,
                n_layer=config.n_layer,
                n_head=config.n_head,
                n_embd=config.n_embd,
            )
        )

    def forward(self, input_ids, labels=None):
        logits, loss = self.model(input_ids, labels)
        return {"logits": logits, "loss": loss}

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    def save_pretrained(self, save_directory):
        super().save_pretrained(save_directory, safe_serialization=False)  # Fix: Avoid tensor tying error
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.model.state_dict(), model_path)


def upload_model_to_hub(model_path: str, repo_name: str, token: str):
    print("Loading model...")
    custom_model = load_model(model_path)

    config = CustomGPTConfig(
        block_size=custom_model.config.block_size,
        vocab_size=custom_model.config.vocab_size,
        n_layer=custom_model.config.n_layer,
        n_head=custom_model.config.n_head,
        n_embd=custom_model.config.n_embd,
    )

    hf_model = CustomGPTPreTrainedModel(config)
    hf_model.model.load_state_dict(custom_model.state_dict())

    print(f"Creating repository: {repo_name}")
    api = HfApi()
    create_repo(repo_name, token=token, exist_ok=True)

    output_dir = "temp_model"
    os.makedirs(output_dir, exist_ok=True)
    hf_model.save_pretrained(output_dir)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.save_pretrained(output_dir)

    model_card = f"""
    # Custom GPT Model

    This is a custom GPT model with:
    - RMS normalization
    - Rotary positional embeddings (RoPE)
    - Separate Q,K,V projections
    - Squared ReLU activation in MLP
    - QK normalization in attention
    - Zero initialization for projection layers

    ## Architecture
    - Vocabulary Size: {config.vocab_size}
    - Context Length: {config.block_size}
    - Number of Layers: {config.n_layer}
    - Number of Heads: {config.n_head}
    - Embedding Dimension: {config.n_embd}

    ## Usage
    ```python
    from transformers import AutoModel
    model = AutoModel.from_pretrained("{repo_name}")
    ```
    """

    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(model_card)

    print("Uploading to Hugging Face Hub...")
    api.upload_folder(folder_path=output_dir, repo_id=repo_name, token=token)

    import shutil
    shutil.rmtree(output_dir)
    print(f"Model uploaded successfully to: https://huggingface.co/{repo_name}")


if __name__ == "__main__":
    MODEL_PATH = "customGPT-2/save_states/state_step555000.pt"
    REPO_NAME = "Arjun-G-Ravi/Custom-GPT-555k"
    TOKEN = ""  # Add your Hugging Face token here
    
    upload_model_to_hub(MODEL_PATH, REPO_NAME, TOKEN)
