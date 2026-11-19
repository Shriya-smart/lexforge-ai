"""Cache the exact pretrained GPT-2 files required for local inference."""

from huggingface_hub import snapshot_download

from lexforge.config import BASE_MODEL, BASE_REVISION

if __name__ == "__main__":
    print(snapshot_download(BASE_MODEL, revision=BASE_REVISION,
                            allow_patterns=["config.json", "model.safetensors",
                                            "tokenizer.json", "tokenizer_config.json"]))
