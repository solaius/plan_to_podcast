import requests


def get_models() -> list[str]:
    """Get a list of available models."""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = sorted([model["name"] for model in response.json()["models"]])
            return models
        return ["qwen2.5:32b"]  # Default model if can't connect
    except:
        return ["qwen2.5:32b"]  # Default model if can't connect
