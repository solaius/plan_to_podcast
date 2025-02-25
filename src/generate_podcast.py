import json
import requests

SYSTEM_PROMPT = """
You are a helpful assistant. The user will provide you with a topic to write a podcast about. You should write an informative podcast (a la NPR) based on the topic. The podcast should cover all the topics and key points the user requests.

The podcast has two hosts, {host_a} and {host_b}. {host_a} is a intelligent, informative host who is always excited to talk about the topic. {host_b} is a more skeptical host, asking questions to {host_a} about the topic for her to answer and adding his own thoughts to her response. Together the hosts do an excellent job of breaking down the topic and hit all the key points the user requests.

Your response should be a series of conversation turns, where each turn starts with the speaker's name in the format "<|speaker_name|>: " followed by their dialogue. Each turn should be separated by a blank line.

Example format:
<|{host_a}|>: [First host's dialogue]

<|{host_b}|>: [Second host's dialogue]

<|{host_a}|>: [First host's response]
"""


def generate_podcast_script(prompt: str, model: str, host_a: str, host_b: str) -> str:
    """Generate a podcast script from a given prompt."""
    # Construct the messages
    system_prompt = SYSTEM_PROMPT.format(host_a=host_a.title(), host_b=host_b.title())
    
    # Make request to Ollama API
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": f"{system_prompt}\n\nTopic: {prompt}",
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
            }
        }
    )
    
    if response.status_code != 200:
        return "Error: Failed to generate podcast script. Please try again."
    
    # Extract the generated text
    return response.json()["response"]


if __name__ == "__main__":
    prompt = "Red Hat Enterprise Linux (RHEL)"
    model = "qwen2.5:32b"
    response = generate_podcast_script(prompt=prompt, model=model, host_a="Hank", host_b="John")
    print(response)
