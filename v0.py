from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "llama3.2"

def chat(user_message: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
                {"role": "system", "content": "You are a helpful personal assistant."},
                {"role": "user", "content": user_message},
            ],
    )
    return response.choices[0].message.content or ""

if __name__ == "__main__":
    print(f"v0 assistant ({MODEL}) - ctrl+c to exit")
    try:
        while True:
            question = input("You: ")
            print("\nAssistant:", chat(question))
    except (EOFError, KeyboardInterrupt):
        print("\nbye!")