import json
from pathlib import Path
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "llama3.2"

WORKSPACE = Path(__file__).parent / "workspace"


# ------------ the actual tool: plain Python function ------------

def list_file() -> str:
    """List all files in the workspace directory."""
    return "\n".join(p.name for p in WORKSPACE.iterdir()) or "(empty)"

def read_file(filename: str) -> str:
    """Read the content of a file in the workspace directory."""
    path = WORKSPACE / filename
    if not path.exists():
        return f"Error: File '{filename}' does not exist."
    return path.read_text(encoding="utf-8")

# 1. Define get_safe_path FIRST
def get_safe_path(filename: str) -> Path:

    if not filename or not filename.strip():
        raise ValueError("Filename cannot be empty.")

    path = (WORKSPACE / filename).resolve()
    workspace = WORKSPACE.resolve()

    if not path.is_relative_to(workspace):
        raise ValueError("Access outside workspace is not allowed.")

    return path

def write_file(filename: str, content: str) -> str:
    """Create or update a file inside the workspace directory."""

    try:
        path = get_safe_path(filename)

        if path.exists() and path.is_dir():
            return f"Error: '{filename}' is a directory."

        # Create parent directories if necessary
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write content into file
        path.write_text(content, encoding="utf-8")

        return f"Success: File '{filename}' created/updated successfully."

    except Exception as e:
        return f"Error writing file: {e}"


TOOLS = {
    "list_file": list_file,
    "read_file": read_file,
    "write_file": write_file
}

TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
                "name": "list_file",
                "description": "List the files in the user's workspace folder.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                },
            },
    },
    {
        "type": "function",
        "function": {
                "name": "read_file",
                "description": "Read one file from the user's workspace folder.",
                "parameters": {
                    "type": "object",
                    "properties": {"filename": {"type": "string", "description": "The name of the file to read."}}
                },
            },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create a new file or update an existing file in the workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the file, for example notes.txt or folder/test.py"
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete content to write into the file."
                    }
                },
                "required": ["filename", "content"],
                "additionalProperties": False
            }
        }
    }
    
]

def run_agent(user_message: str) -> str:
        messages = [
                        {"role": "system", "content": "You are a helpful personal assistant. "},
                        {"role": "user", "content": user_message},
                    ]
        # The agentic loop. This is the whole trick
        while True:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools = TOOL_SCHEMA,
            )
            message = response.choices[0].message
            
            # The model didn't want a tool -just answer.
            if not message.tool_calls:
                return message.content or ""
            
            # Run the tool it asked for, and add the result to the conversation.
            messages.append(message)
            for call in message.tool_calls:
                args = json.loads(call.function.arguments or "{}")
                print(f"Tool call: {call.function.name}({args})")
                result = TOOLS[call.function.name](**args)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
                
            # final_response = client.chat.completions.create(
            #     model=MODEL,
            #     messages=messages,
            # )
            # return final_response.choices[0].message.content or ""
    
if __name__ == "__main__":
    print(f"v1 assistant ({MODEL}) - ctrl+c to exit")
    try:
        while True:
            question = input("You: ")
            print("\nAssistant:", run_agent(question))
    except (EOFError, KeyboardInterrupt):
        print("\nbye!")
        