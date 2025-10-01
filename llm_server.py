from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama

app = FastAPI()

# Load your quantized LLaMA model
llm = Llama(
    model_path=r"D:\thesis\Meta-Llama-3.1-8B-Instruct-GGUF\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    n_gpu_layers=-1,
    n_ctx=4096,
    verbose=False
)

class Query(BaseModel):
    messages: list  # full chat-style history

@app.post("/chat")
def chat(query: Query):
    try:
        out = llm.create_chat_completion(
            messages=query.messages,
            max_tokens=150,
            temperature=0.7,
            top_p=0.9,
            stop=["<|eot_id|>", "User:"]
        )
        return {"reply": out["choices"][0]["message"]["content"].strip()}
    except Exception as e:
        import traceback
        print(f"[LLM Server Error] {e}")
        traceback.print_exc()
        return {"reply": "[LLM internal error]"}