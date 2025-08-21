# EECE5260

## Week 3 Hands on activity

### 🚀 Getting Started with Ollama
1. Install Ollama from [ollama.com](https://ollama.com)  
2. Select a LLM based on your hardware capacity:

| Purpose                   | Model size | Minimum Memory usage (Int4 quantization) | Recommended memory                  |
|----------------------------|------------|---------------------|-------------------------------------|
| For fun                   | 3B         | 1.9 GB              | 8 GB                                |
|                            | 7B         | 4.7 GB              |  8 GB                               |
| Assist daily work          | 14B        | 9.0 GB              | 12 GB, GPU recommended (RTX5070)    |
|                            | 32B        | 20 GB               | 24 GB, GPU recommended (RTX5090)    |
| Enterprise-level app       | 70B        | 43 GB               | 2×24 GB, GPU required (2×RTX5090)   |
|                            | 235B (Qwen3-235B) | 142 GB     | Data center GPU (H100) or Mac Studio M-series unified memory|
|                            | 671B (DeepSeek R1) | 404 GB     |   ''                                  |

### 💻 Example: Calling Ollama API from Python (Jupyter Notebook)

```python
import requests
from IPython.display import Markdown, display

# One question
question = "How to produce b-carotene in yarrowia lipolytica?"
system_prompt = "You are a helpful assistant."

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": question}
]

try:
    response = requests.post(
        "http://localhost:11434/v1/chat/completions",
        json={
            "model": "qwen3:8b",  # make sure this model is installed with `ollama list`
            "messages": messages,
            "max_tokens": 2000,
            "request_timeout": 10000
        }
    )
    response.raise_for_status()
    
    answer = response.json()["choices"][0]["message"]["content"].strip()
    
    # Display nicely in Markdown
    display(Markdown(f"**Q:** {question}\n\n{answer}"))

except Exception as e:
    print(f"Error getting response: {e}")
```

### 📊 Example: Reading GSM Reactions with Local LLM (qwen3:8b)

Constructing a **13C Metabolic Flux Analysis (MFA) model** often begins with simplifying a **Genome-scale model (GSM)**.  
This process requires carefully deciding which reactions belong to the *central carbon metabolism* (glycolysis, PPP, TCA cycle, anaplerotic shunts, and key exchange reactions).  

Manually reviewing ~2000 reactions one by one is **extremely time-consuming** and prone to inconsistency.  
Here, a Local LLM `qwen3:8b` can be used to quickly screen reactions, providing consistent Yes/No judgments.  
This makes LLMs a valuable assistant for researchers.

```python
from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
from openai import OpenAI

try:
    from tqdm.auto import tqdm  # type: ignore
    _USE_TQDM = True
except ModuleNotFoundError:
    _USE_TQDM = False

# ──────────────────────────────────────────────────────────────────────────────
# Config
INPUT_FILE = Path("iLst996.xlsx")
OUTPUT_FILE = Path("iLst996_with_MFA_flag(LLM).xlsx")
SHEET_NAME = "Reactions"           # must match the sheet's name exactly
MODEL_NAME = "qwen3:8b"            # verify with `ollama list`
NEW_COLUMN = "Include 13C_MFA"     # Yes/No

SYSTEM_PROMPT = (
    "/no_think\n"
    "You are an expert in metabolic engineering performing 13C metabolic flux analysis.\n"
    "Decide if a reaction belongs in a *simplified* central carbon metabolic model "
    "used for 13C MFA. Reply with exactly one word: Yes or No."
)

def user_prompt(name: str, equation: str) -> str:
    return (
        f"Reaction Name: {name}\n"
        f"Reaction Equation: {equation}\n\n"
        "Belongs to central carbon metabolism (glycolysis, PPP, TCA, anaplerotic shunts/"
        "glyoxylate cycle, or a direct transport/exchange connecting them)?"
    )

# ──────────────────────────────────────────────────────────────────────────────
# Improved normalisation
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)

def normalise_answer(text: str) -> str:
    """Coerce the model's response to exactly 'Yes' or 'No'."""
    cleaned = THINK_BLOCK_RE.sub("", text or "")   # remove think blocks
    # Search for first explicit Yes/No anywhere
    m = re.search(r"\b(yes|no)\b", cleaned, flags=re.IGNORECASE)
    if m:
        return "Yes" if m.group(1).lower() == "yes" else "No"
    # Fallback: look at first token
    word = cleaned.strip().split()[0].lower() if cleaned.strip() else ""
    return "Yes" if word.startswith("y") else "No"

# ──────────────────────────────────────────────────────────────────────────────
# Main
def main() -> None:
    # Point the OpenAI SDK at Ollama's OpenAI-compatible server
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="not-needed")

    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

    # Sanity check columns exist
    for col in ("Name", "Equation"):
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    answers: list[str] = []

    it = df.iterrows()
    if _USE_TQDM:
        it = tqdm(it, total=len(df), desc="Classifying reactions", unit="rxn")

    for _, row in it:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(row["Name"], row["Equation"])},
        ]

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=15,    # enough for “Yes” or “No”
        )

        raw_output = resp.choices[0].message.content or ""
        cleaned = normalise_answer(raw_output)

        # Print both raw + cleaned
        print(f"Reaction: {row['Name']}")
        print(f"Raw LLM output: {raw_output!r}")
        print(f"Normalized: {cleaned}")
        print("-" * 50)

        answers.append(cleaned)

    df[NEW_COLUMN] = answers
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"✓ Saved results to {OUTPUT_FILE.resolve()}")

if __name__ == "__main__":
    main()
```


## Week 5-6 Data retrieval and knowledge synthesis
Please follow the repository Advanced-Biotechnology-Series for NEKO and GraphRAG

https://github.com/xiao-zhengyang/Advanced-Biotechnology-Series 

<img width="1687" height="361" alt="image" src="https://github.com/user-attachments/assets/908fbe0a-c878-42ae-87e0-e49259c3f3d8" />

