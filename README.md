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

### 🔍 Use NEKO for Knowledge Mining from PubMed Search  
NEKO integrates PubMed searches for literature mining. It automatically uses LLMs (e.g., ChatGPT, Qwen) to identify entities and causal relationships in studies.  
Use the uploaded jupyter notebook for using NEKO.
- Recommended model: ```qwen3:8b``` (download via [Ollama](https://ollama.ai/))
  ```bash
  ollama pull qwen3:8b
  ```
- Also download: ```nomic-embed-text:latest``` for embedding
  ```bash
  ollama pull nomic-embed-text:latest
  ```
- NEKO outputs include:  
  - Summarized reports  
  - Knowledge graphs (via PyVis)  

---

### 🧩 Use GraphRAG  

GraphRAG is an advanced retrieval-augmented generation (RAG) system that reads text and organizes it into knowledge graphs for better AI information retrieval.

1. **Install Ollama.**  
   [https://ollama.com/](https://ollama.com/)

2. **Install Python 3.11 for compatibility.**  
   [https://www.python.org/downloads/release/python-3110/](https://www.python.org/downloads/release/python-3110/)

3. **Open Command Prompt (cmd terminal).**  
   - On Windows: Press Win + R, type in "cmd" and run.  
   - On Mac: Press Cmd + Space, then type "Terminal".

4. **Install Jupyter Lab.**  
   Run the following command in the terminal.
   ```bash
   pip install jupyterlab
   ```
   Then start jupyter lab in the terminal
   ```bash
   jupyter lab
   ```

6. **Install GraphRAG.**  
   Type the following in the terminal
   ```bash
   pip install graphrag==1.2.0

   ```

7. **Download Qwen2.5-7b-INT4 model (or a larger model if your hardware supports it).**  
   ```bash
   ollama pull qwen2.5:7b
   ```

8. **Create a GraphRAG folder and navigate to your GraphRAG folder in Jupyter Lab.**  
   For example, navigate using the file explorer on the left of Jupyter Lab, and open a new terminal or type these in a new terminal
   ```bash
   cd (Path to your GraphRAG folder)
   ```

9. **Create an 4k context length qwen2.5:7b LLM.**  
   Run this in the terminal. This step is necessary to avoid truncations of LLM output. This command generates a settings file in the terminal's current directory.
   ```bash
   ollama show qwen2.5:7b --modelfile > settings.txt
   ```

10. **Add this into the settings.txt:**
   ```
   PARAMETER num_ctx 4096
   ```

11. **Run this in the terminal to create the extended context model.**
    ```bash
    ollama create qwen2.5:7b_4k -f settings.txt
    ```

12. **Download the ClassDemo example from GitHub.**

13. **[Optional] In the ClassDemo example, open settings.yaml and inspect the settings to make sure:**  
    - The correct LLM model loaded.
    - The correct embedding loaded.
    - The request timeout is high (over 10000) if you’re not using a GPU.
    - Visualization settings meet GraphRAG guidelines.  
    [GraphRAG Visualization Guide](https://microsoft.github.io/graphrag/visualization_guide/)

14. **[Optional] Use autotune to do prompt tuning.**  
    The prompt is highly dependent on the input text because GraphRAG uses prompts to extract key entities from the text. (This step has been completed for you)  
    Run the following command for automatic prompt tuning:
    ```bash
    python -m graphrag prompt-tune --root ./ClassDemo --config ./ClassDemo/settings.yaml
    ```
    *Note: Autotune might occasionally fail with smaller LLMs (<72B). After tuning, review the updated prompt in the GraphRAG prompt folder and, if satisfactory, move it to the ClassDemo prompt folder.*

15. **Run this command to start the indexing process.**  
    This step may take longer than 1 hour if you do not have a GPU or a powerful CPU. Make sure you turn off auto sleep of your computer. You may complete this in the computer lab in Whitaker Hall. Those computers have an entry level GPU, but they will auto logout if you are away for more than 1 hour.
    ```bash
    graphrag index --root ./ClassDemo
    ```

16. **After it finishes, you can start to ask some questions:**
    
    - **Some general questions like summarization (use global query)**
      ```bash
      graphrag query --root ./ClassDemo --method global --query "Summarize this text into bullet points."
      ```

    - **Some specific questions (use local query)**
      ```bash
      graphrag query --root ./ClassDemo --method local --query "Which hospital did Fleming work at?"
      ```

17. **Use Gephi or other compatible software for knowledge graph visualization.**
- **DRIFT Search** → Dynamic context-aware queries  

Visualization outputs (stored in `output/`) can be explored using **Gephi** or **Cytoscape** for graph-based knowledge discovery.  

