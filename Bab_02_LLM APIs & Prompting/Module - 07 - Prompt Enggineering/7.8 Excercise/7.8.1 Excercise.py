import anthropic
import os
import json
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# =========================================================
# API Configuration (Using xKiro)
# =========================================================
client = anthropic.Anthropic(
    api_key=os.environ.get("XKIRO_API_KEY"),
    base_url="https://api.xkiro.com"
)
MODEL = "qwen/qwen3.7-flash:free"


# =========================================================
# Exercise 1: Code Review Assistant (3 Versions)
# =========================================================
BASIC_PROMPT = "You are a code reviewer. Review the following code."

INTERMEDIATE_PROMPT = """You are a code reviewer. Review the following code for bugs, security issues, and bad practices. Provide suggestions to fix them."""

EXPERT_PROMPT = """You are an expert senior software engineer. Review the following code and provide feedback in this strict structure:
1. Bugs & Security
2. Performance & Readability
3. Refactored Code
Be concise, direct, and strictly follow the format."""

CODE_SNIPPETS = [
    "def add(a, b): return a + b",
    "def login(user, pw):\n  query = f'SELECT * FROM users WHERE u={user} AND p={pw}'\n  db.execute(query)",
    "def get_data():\n  try:\n    pass\n  except:\n    pass",
    "def loop():\n  l = []\n  for i in range(1000000):\n    l.append(i)",
    "def foo(x=[]):\n  x.append(1)\n  return x"
]

def run_exercise_1():
    print("\n--- Exercise 1: Code Review Assistant ---")
    prompts = {"Basic": BASIC_PROMPT, "Intermediate": INTERMEDIATE_PROMPT, "Expert": EXPERT_PROMPT}
    
    # We will test only the 2nd snippet (SQL Injection) for demonstration to save time
    test_snippet = CODE_SNIPPETS[1]
    
    for level, sys_prompt in prompts.items():
        resp = client.messages.create(
            model=MODEL,
            max_tokens=512,
            system=sys_prompt,
            messages=[{"role": "user", "content": test_snippet}]
        )
        print(f"\n[{level} Output]:")
        print(resp.content[0].text[:300] + "...\n")


# =========================================================
# Exercise 2: PromptLibrary
# =========================================================
@dataclass
class PromptTemplate:
    name: str
    system: str
    user: str
    version: str = "1.0"
    last_eval_version: Optional[str] = None

class PromptLibrary:
    def __init__(self, storage_file="prompts.json"):
        self.storage_file = storage_file
        self.templates: dict[str, PromptTemplate] = {}

    def add(self, template: PromptTemplate):
        self.templates[template.name] = template

    def get(self, name: str) -> Optional[PromptTemplate]:
        return self.templates.get(name)

    def save(self):
        data = {name: t.__dict__ for name, t in self.templates.items()}
        with open(self.storage_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Library saved to {self.storage_file}")

    def load(self):
        if os.path.exists(self.storage_file):
            with open(self.storage_file, 'r') as f:
                data = json.load(f)
                self.templates = {k: PromptTemplate(**v) for k, v in data.items()}
            print(f"Library loaded from {self.storage_file}")

def run_exercise_2():
    print("\n--- Exercise 2: PromptLibrary ---")
    lib = PromptLibrary()
    template = PromptTemplate(
        name="hello_world",
        system="You are a helpful bot.",
        user="Say hello to {name}."
    )
    lib.add(template)
    lib.save()
    
    lib2 = PromptLibrary()
    lib2.load()
    print("Loaded template name:", lib2.get("hello_world").name)


# =========================================================
# Exercise 3: Automatic JSON Repair
# =========================================================
def safe_json_parse(text: str) -> dict:
    # 1. Try raw
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print("Raw JSON parsing failed. Trying to strip markdown...")
    
    # 2. Strip markdown fences
    clean_text = text.strip()
    if clean_text.startswith("```"):
        lines = clean_text.split('\n')
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines[-1].startswith("```"):
            lines = lines[:-1]
        clean_text = '\n'.join(lines).strip()
        try:
            return json.loads(clean_text)
        except json.JSONDecodeError:
            print("Markdown stripping failed. Asking LLM to fix...")
            
    # 3. Ask LLM to fix
    prompt = f"Fix this broken JSON string and return ONLY the valid JSON object. No explanation.\n{text}"
    resp = client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    fixed = resp.content[0].text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(fixed)

def run_exercise_3():
    print("\n--- Exercise 3: Safe JSON Parse ---")
    broken_json = """```json\n{\n  "name": "Alice",\n  "age": 25,\n  "is_student": true,\n```""" # Missing closing brace
    result = safe_json_parse(broken_json)
    print("Successfully repaired JSON:", result)


# =========================================================
# Exercise 4: CoT Ranking Prompt
# =========================================================
COT_RANKING_PROMPT = """You are an AI benchmark analyst. 
Given a list of 10 LLM evaluation scores across 3 tasks, rank the top 3 models by their average score.
Think step by step using this exact XML structure:

<thinking>
1. Calculate the average score for each model.
2. Sort the models based on the average score in descending order.
3. Identify the top 3 models.
4. Draft a 2-sentence recommendation based on the top models' strengths.
</thinking>
<answer>
1. [Rank 1 Model] (Avg: [Score])
2. [Rank 2 Model] (Avg: [Score])
3. [Rank 3 Model] (Avg: [Score])

Recommendation: [2-sentence recommendation]
</answer>

Return ONLY the XML structure."""

def run_exercise_4():
    print("\n--- Exercise 4: CoT Ranking ---")
    input_data = """
    Model A: Math=90, Logic=80, Code=70
    Model B: Math=85, Logic=85, Code=85
    Model C: Math=60, Logic=60, Code=60
    Model D: Math=95, Logic=90, Code=85
    Model E: Math=50, Logic=40, Code=30
    Model F: Math=88, Logic=82, Code=75
    Model G: Math=78, Logic=72, Code=80
    Model H: Math=92, Logic=88, Code=90
    Model I: Math=65, Logic=70, Code=75
    Model J: Math=80, Logic=90, Code=85
    """
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=COT_RANKING_PROMPT,
        messages=[{"role": "user", "content": input_data}]
    )
    print(resp.content[0].text)


# =========================================================
# Main Execution
# =========================================================
if __name__ == "__main__":
    # Uncomment the ones you want to run
    run_exercise_1()
    run_exercise_2()
    run_exercise_3()
    run_exercise_4()
    pass
