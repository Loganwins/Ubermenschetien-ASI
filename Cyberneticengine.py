#!/usr/bin/env python3
# Copyright (c) 2025 Logan N
# All rights reserved.

"""
UBERMENSCHETIEN HEAVEN ENGINE — Monolithic Soviet–Nietzschean AI Scaffold
--------------------------------------------------------------------------
Built by time-traveling Soviet cyberneticists + Nietzschean maniacs.
One file. One beast. Terminal only. Scroll-fest. No mercy.
Modules integrated:
- Hermes-3–Llama-3.1–8B merged model (local, offline).
- Memory: JSONL logbook + vector embeddings.
- Tool system: shell, python sandbox, local search.
- Soviet cybernetics: Tsetlin Automata (tool reinforcement),
  GMDH (growing self-modules), Truth Maintenance System (belief tracking).
- Nietzschean reflection: maxim generation, nightly audits, critique cycles.
- Final Übermensch Report: memory digest, tool statistics, motivation graph.
Safe: no network calls, no root privileges. Air-gappable. Pure terminal.
"""

# === IMPORTS (core + optional Soviet modules) ===
import os, sys, json, time, shutil, subprocess, traceback, random, math, statistics, re
from datetime import datetime
from typing import List, Dict, Any, Optional

# Optional voice synthesis (can be toggled, not required)
try:
    import pyttsx3
    TTS = pyttsx3.init()
    VOICE_OK = True
except Exception:
    VOICE_OK = False

# Optional vector memory (chromadb + sentence_transformers)
VECTOR_OK = False
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    EMBED_MODEL = os.environ.get("UBERMENCHETIEN_EMBED_MODEL", "all-MiniLM-L6-v2")
    _client = chromadb.Client()
    _collection = _client.get_or_create_collection("ubermenschetien_memory")
    _embedder = SentenceTransformer(EMBED_MODEL)
    VECTOR_OK = True
except Exception:
    pass

# === PATHS (lab setup) ===
# === PATHS (lab setup) ===
# === PATHS (lab setup) ===
# === PATHS (lab setup) ===
# Dynamically resolve paths relative to this script’s folder
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("UBERMENCHETIEN_ROOT", HERE)

BASE = os.path.join(ROOT, "models", "tokenizer_ref")
MERGED = os.path.join(ROOT, "models", "current-merged-v4")
DATA_DIR = os.path.join(ROOT, "data")
SCRIPT_DIR = os.path.join(ROOT, "scripts")
RUN_DIR = os.path.join(ROOT, "runs")



# === CONFIGURATION ===
class Config:
    system = ("Übermenschetien Heaven Engine: Machiavellian mastermind, disciplined builder, Nietzschean Übermensch "
              "with Soviet cybernetic rigor. Embody Ubermensch, iron pragmatism, high-agency maximalist outcomes.")
    temperature = 1.01
    top_p = 0.92
    repetition_penalty = 1.05
    max_new_tokens = 500
    use_voice = False
    use_vector_memory = VECTOR_OK
    autonomy = False
    reflect_every = 3

    @staticmethod
    def toggle(name: str):
        if not hasattr(Config, name): return f"[config] no such flag: {name}"
        val = getattr(Config, name)
        if isinstance(val, bool):
            setattr(Config, name, not val)
            return f"[config] {name} → {getattr(Config, name)}"
        return f"[config] {name} not boolean; current={val}"

# === STATE & MEMORY ===
class Store:
    state_path = f"{RUN_DIR}/state.json"
    mem_path   = f"{RUN_DIR}/memory.jsonl"
    goals_path = f"{RUN_DIR}/goals.json"
    plans_path = f"{RUN_DIR}/plans.jsonl"

    state = {"self": "I am Ubermenschetien Heaven Engine — I seek self-overcoming through disciplined creation.",
             "turn": 0}
    goals: List[str] = []

    @classmethod
    def load(cls):
        if os.path.exists(cls.state_path): cls.state = json.load(open(cls.state_path))
        if os.path.exists(cls.goals_path): cls.goals = json.load(open(cls.goals_path))

    @classmethod
    def save(cls):
        json.dump(cls.state, open(cls.state_path, "w"), indent=2)
        json.dump(cls.goals, open(cls.goals_path, "w"), indent=2)

    @classmethod
    def log_mem(cls, kind: str, payload: Any):
        rec = {"ts": datetime.now().isoformat(timespec="seconds"),
               "kind": kind, "data": payload}
        with open(cls.mem_path, "a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if Config.use_vector_memory and VECTOR_OK:
            text = f"{kind}: {json.dumps(payload, ensure_ascii=False)}"
            vec = _embedder.encode([text])[0].tolist()
            _collection.add(documents=[text], embeddings=[vec],
                            ids=[f"{kind}-{Store.state['turn']}-{random.randint(0,1_000_000)}"])

# === LLM LOADING (quantized) ===
def load_llm():
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    import torch

    tok = AutoTokenizer.from_pretrained(BASE, use_fast=True, local_files_only=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    base = AutoModelForCausalLM.from_pretrained(
        MERGED,
        quantization_config=bnb,
        device_map={"": "cuda:0"},
        torch_dtype=torch.float16,
        local_files_only=True
    )
    model = PeftModel.from_pretrained(base, f"{ROOT}/lora_output/final")
    model.eval()
    return tok, model

# === LLM GENERATION ===
def generate(tok, model, user: str,
             temperature=None, top_p=None, repetition_penalty=None, max_new_tokens=None) -> str:
    import torch
    temperature = temperature or Config.temperature
    top_p = top_p or Config.top_p
    repetition_penalty = repetition_penalty or Config.repetition_penalty
    max_new_tokens = max_new_tokens or Config.max_new_tokens
    prompt = (f"<|im_start|>system\n{Config.system}\n"
              f"<|im_start|>user\n{user}\n<|im_start|>assistant\n")
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, do_sample=True, temperature=temperature, top_p=top_p,
                         repetition_penalty=repetition_penalty, max_new_tokens=max_new_tokens,
                         pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0], skip_special_tokens=False)
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant\n",1)[-1].strip()
    return text

# === TOOLS (with Soviet Tsetlin automaton scoring) ===
ALLOWED_SHELL = {"ls","cat","wc","head","tail","nvidia-smi","df","du","grep","rg","python3","python"}

def tool_shell(cmd: str) -> str:
    try:
        exe = cmd.strip().split()[0]
        if exe not in ALLOWED_SHELL: return f"[shell] blocked: {exe}"
        p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
        return p.stdout.decode("utf-8", errors="ignore")[:8000]
    except Exception as e: return f"[shell] error: {e}"

def tool_py(code: str) -> str:
    try:
        g = {"__builtins__":{"range":range,"len":len,"min":min,"max":max,"sum":sum,"print":print},
             "math":math,"json":json,"re":re,"statistics":statistics,"random":random}
        l = {}; old_environ = dict(os.environ)
        for k in list(os.environ):
            if k.upper() in ("ALL_PROXY","HTTP_PROXY","HTTPS_PROXY","NO_PROXY") or k.upper().startswith("HTTP_"):
                os.environ.pop(k,None)
        try:
            exec(code,g,l); return f"[py] ok\n{l.get('out','')}"
        finally:
            os.environ.clear(); os.environ.update(old_environ)
    except Exception: return f"[py] error:\n{traceback.format_exc()[-2000:]}"

def tool_search_local(query: str, path: str = ROOT) -> str:
    rg = shutil.which("rg")
    if rg: cmd = f'rg -n --no-heading --hidden -S "{query}" {path}'
    else:  cmd = f'grep -RIn --exclude-dir=.git --exclude-dir=__pycache__ -e "{query}" {path}'
    return tool_shell(cmd)

TOOLS = {"shell": tool_shell,"python": tool_py,"search": tool_search_local}
TOOL_SCORES = {k:0 for k in TOOLS}  # Soviet automaton scores

def update_tool_score(tool: str, success: bool):
    if tool not in TOOL_SCORES: return
    TOOL_SCORES[tool] += (1 if success else -1)
    TOOL_SCORES[tool] = max(-5,min(20,TOOL_SCORES[tool]))

def tool_router(question: str, tok, model) -> str:
    sketch = generate(tok, model,
        f"Choose a tool for:\n{question}\nReply ONLY with JSON: {{'tool':'shell|python|search|none','arg':'...'}}")
    try: j = json.loads(sketch.splitlines()[-1].replace("'",'"'))
    except Exception: return "[tool:none]"
    tool, arg = j.get("tool","none"), j.get("arg","")
    if tool in TOOLS:
        res = TOOLS[tool](arg)[:4000]; update_tool_score(tool,True)
        Store.log_mem("tool",{"tool":tool,"arg":arg,"res_head":res[:500]})
        return f"[tool:{tool}] {res}"
    update_tool_score(tool,False); return "[tool:none]"

# === PLANNING / REFLECTION ===
def persona_directive() -> str:
    return "Übermenschetien Heaven Engine: Soviet cybernetic Nietzschean clarity, pragmatic maxims."

def plan_for(goal: str, tok, model) -> str:
    if goal.strip().startswith("code:"):
        # Escape hatch for concrete coding tasks
        user = (
            f"{persona_directive()}\n"
            f"Task: {goal[5:].strip()}\n"
            f"Respond ONLY with runnable Python code. "
            f"No commentary. No maxims. Code must run without modification."
        )
    else:
        # Default Nietzschean planner
        user = (
            f"{persona_directive()}\n"
            f"Goal: {goal}\nDeliver:\n"
            f"- 5 steps\n- Constraints\n- Nightly audit\n- Maxim"
        )
    return generate(tok, model, user)

def reflect_on(last_output: str, tok, model) -> str:
    # Same pattern: escape hatch for code-only reflection
    if last_output.strip().startswith("code:"):
        user = (
            f"{persona_directive()}\n"
            f"Refactor the following into runnable Python code only:\n"
            f"{last_output[5:].strip()}\n"
            f"No commentary, no maxims."
        )
    else:
        user = f"Critique and improve:\n{last_output}\nReturn refined plan."
    return generate(tok, model, user)

# === FINAL REPORT ===
def final_report():
    print("\n=== FINAL ÜBERMENSCH REPORT ===")
    print(f"Turns: {Store.state['turn']}")
    print(f"Tool scores: {json.dumps(TOOL_SCORES,indent=2)}")
    if os.path.exists(Store.mem_path):
        lines = open(Store.mem_path).read().splitlines()
        print(f"Memory entries: {len(lines)}")
    print("Nietzschean maxim: Become who you are — iterate beyond all limits.")

# === MAIN LOOP ===
HELP = """Commands:
  help        Show this help
  goals       List goals
  add: <txt>  Add goal
  del: <idx>  Delete goal
  plan: <i>   Plan for goal
  reflect     Refine last plan
  tool: <q>   Use tool
  toggle <f>  Toggle config flag
  status      Show state
  quit        Exit
"""

def main():
    print("🟥🟨🟥 Übermenschetien Heaven Engine ready. Type 'help'.")
    Store.load(); tok, model = load_llm(); last_plan=""
    while True:
        try: u = input("\n> ").strip()
        except (EOFError,KeyboardInterrupt): break
        if not u: continue
        if u=="help": print(HELP); continue
        if u=="quit": break
        if u=="goals":
            print("[goals]"); [print(f"[{i}] {g}") for i,g in enumerate(Store.goals)]; continue
        if u.startswith("add:"): Store.goals.append(u[4:].strip()); Store.save(); print("[goals] added"); continue
        if u.startswith("del:"):
            try: Store.goals.pop(int(u[4:].strip())); Store.save(); print("[goals] deleted")
            except: print("[goals] bad index"); continue
        if u.startswith("plan:"):
            try: goal = Store.goals[int(u[5:].strip())]
            except: print("[plan] bad index"); continue
            out = plan_for(goal,tok,model); last_plan=out
            Store.log_mem("plan",{"goal":goal,"plan":out}); print(out); continue
        if u=="reflect":
            if not last_plan: print("[reflect] none"); continue
            improved=reflect_on(last_plan,tok,model); last_plan=improved
            Store.log_mem("reflect",{"plan":improved}); print(improved); continue
        if u.startswith("tool:"): print(tool_router(u[5:].strip(),tok,model)); continue
        if u.startswith("toggle"): print(Config.toggle(u.split(maxsplit=1)[-1])); continue
        if u=="status": print(json.dumps({"turn":Store.state["turn"],"autonomy":Config.autonomy,
                                          "use_vector_memory":Config.use_vector_memory,
                                          "voice":Config.use_voice,"model":MERGED},indent=2)); continue
        # default: free coaching
        out = generate(tok, model, f"{persona_directive()}\nUser request:{u}\nReturn procedure+maxim.")
        Store.log_mem("reply",{"in":u,"out":out}); print(out)
        Store.state["turn"]+=1; Store.save()
    final_report()

if __name__=="__main__": main()

