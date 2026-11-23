
# eval/generation.py -- model loading, chat-template prompt builders, resumable greedy batch generation
import json
import os
import time
import numpy as np
import torch

ROUTER_TEMPLATE = (
    'You are the intent router of the KharidiKart customer-support assistant. '
    'Classify the customer message into exactly one intent from this list: {intents}. '
    'The category must be one of: {categories}. '
    'Reply with only a JSON object like {{"intent": "<intent>", "category": "<CATEGORY>"}} and nothing else.'
)


def make_router_system(intents, categories):
    return ROUTER_TEMPLATE.format(intents=", ".join(sorted(intents)), categories=", ".join(sorted(categories)))


def route_target(intent, category):
    return json.dumps({"intent": intent, "category": category})


def build_response_messages(system, instruction, exemplars=None):
    msgs = [{"role": "system", "content": system}]
    for ex in exemplars or []:
        msgs.append({"role": "user", "content": ex["instruction"]})
        msgs.append({"role": "assistant", "content": ex["response"]})
    msgs.append({"role": "user", "content": instruction})
    return msgs


def build_route_messages(router_system, instruction, exemplars=None):
    msgs = [{"role": "system", "content": router_system}]
    for ex in exemplars or []:
        msgs.append({"role": "user", "content": ex["instruction"]})
        msgs.append({"role": "assistant", "content": route_target(ex["intent"], ex["category"])})
    msgs.append({"role": "user", "content": instruction})
    return msgs


def render(tok, messages):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_model_4bit(model_id, adapter_path=None, cache_dir=None):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(quantization_config=bnb, device_map={"": 0}, cache_dir=cache_dir)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, **kw)
    except (TypeError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, **kw)
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tok, model


def _run_batch(model, tok, texts, max_new_tokens):
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                             top_p=None, top_k=None, repetition_penalty=1.0, pad_token_id=tok.pad_token_id)
    dt = time.time() - t0
    plen = enc["input_ids"].shape[1]
    return out[:, plen:].cpu(), dt, plen


def _gen_texts(model, tok, texts, max_new_tokens):
    try:
        gen, dt, plen = _run_batch(model, tok, texts, max_new_tokens)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(texts) == 1:
            raise
        h = len(texts) // 2
        return _gen_texts(model, tok, texts[:h], max_new_tokens) + _gen_texts(model, tok, texts[h:], max_new_tokens)
    res = []
    for j in range(len(texts)):
        toks = gen[j]
        n_new = int((toks != tok.pad_token_id).sum())
        res.append((tok.decode(toks, skip_special_tokens=True).strip(), n_new, dt / len(texts), plen))
    return res


def generate_chat(model, tok, messages_list, ids, max_new_tokens, batch_size, out_path):
    # Greedy, chat-template based, resumable: finished ids are read back from out_path and skipped.
    from tqdm.auto import tqdm
    done = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[r["id"]] = r
    todo = [i for i in range(len(messages_list)) if ids[i] not in done]
    texts = {i: render(tok, messages_list[i]) for i in todo}
    order = sorted(todo, key=lambda i: len(texts[i]))
    tok.padding_side = "left"
    with open(out_path, "a", encoding="utf-8") as f:
        for b in tqdm(range(0, len(order), batch_size), desc=os.path.basename(out_path)):
            idx = order[b:b + batch_size]
            res = _gen_texts(model, tok, [texts[i] for i in idx], max_new_tokens)
            for i, (txt, n_new, share, plen) in zip(idx, res):
                rec = {"id": ids[i], "output": txt, "n_new_tokens": n_new,
                       "batch_share_s": round(share, 4), "prompt_tokens": plen}
                done[ids[i]] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
    return [done[ids[i]] for i in range(len(messages_list))]


def latency_benchmark(model, tok, messages_list, max_new_tokens, n=10, warmup=1):
    # batch size 1; returns per-sample seconds, tokens/s and peak GPU memory (GB)
    for m in messages_list[:warmup]:
        _gen_texts(model, tok, [render(tok, m)], 16)
    torch.cuda.reset_peak_memory_stats()
    secs, toks = [], 0
    for m in messages_list[warmup:warmup + n]:
        txt, n_new, dt, _ = _gen_texts(model, tok, [render(tok, m)], max_new_tokens)[0]
        secs.append(dt)
        toks += n_new
    return {"seconds": secs, "tokens_per_s": float(toks / max(sum(secs), 1e-9)),
            "peak_gpu_gb": float(torch.cuda.max_memory_allocated() / 1e9)}
