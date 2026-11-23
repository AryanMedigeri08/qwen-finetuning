
# eval/metrics.py -- evaluation harness (T17): intent F1, ROUGE-L, BERTScore, perplexity, leakage, latency
import re
import json
import numpy as np

PLACEHOLDER_RE = re.compile(r"\{\{.*?\}\}")
INVALID = "__INVALID__"


def extract_placeholders(text):
    return PLACEHOLDER_RE.findall(text or "")


def has_placeholder(text):
    return bool(PLACEHOLDER_RE.search(text or ""))


def placeholder_leakage(responses):
    responses = list(responses)
    if not responses:
        return 0.0
    return 100.0 * float(np.mean([has_placeholder(r) for r in responses]))


def parse_route_json(text, valid_intents=None):
    # returns (intent, category); (None, None) if nothing usable
    if text is None:
        return None, None
    m = re.search(r"\{.*?\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            intent = str(obj.get("intent", "")).strip()
            cat = str(obj.get("category", "")).strip()
            if intent:
                return intent, (cat or None)
        except Exception:
            pass
    m = re.search(r"intent\W{0,4}\s*[:=]\s*\W{0,2}([a-z_]+)", text)
    if m:
        return m.group(1), None
    if valid_intents:
        low = text.lower()
        hits = [i for i in valid_intents if i in low]
        if len(hits) == 1:
            return hits[0], None
    return None, None


def intent_metrics(gold, pred, labels=None):
    from sklearn.metrics import accuracy_score, f1_score
    gold = list(gold)
    pred = [p if p is not None else INVALID for p in pred]
    labels = sorted(set(labels) if labels is not None else set(gold))
    return {
        "intent_acc": float(accuracy_score(gold, pred)),
        "intent_macro_f1": float(f1_score(gold, pred, labels=labels, average="macro", zero_division=0)),
        "invalid_rate": float(np.mean([p == INVALID for p in pred])),
    }


def rouge_l(preds, refs):
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    f = [sc.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]
    return float(np.mean(f)), f


_BERT = {}


def bertscore_f1(preds, refs, batch_size=32, model_type="roberta-large", device="cuda"):
    from bert_score import BERTScorer
    key = (model_type, device)
    if key not in _BERT:
        _BERT[key] = BERTScorer(model_type=model_type, lang="en", device=device)
    preds = [p if p.strip() else "." for p in preds]
    _, _, F = _BERT[key].score(preds, list(refs), batch_size=batch_size)
    return float(F.mean()), F.tolist()


def response_perplexity(model, tok, items, batch_size=4, max_len=1024):
    # items: list of (messages_without_assistant, reference_response). Loss on response tokens only.
    import torch
    model.eval()
    tot_nll, tot_tok = 0.0, 0
    pad = tok.pad_token_id
    for i in range(0, len(items), batch_size):
        seqs, labs = [], []
        for msgs, resp in items[i:i + batch_size]:
            ptxt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            p_ids = tok(ptxt, add_special_tokens=False)["input_ids"]
            r_ids = tok(resp + tok.eos_token, add_special_tokens=False)["input_ids"]
            ids = (p_ids + r_ids)[:max_len]
            lab = ([-100] * len(p_ids) + r_ids)[:max_len]
            seqs.append(ids)
            labs.append(lab)
        L = max(len(s) for s in seqs)
        ids_t = torch.tensor([s + [pad] * (L - len(s)) for s in seqs], device=model.device)
        att_t = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs], device=model.device)
        lab_t = torch.tensor([l + [-100] * (L - len(l)) for l in labs], device=model.device)
        with torch.no_grad():
            logits = model(input_ids=ids_t, attention_mask=att_t).logits[:, :-1, :].float()
        tgt = lab_t[:, 1:]
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                                                 ignore_index=-100, reduction="sum")
        tot_nll += float(loss.item())
        tot_tok += int((tgt != -100).sum().item())
        del logits
    return float(np.exp(tot_nll / max(tot_tok, 1)))


def latency_stats(seconds):
    a = np.asarray(list(seconds), dtype=float)
    return {"p50_s": float(np.percentile(a, 50)), "p95_s": float(np.percentile(a, 95))}
