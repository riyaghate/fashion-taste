"""Turn item embeddings into task scores (shared by all CLIP variants)."""
import numpy as np


def outfit_score(table, items):
    """Mean pairwise cosine similarity between the outfit's items."""
    V = table[items]
    S = V @ V.T
    n = len(items)
    return float((S.sum() - np.trace(S)) / (n * (n - 1)))


def outfit_scores(table, outfits):
    return np.array([outfit_score(table, o) for o in outfits])


def fitb_choice(table, question, answers):
    """Candidate with the highest mean similarity to the partial outfit."""
    Q = table[question]
    A = table[answers]
    return int(np.argmax((A @ Q.T).mean(1)))


def has_all(table, task):
    ids = task.get("items") or (task.get("question", []) + task.get("answers", []))
    return all(i in table for i in ids)


def score_tasks(table, tasks):
    """Rows of {task, id, score/choice}, in the same format the Gemini runner writes."""
    rows = []
    for t in tasks:
        if not has_all(table, t):
            continue
        if t["task"] == "fitb":
            rows.append({"task": "fitb", "id": t["id"], "choice": fitb_choice(table, t["question"], t["answers"])})
        else:
            rows.append({"task": t["task"], "id": t["id"], "score": outfit_score(table, t["items"])})
    return rows
