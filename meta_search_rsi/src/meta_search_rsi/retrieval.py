# ORIGIN: ORIGINAL
# IMPLEMENTATION_NOTE: BM25 evidence retrieval and explicit diversity; no vector DB.
"""Metadata filtering precedes lexical scoring, including replay-run exclusion."""
from collections import Counter
from dataclasses import dataclass
import math
import re


@dataclass(frozen=True)
class RetrievedMemory:
    """A sourced piece of evidence, never an instruction to execute."""
    memory_id: str
    source: str
    relevance: float
    source_run: str
    source_nodes: list[str]
    raw_text: str
    abstracted_text: str
    kind: str
    outcome_label: str
    structure: str
    selection_reason: str


def tokens(text: str) -> list[str]:
    """Tokenize Latin words and CJK characters without external models."""
    return re.findall(r'[a-zA-Z_][a-zA-Z_0-9]*|[\u3040-\u30ff\u3400-\u9fff]', text.lower())


class Retriever:
    """An explicit, bounded memory collection; callers select source runs first."""

    def __init__(self, documents: list[dict], k1: float = 1.2, b: float = 0.75):
        self.documents = documents
        self.k1, self.b = k1, b

    def search(self, query: str, k: int = 6, *, exclude_runs: set[str] | None = None,
               exclude_tasks: set[str] | None = None, source: str | None = None,
               kind: str | None = None, structure: str | None = None) -> list[RetrievedMemory]:
        """Return success/failure/structural alternatives when matching evidence exists."""
        docs = [d for d in self.documents if d['source_run'] not in (exclude_runs or set())
                and d['task_name'] not in (exclude_tasks or set())
                and (source is None or d['source'] == source) and (kind is None or d['kind'] == kind)]
        if not docs or k <= 0:
            return []
        counts = [Counter(tokens(d['abstracted_text'] + '\n' + d['task_name'])) for d in docs]
        lengths = [sum(c.values()) for c in counts]
        average = sum(lengths) / len(lengths) or 1.0
        query_terms = set(tokens(query))
        frequencies = {term: sum(term in c for c in counts) for term in query_terms}
        scored = []
        for doc, count, length in zip(docs, counts, lengths):
            score = 0.0
            for term in query_terms:
                tf = count[term]
                if not tf:
                    continue
                idf = math.log(1 + (len(docs) - frequencies[term] + 0.5) / (frequencies[term] + 0.5))
                score += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / average))
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda pair: (-pair[0], pair[1]['memory_id']))
        selected, seen = [], set()
        categories = [('success', lambda d: d['outcome_label'] == 'success'),
                      ('failure', lambda d: d['outcome_label'] == 'failure'),
                      ('structural_alternative', lambda d: structure is not None and d['structure'] != structure)]
        for reason, predicate in categories:
            match = next(((s, d) for s, d in scored if predicate(d) and d['memory_id'] not in seen), None)
            if match and len(selected) < k:
                selected.append((*match, reason)); seen.add(match[1]['memory_id'])
        for score, doc in scored:
            if len(selected) >= k:
                break
            if doc['memory_id'] not in seen:
                selected.append((score, doc, 'relevance')); seen.add(doc['memory_id'])
        return [RetrievedMemory(d['memory_id'], d['source'], s, d['source_run'], d['source_nodes'],
                d['raw_text'], d['abstracted_text'], d['kind'], d['outcome_label'], d['structure'], reason)
                for s, d, reason in selected]
