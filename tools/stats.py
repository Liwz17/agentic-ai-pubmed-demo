from collections import Counter


def summarize_papers_stats(papers):
    n_papers = len(papers)

    journal_counter = Counter()
    keyword_counter = Counter()
    missing_abstract_count = 0
    abstract_lengths = []

    for p in papers:
        journal = p.get("journal")
        if journal:
            journal_counter[journal] += 1

        keywords = p.get("keywords") or []
        for kw in keywords:
            if kw:
                keyword_counter[kw] += 1

        abstract = p.get("abstract")
        if not abstract or abstract == "No abstract available":
            missing_abstract_count += 1
        else:
            abstract_lengths.append(len(abstract))

    avg_abstract_length = (
        sum(abstract_lengths) / len(abstract_lengths)
        if abstract_lengths else 0
    )

    return {
        "n_papers": n_papers,
        "top_journals": journal_counter.most_common(5),
        "top_keywords": keyword_counter.most_common(10),
        "missing_abstract_count": missing_abstract_count,
        "avg_abstract_length": round(avg_abstract_length, 2),
    }