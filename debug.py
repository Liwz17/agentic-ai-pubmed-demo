from tools.pubmed_trials import PubMedFilter, build_query_B_llm, build_query_C, build_query_A

f = PubMedFilter(
    pub_date_start="2020/01/01",
    pub_date_end="2024/12/31",
    publication_types=["rct", "meta_analysis"],
)

fields = {"disease": "lung cancer", "drugs": ["pembrolizumab"], "phase": "PHASE3", "nct_id": "NCT12345678"}
semantic_terms = {"disease_terms": ["NSCLC"], "drug_terms": ["pembrolizumab"], "setting_terms": ["first-line"], "other_terms": []}

print(build_query_A(fields, pubmed_filter=f))
print()
print(build_query_C(fields, pubmed_filter=f))
print()
print(build_query_B_llm(fields, semantic_terms, pubmed_filter=f))


