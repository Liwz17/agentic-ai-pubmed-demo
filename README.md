# Agentic AI PubMed Demo

This project demonstrates a simple agentic AI workflow for biomedical literature analysis.

Pipeline:

1. Retrieve papers from PubMed
2. Compute simple statistical summaries
3. Use LLM to rerank papers by relevance
4. Generate a summary

Components:

agent.py  
Agent orchestration

tools/pubmed.py  
PubMed retrieval tool

tools/stats.py  
Statistical summary tool

llm.py  
LLM reranking and summarization

main.py  
Command line entry point
