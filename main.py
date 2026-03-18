from agent import DemoAgent


# def main():
#     agent = DemoAgent()
#     query = input("Enter research query: ")
#     results = agent.run(query)

#     print("\n=== Candidate Set Statistics (Top 30 Retrieved) ===\n")
#     print(f"Number of candidate papers: {results['stats']['n_papers']}")
#     print(f"Missing abstracts: {results['stats']['missing_abstract_count']}")
#     print(f"Average abstract length: {results['stats']['avg_abstract_length']}")

#     print("\nTop journals:")
#     for journal, count in results["stats"]["top_journals"]:
#         print(f"- {journal}: {count}")

#     print("\nTop keywords:")
#     for keyword, count in results["stats"]["top_keywords"]:
#         print(f"- {keyword}: {count}")

#     print("\n=== LLM Summary of Reranked Top 10 ===\n")
#     print(results["summary"])

#     print("\n=== Reranked Top Papers ===\n")
#     for i, p in enumerate(results["papers"], 1):
#         print(f"{i}. PMID: {p['pubmed_id']} | {p['title']}")

def main():
    agent =  DemoAgent()
    print("Clinical Trial Agent (type 'exit' to quit)\n")

    while True:
        user_input = input("Query: ")

        if user_input.lower() in ["exit", "quit"]:
            break

        try:
            df = agent.run_agent(user_input)
            print(f"\nFound {len(df)} trials\n")

        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    main()
