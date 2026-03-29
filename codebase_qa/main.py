from app.agent import build_agent

agent = build_agent()

while True:
    query = input("\nAsk about codebase: ")

    if query.lower() in ["exit", "quit"]:
        break

    result = agent.invoke({"input": query})

    print("\n--- ANSWER ---\n")
    print(result["output"])
