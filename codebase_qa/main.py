from app.agent import build_agent
from app.config import logger

agent = build_agent()

while True:
    try:
        query = input("\nAsk about codebase: ")

        if query.lower() in ["exit", "quit"]:
            break

        logger.debug(f"[USER QUERY] {query}")

        result = agent.invoke({"input": query})

        print("\n--- ANSWER ---\n")
        print(result.get("output", "No output returned"))

    except KeyboardInterrupt:
        print("\nExiting...")
        break

    except Exception as e:
        logger.error(f"[RUNTIME ERROR] {e}")
        print("Something went wrong. Check logs.")
