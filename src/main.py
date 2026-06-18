from query import ask_question

print("=" * 50)
print("Document Q&A Bot")
print("Type 'exit' to quit")
print("=" * 50)

while True:

    question = input("\nQuestion: ")

    if question.lower() == "exit":
        print("\nGoodbye!")
        break

    try:
        result = ask_question(question)

        print("\nAnswer:")
        print(result["answer"])

        print("\nSources Used:")

        for source in set(result["citations"]):
            print(f"- {source}")

    except Exception as e:
        print(f"\nError: {e}")