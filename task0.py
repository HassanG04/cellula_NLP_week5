from assistant.services.langgraph_code_service import run_code_assistant

if __name__ == "__main__":
    print(run_code_assistant(input("Generate or explain Python: ")))
