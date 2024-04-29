import argparse
import time

from models.online_mistral_qa_rag import OnlineMistralQaRag
from models.online_openai_qa_rag import OnlineOpenAiQaRag


def _parse_arguments() -> str:
    parser = argparse.ArgumentParser(description="Process files")
    parser.add_argument("--folder", type=str, help="The context location.")

    # Parse command-line arguments
    args = parser.parse_args()

    folder = args.folder
    return folder


if __name__ == "__main__":
    # export HF_TOKEN=<YOUR_HUGGINGFACE_ACCESS_TOKEN>
    # python app.py --folder=<FOLDER_ADDRESS>
    folder = _parse_arguments()
    print(f"Folder selected: {folder}")

    tic = time.time()
    rag = OnlineOpenAiQaRag(folder) #OnlineMistralQaRag(folder)
    toc = time.time()
    print(f"Took {(toc-tic):.2f} seconds to initialize the RAG.\n")

    while True:
        user_input = input("Question (type 'quit' to exit):   ")
        if user_input.strip().lower == "quit":
            break
        answer = rag.invoke(question=user_input)

        print()
        print("-"*25)
        print(f"Question: {user_input}")
        print(f"Answer:\n{answer}")
        print("-"*25)
        print()
