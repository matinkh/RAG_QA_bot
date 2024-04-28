import argparse

from models.online_qa_rag import OnlineQaRag


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

    rag = OnlineQaRag(folder)
