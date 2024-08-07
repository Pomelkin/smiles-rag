import argparse
from baseline.knowledge_base import QdrantKnowledgeBase

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload data to the vector database for RAG benchmark"
    )

    parser.add_argument("--qdrant_host", type=str, required=True, help="Qdrant host")
    parser.add_argument("--qdrant_port", type=int, required=True, help="Qdrant port")
    parser.add_argument(
        "--qdrant_collection_name",
        type=str,
        required=True,
        help="Qdrant collection name",
    )
    parser.add_argument(
        "--path_to_data", type=str, required=True, help="Path to dataset"
    )
    parser.add_argument("--parent_chunk_size", type=int, help="Parent chunk size")
    parser.add_argument("--child_chunk_size", type=int, help="Child chunk size")
    parser.add_argument(
        "--parent_chunk_overlap", type=float, help="Parent chunk overlap"
    )

    args = parser.parse_args()

    qdrant_kb = QdrantKnowledgeBase()

    qdrant_kb.upload_data(
        path=args.path_to_data,
        parent_chunk_size=args.parent_chunk_size,
        child_chunk_size=args.child_chunk_size,
        parent_chunk_overlap=args.parent_chunk_overlap,
    )
