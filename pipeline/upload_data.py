import argparse
from pipeline.knowledge_base import QdrantKnowledgeBase

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload data to the vector database for RAG benchmark"
    )

    parser.add_argument(
        "--path_to_data", type=str, required=True, help="Path to dataset"
    )
    parser.add_argument(
        "--parent_chunk_size", type=int, help="Parent chunk size", default=4_000
    )
    parser.add_argument(
        "--child_chunk_size", type=int, help="Child chunk size", default=1_000
    )
    parser.add_argument(
        "--parent_chunk_overlap", type=int, help="Parent chunk overlap", default=1_000
    )
    parser.add_argument("--batch_chunks", type=int, help="Batch chunks", default=1)
    parser.add_argument("--using_cache", type=bool, help="Using cache", default=True)
    parser.add_argument("--slice_start", type=int, help="Slice start", default=0)
    parser.add_argument("--slice_stop", type=int, help="Slice stop", default=-1)

    args = parser.parse_args()

    qdrant_kb = QdrantKnowledgeBase()

    qdrant_kb.upload_data(
        path=args.path_to_data,
        parent_chunk_size=args.parent_chunk_size,
        child_chunk_size=args.child_chunk_size,
        parent_chunk_overlap=args.parent_chunk_overlap,
        batch_chunks=args.batch_chunks,
        use_text_cache=args.using_cache,
        slice_start=args.slice_start,
        slice_stop=args.slice_stop,
    )
