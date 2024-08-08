import time
import uuid

import torch
from baseline.config import settings
from qdrant_client import QdrantClient, models
from tqdm.auto import tqdm
from pathlib import Path
import copy
from .utils import safe_truncate
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL_PATH = "Alibaba-NLP/gte-base-en-v1.5"


class QdrantKnowledgeBase:
    """Knowledge base interface for RAG"""

    def __init__(self):
        self._qdrant_client = QdrantClient(
            host=settings.qdrant.host, port=settings.qdrant.port, timeout=60
        )
        self._collection_name = settings.qdrant.collection_name
        self._model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        # check collection
        self.validate_collection()

        # push model to device/devices
        print("-- Total number of devices: ", torch.cuda.device_count())
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print("-- Using multiple GPUs")
            self._model = torch.nn.DataParallel(self._model)
        elif torch.cuda.is_available():
            print("-- Using single GPU")
        else:
            print("-- Using CPU")

        self._model.to(self._device)
        self._model.eval()
        torch.set_grad_enabled(False)

    def validate_collection(self) -> None:
        """Validate collection existing. If collection does not exist - it will be created"""
        is_exists = self._qdrant_client.collection_exists(
            collection_name=settings.qdrant.collection_name
        )
        if is_exists:
            print("✅ Collection exists.")
        else:
            print("⚠️ Collection does not exist")
            print("\n▶️ Creating collection...")
            self._qdrant_client.create_collection(
                collection_name=self._collection_name,
                vectors_config=models.VectorParams(
                    size=768,
                    distance=models.Distance.COSINE,
                ),
            )
        return

    def clear_collection(self) -> None:
        """Clear collection and create new one"""
        print("Clearing collection...")
        self._qdrant_client.delete_collection(collection_name=self._collection_name)
        self._qdrant_client.create_collection(
            collection_name=self._collection_name,
            vectors_config=models.VectorParams(
                size=768,
                distance=models.Distance.COSINE,
            ),
        )
        print("✅ Collection cleared")
        return

    def upload_data(
        self,
        path: str | Path,
        parent_chunk_size: int = 8_000,
        child_chunk_size: int = 2_000,
        parent_chunk_overlap: int = 1_000,
    ) -> None:
        """Upload data to the vector database for RAG benchmark"""
        print("=" * 100 + "\nStarting uploading data to the vector database")

        if isinstance(path, str):
            path = Path(path)

        # search files with text
        text_file_paths = [_ for _ in path.glob("**/*.txt")]
        print("=" * 100 + "\n" + f"total files: {len(text_file_paths)}")

        self.clear_collection()

        # extract chunks and upload to qdrant
        for text_file_path in tqdm(
            text_file_paths, desc="Uploading data to the vector database"
        ):
            text = text_file_path.read_text()

            for parent_chunk_ind in range(
                0, len(text), parent_chunk_size - parent_chunk_overlap + 1
            ):
                # get parent chunk
                parent_chunk = safe_truncate(
                    text=text,
                    start=parent_chunk_ind,
                    stop=parent_chunk_ind + parent_chunk_size - parent_chunk_overlap,
                )
                # get child chunks
                child_chunks = [
                    safe_truncate(
                        text=parent_chunk,
                        start=chunk_ind,
                        stop=chunk_ind + child_chunk_size,
                    )
                    for chunk_ind in range(0, len(parent_chunk), child_chunk_size + 1)
                ]

                # get batch size
                batch_size = len(child_chunks) // 4 if len(child_chunks) > 3 else 1

                points = []
                for batch_ind in range(0, len(child_chunks), batch_size):
                    batch = child_chunks[batch_ind : batch_ind + batch_size]

                    # get embeddings
                    batch_dict = self._tokenizer(
                        batch,
                        max_length=8192,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    ).to(self._device)
                    outputs = self._model(**batch_dict)
                    embeddings = outputs.last_hidden_state[:, 0]
                    embeddings = F.normalize(embeddings, p=2, dim=1)

                    # preprocess and insert embeddings
                    for embedding_ind in range(embeddings.shape[0]):
                        point = models.PointStruct(
                            id=uuid.uuid4().hex,
                            payload={"text": parent_chunk},
                            vector=embeddings[embedding_ind].cpu().tolist(),
                        )
                        # Append point using deepcopy to avoid shallow copy, hence to avoid identical points
                        points.append(copy.deepcopy(point))
                # Upload points to qdrant. Retry if error raised
                while True:
                    try:
                        self._qdrant_client.upsert(
                            collection_name=self._collection_name,
                            points=points,
                        )
                        break
                    except Exception as e:
                        print(f"Error: {e}")
                        print("💤 Sleeping for 5 seconds...")
                        time.sleep(5)
                        print("⚠️ Retrying...")
                torch.cuda.empty_cache()
        return

    def get_similar_points(
        self, query: str, k_nearest: int = 9
    ) -> list[models.ScoredPoint]:
        """Get similar points (vectors) base on Cosine distance between query and vectors in the collection"""

        tokens = self._tokenizer(
            query, max_length=8192, truncation=True, return_tensors="pt"
        ).to(self._device)

        outputs = self._model(**tokens)
        embeddings = outputs.last_hidden_state[:, 0]
        embeddings = F.normalize(embeddings, p=2, dim=1)

        points = self._qdrant_client.search(
            collection_name=self._collection_name,
            query_vector=embeddings[0].cpu().numpy(),
            limit=k_nearest,
            with_payload=True,
            with_vectors=True,
        )
        return points
