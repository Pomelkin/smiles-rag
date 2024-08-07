import uuid
import torch
from baseline.config import settings
from qdrant_client import QdrantClient, models
from tqdm.auto import tqdm
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

MODEL_PATH = "Alibaba-NLP/gte-base-en-v1.5"


class QdrantKnowledgeBase:
    """Knowledge base interface for RAG"""

    def __init__(self):
        self._qdrant_client = QdrantClient(
            host=settings.qdrant.host, port=settings.qdrant.port
        )
        self._collection_name = settings.qdrant.collection_name
        self._model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        # check collection
        self.validate_collection()

        # push model to device
        self._model.to(self._device)

    def validate_collection(self) -> None:
        """Validate collection existing. If collection does not exist - it will be created"""
        is_exists = self._qdrant_client.collection_exists(
            collection_name=settings.qdrant.collection_name
        )
        if is_exists is None:
            print("⚠️ Collection does not exist. Creating...")
            self._qdrant_client.create_collection(
                collection_name=self._collection_name,
                vectors_config=models.VectorParams(
                    size=768,
                    distance=models.Distance.COSINE,
                ),
            )
        return

    def upload_data(
        self,
        path: str | Path,
        parent_chunk_size: int = 10_000,
        child_chunk_size: int = 2_000,
        parent_chunk_overlap: int = 1_000,
    ) -> None:
        """Upload data to the vector database for RAG benchmark"""
        if isinstance(path, str):
            path = Path(path)

        # search files with text
        text_file_paths = [_ for _ in path.glob("**/*.txt")]
        print("=" * 100 + "\n" + f"total files: {len(text_file_paths)}")

        # extract chunks and upload to qdrant
        for text_file_path in tqdm(
            text_file_paths, desc="Uploading data to the vector database"
        ):
            text = text_file_path.read_text()
            print(f"Uploading file: {text_file_path}, total length: {len(text)}")
            for parent_chunk_ind in range(
                0, len(text), parent_chunk_size - parent_chunk_overlap + 1
            ):
                # get parent chunk
                parent_chunk = self.safe_truncate(
                    text=text,
                    start=parent_chunk_ind,
                    stop=parent_chunk_ind + parent_chunk_size - parent_chunk_overlap,
                )
                print(f"Parent chunk length: {len(parent_chunk)}")
                # get child chunks
                child_chunks = [
                    self.safe_truncate(
                        text=parent_chunk,
                        start=chunk_ind,
                        stop=chunk_ind + child_chunk_size,
                    )
                    for chunk_ind in range(0, len(parent_chunk), child_chunk_size + 1)
                ]
                print(f"Child chunks length: {len(child_chunks)}")
                # get embeddings
                batch_dict = self._tokenizer(
                    child_chunks,
                    max_length=8192,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self._device)
                outputs = self._model(**batch_dict)
                embeddings = outputs.last_hidden_state[:, 0]

                # preprocess and insert embeddings
                points = []
                for embedding_ind in range(embeddings.shape[0]):
                    point = models.PointStruct(
                        id=uuid.uuid4(),
                        payload={"text": parent_chunk},
                        vector=embeddings[embedding_ind].cpu().tolist(),
                    )
                    points.append(point)

                self._qdrant_client.upsert(
                    collection_name=self._collection_name,
                    points=points,
                )
        return

    @staticmethod
    def safe_truncate(text: str, start: int, stop: int) -> str:
        """Truncate text with safe bounds"""
        while text[start] != text[stop] != " ":
            start = start - 1 if text[start] != " " or start == 0 else start
            stop = stop + 1 if text[stop] != " " or stop == len(text) - 1 else stop
        return text[start : stop + 1]
