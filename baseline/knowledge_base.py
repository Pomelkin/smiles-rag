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

        # push model to device/devices
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self._model = torch.nn.DataParallel(self._model)

        self._model.to(self._device)

    def validate_collection(self) -> None:
        """Validate collection existing. If collection does not exist - it will be created"""
        is_exists = self._qdrant_client.collection_exists(
            collection_name=settings.qdrant.collection_name
        )
        if is_exists:
            print("✅ Collection exists. Cleaning...")
            self._qdrant_client.delete_collection(collection_name=self._collection_name)
            print("✅ Collection cleaned")
        else:
            print("⚠️ Collection does not exist")

        print("▶️ Creating collection...")
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
            for parent_chunk_ind in range(
                0, len(text), parent_chunk_size - parent_chunk_overlap + 1
            ):
                # get parent chunk
                parent_chunk = self.safe_truncate(
                    text=text,
                    start=parent_chunk_ind,
                    stop=parent_chunk_ind + parent_chunk_size - parent_chunk_overlap,
                )
                # get child chunks
                child_chunks = [
                    self.safe_truncate(
                        text=parent_chunk,
                        start=chunk_ind,
                        stop=chunk_ind + child_chunk_size,
                    )
                    for chunk_ind in range(0, len(parent_chunk), child_chunk_size + 1)
                ]

                batch_size = len(child_chunks) // 2
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

                    # preprocess and insert embeddings
                    for embedding_ind in range(embeddings.shape[0]):
                        point = models.PointStruct(
                            id=uuid.uuid4().hex,
                            payload={"text": parent_chunk},
                            vector=embeddings[embedding_ind].cpu().tolist(),
                        )
                        points.append(point)

                self._qdrant_client.upsert(
                    collection_name=self._collection_name,
                    points=points,
                )
                torch.cuda.empty_cache()
        return

    @staticmethod
    def safe_truncate(text: str, start: int, stop: int) -> str:
        """Truncate text with safe bounds"""
        # validate right bound
        stop = stop if stop < len(text) - 1 else len(text) - 1

        # find first and last spaces. This need for guarantee that we will not cut word in the middle.
        while (text[start] != " " and start != 0) or (
            text[stop] != " " and stop != len(text) - 1
        ):
            start = start - 1 if text[start] != " " and start != 0 else start
            stop = stop + 1 if text[stop] != " " and stop != len(text) - 1 else stop
        return text[start : stop + 1]
