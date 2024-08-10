from pathlib import Path
import json
from tqdm.auto import tqdm
from typing import Generator


class DataParser:
    """Dataset markup parser. Parse data from TriviaQA bench markup and return clean data for evaluation"""

    def __init__(self, path: Path | str, clean_data: bool = False) -> None:
        if isinstance(path, str):
            path = Path(path)

        assert path.exists(), f"File not found: {path}"
        self._file_paths = self.collect_data(path)

    @staticmethod
    def collect_data(dataset_path: Path) -> list[Path]:
        """Collect data paths"""
        file_paths = [
            file_path
            for file_path in dataset_path.iterdir()
            if file_path.is_file()
            and file_path.suffix == ".json"
            and "without-answers" not in file_path.name
        ]

        assert len(file_paths) > 0, f"No markup files found in {dataset_path}"
        print(f"▶️ Found {len(file_paths)} markup files")
        return file_paths

    def __getitem__(self, ind: int) -> Generator[dict[str, str], None, None]:
        """Yields instances of preprocessed data from json with markup. JSON file is accessed by index.
        Output dict structure: {"question":question, "gt_answer":gt_answer}"""
        file_path = self._file_paths[ind]
        raw_markup = json.load(file_path.open("r"))

        for data in tqdm(
            raw_markup,
            desc=f"Parsing {file_path.name}| Total files: {len(self._file_paths)}| Parsed files: {ind}",
        ):
            preprocessed_data = dict()
            try:
                gt_answer = data["Answer"]["Value"]
                question = data["Answer"]["Question"]
            except KeyError:
                continue
            preprocessed_data["question"] = question
            preprocessed_data["gt_answer"] = gt_answer
            yield preprocessed_data
        return

    def __len__(self):
        return len(self._file_paths)
