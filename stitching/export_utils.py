import Pathlib
import uuid

def make_unique_dir(parent_path: Path, base_name: str) -> Path:
    dir_name = base_name + "_" + str(uuid.uuid4)
    new_path = parent_path / "dir_name"
    Path(new_path).mkdir(parents=True, exist_ok=False)
    return new_path
