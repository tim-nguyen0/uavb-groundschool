from pathlib import Path
import uuid
from enum import Enum
from datetime import datetime



class id_type(Enum):
    UUID = 1
    DATETIME = 2
    CUSTOM = 3

def make_unique_dir(parent_path: Path, base_name: str, identifier: id_type=id_type.UUID, custom_id: str="") -> Path:

    id = ""

    match identifier:
        case id_type.UUID:
            id = str(uuid.uuid4())
        case id_type.DATETIME:
            id = str(datetime.now().strftime("%Y%m%d_%H%M%S"))
        case id_type.CUSTOM:
            id = custom_id
        case _:
            raise ValueError("Invalid ID Type")

    dir_name = base_name + "_" + id
    new_path = Path(parent_path) / dir_name
    try:
        Path(new_path).mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"Directory {new_path} already exists, defaulting datetime with increased precision")
        precise_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        dir_name_precise = base_name + "_" + precise_id
        new_path = Path(parent_path) / dir_name_precise
        Path(new_path).mkdir(parents=True, exist_ok=False)        
    return new_path

def repeat_func(f, n):
    def out(x):
        for _ in range(n):
            x = f(x)
        return x
    return out