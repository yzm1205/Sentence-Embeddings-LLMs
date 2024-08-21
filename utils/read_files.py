from pathlib import Path
from typing import Union

def full_path(inp_dir_or_path: str) -> Path:
    """Returns full path"""
    return Path(inp_dir_or_path).expanduser().resolve()


def mkdir_p(inp_dir_or_path: Union[str, Path]) -> Path:
    """Give a file/dir path, makes sure that all the directories exists"""
    inp_dir_or_path = full_path(inp_dir_or_path)
    if inp_dir_or_path.suffix:  # file
        inp_dir_or_path.parent.mkdir(parents=True, exist_ok=True)
    else:  # dir
        inp_dir_or_path.mkdir(parents=True, exist_ok=True)
    return inp_dir_or_path
