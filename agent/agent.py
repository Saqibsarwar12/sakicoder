from .tools import list_files, read_file, write_file, run_command


class Agent:
    def __init__(self, safe_mode: bool = True):
        self.safe_mode = safe_mode

    def handle(self, instruction: str):
        # Very small dispatch: if instruction mentions list/read/run
        if instruction.startswith("list_files"):
            return list_files(instruction.split("(")[1].rstrip(")\n\r\'") )
        if instruction.startswith("read_file"):
            path = instruction.split("(")[1].strip(" )'\"")
            return read_file(path)
        if instruction.startswith("write_file"):
            # format: write_file(path, content)
            return {"ok": False, "error": "write_file not implemented in simple agent"}
        return {"ok": False, "error": "unknown instruction"}
