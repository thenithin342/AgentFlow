import sys
import asyncio
from backend.graph.tools import code_interpreter

if __name__ == "__main__":
    result = code_interpreter.invoke({"code": "print('hello world from multiprocessing!')"})
    print("RESULT:", result)
    if "hello world" in result:
        sys.exit(0)
    else:
        sys.exit(1)
