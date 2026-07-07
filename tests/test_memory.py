import os
import shutil
from pathlib import Path
from langchain_core.messages import AIMessage, HumanMessage

from backend.memory.stm import should_compress, build_stm_prefix, STM_WINDOW
from backend.memory.ltm import write_ltm, read_ltm, LTM_ROOT

def test_stm_should_compress():
    assert not should_compress(0)
    assert should_compress(STM_WINDOW)
    assert not should_compress(STM_WINDOW + 1)
    assert should_compress(STM_WINDOW * 2)

def test_build_stm_prefix():
    prefix = build_stm_prefix("")
    assert prefix is None

    prefix = build_stm_prefix("User loves tests")
    assert prefix is not None
    assert "User loves tests" in prefix.content

def test_ltm_read_write():
    test_user = "test_user_ltm"
    user_dir = LTM_ROOT / test_user
    if user_dir.exists():
        shutil.rmtree(user_dir)
    
    # Empty read
    empty_mem = read_ltm(test_user, "hello")
    assert empty_mem == ""

    # Write some facts
    facts = [
        "User's favourite language is Python.",
        "User hates debugging C++ templates.",
        "User's dog is named Rex."
    ]
    write_ltm(test_user, facts, "thread-123")

    # Read relevant
    res1 = read_ltm(test_user, "What is my favourite language?")
    assert "Python" in res1
    
    res2 = read_ltm(test_user, "What is my dog's name?")
    assert "Rex" in res2

    # Cleanup
    shutil.rmtree(user_dir)
