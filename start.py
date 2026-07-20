"""兼容入口：保留 `python start.py ...` 用法，实际逻辑在 rag_system.cli。

推荐用安装后的控制台脚本：`uv run rag ...`。
"""

import sys
from pathlib import Path

# 未安装包时（直接 python start.py），把 src 加入路径。
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rag_system.cli import main

if __name__ == "__main__":
    main()
