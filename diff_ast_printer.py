#!/usr/bin/env python3
"""
diff_ast_printer.py

Read a unified diff from stdin (end with EOF), extract added C/C++ code lines,
parse them with libclang, and print the resulting AST.
"""
import sys
import platform
import tempfile
import os
import subprocess

try:
    import clang.cindex as cidx
except ImportError:
    sys.exit("Error: clang.cindex module not found. Please install clang bindings (e.g., pip install clang).")

# ----------------------------------------------------------------------------
# Configure libclang library path
# - Use LIBCLANG_PATH or LIBCLANG_LIBRARY_FILE env var if set
# - On Windows, default to Program Files path
# - On Unix, try llvm-config --libdir
# - Fallback to common paths and known build location
# ----------------------------------------------------------------------------
env_path = os.getenv("LIBCLANG_PATH") or os.getenv("LIBCLANG_LIBRARY_FILE")
if env_path:
    clang_lib = env_path
else:
    if platform.system() == "Windows":
        clang_lib = r"C:\Program Files\LLVM\bin\libclang.dll"
    else:
        # Try llvm-config
        try:
            libdir = subprocess.check_output(["llvm-config", "--libdir"], text=True).strip()
            default = os.path.join(libdir, "libclang.so")
        except Exception:
            default = "/usr/lib/llvm-10/lib/libclang.so"
        # Known custom build path
        custom = "/ptmp2/nshashwa/llvm-project/build/lib/libclang.so"
        # Choose whichever exists
        if os.path.exists(custom):
            clang_lib = custom
        else:
            clang_lib = default

# Attempt to load libclang
try:
    cidx.Config.set_library_file(clang_lib)
    print(f"INFO: Loaded libclang from {clang_lib}")
except Exception as e:
    sys.exit(
        f"Error: could not load libclang from {clang_lib} ({e}).\n"
        "Please set the LIBCLANG_PATH environment variable to the correct libclang location."
    )


def parse_cpp_ast(code: str):
    """
    Parse the given C/C++ code and print a simple indented AST.
    """
    with tempfile.NamedTemporaryFile(suffix=".cpp", delete=False) as tmp:
        tmp.write(code.encode())
        tmp_path = tmp.name
    try:
        index = cidx.Index.create()
        tu = index.parse(tmp_path, args=["-std=c++17"])

        def visit(node, depth=0):
            print("  " * depth + f"{node.kind.name}: {node.spelling or ''}")
            for child in node.get_children():
                visit(child, depth + 1)

        visit(tu.cursor)
    finally:
        os.remove(tmp_path)


def extract_added_code(diff_text: str) -> str:
    """
    Extract lines starting with '+' (excluding file headers) from the diff.
    """
    lines = []
    for line in diff_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            lines.append(line[1:])
    return "\n".join(lines)


def read_diff() -> str:
    """
    Read multiline diff from stdin until a line with only 'EOF'.
    """
    print("Paste diff (end with EOF on its own line):")
    buf = []
    for line in sys.stdin:
        if line.strip() == "EOF":
            break
        buf.append(line.rstrip("\n"))
    return "\n".join(buf)


def main():
    diff = read_diff()
    added = extract_added_code(diff)
    if not added.strip():
        print("No added code lines found in diff.")
        return
    print("Generated AST:")
    parse_cpp_ast(added)


if __name__ == "__main__":
    main()
