# Install / locate the shared obligex package.
#
# All three notebooks import the SAME schema and prompts. Retyping the system
# prompt into a notebook is how you get a model that scores 0.9 in eval and 0.4
# in production, so this cell exists to make the import reliable rather than
# convenient.
#
# Code lives on local Colab disk, NOT Drive: imports off a Drive FUSE mount are
# slow, __pycache__ writes there are flaky, and nesting the clone inside the
# Drive workspace is what produces the "unknown location" namespace-package
# error. Drive holds data; /content holds code.
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/ihemanthc/obligex.git"
CLONE_DIR = Path("/content/obligex_repo")      # local disk, distinct name

if CLONE_DIR.exists():
    subprocess.run(["git", "-C", str(CLONE_DIR), "pull", "--quiet"], check=False)
    print(f"repo present at {CLONE_DIR} (pulled)")
else:
    r = subprocess.run(
        ["git", "clone", "--depth", "1", "--quiet", REPO_URL, str(CLONE_DIR)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise SystemExit(
            f"clone failed: {r.stderr.strip()}\n"
            f"If the repo is private, use a token URL or upload obligex/ to Drive "
            f"and set SEARCH_ROOTS below to its parent."
        )
    print(f"cloned to {CLONE_DIR}")

# Find the directory CONTAINING the package (the dir whose child is
# obligex/__init__.py). Do not assume the clone root is it — a repo cloned into
# a directory also called 'obligex' nests one level deeper, and putting the
# wrong level on sys.path yields a namespace package that imports with
# __file__ = None and raises "cannot import name ... (unknown location)".
SEARCH_ROOTS = [CLONE_DIR, Path("/content"), Path("/content/drive/MyDrive")]

pkg_parent = None
for root in SEARCH_ROOTS:
    if not root.exists():
        continue
    for init in sorted(root.glob("**/obligex/__init__.py")):
        if "__pycache__" in str(init) or ".git" in str(init).split("/"):
            continue
        pkg_parent = init.parent.parent
        break
    if pkg_parent:
        break

if pkg_parent is None:
    raise SystemExit(
        "Could not find obligex/__init__.py under: "
        + ", ".join(str(r) for r in SEARCH_ROOTS)
    )

print(f"package dir   {pkg_parent / 'obligex'}")
print(f"sys.path entry {pkg_parent}")

# Drop any stale/namespace 'obligex' already bound in this kernel. Without this
# a failed import stays cached and keeps failing after you fix the path, which
# looks exactly like the fix not working.
for name in [m for m in sys.modules if m == "obligex" or m.startswith("obligex.")]:
    del sys.modules[name]

sys.path.insert(0, str(pkg_parent))

from obligex import (  # noqa: E402
    SYSTEM_PROMPT,
    TEACHER_SYSTEM_PROMPT,
    DocumentType,
    ObligationType,
    WindowExtraction,
    build_user_prompt,
    dumps_target,
    ground_obligations,
    guided_schema,
    parse_model_output,
    windows_from_pages,
)
import obligex  # noqa: E402

# A real package has a __file__; a namespace package does not. Assert it, so the
# failure is one clear line instead of a confusing ImportError further down.
assert obligex.__file__ is not None, (
    f"'obligex' resolved to a NAMESPACE package at {list(obligex.__path__)} — "
    "that directory has no __init__.py. sys.path is pointing one level too high; "
    "it must point at the PARENT of the package directory."
)

print(f"\nobligex {obligex.__version__} loaded from {obligex.__file__}")
print(f"  student prompt {len(SYSTEM_PROMPT) // 4:>5} tokens  (train + serve)")
print(f"  teacher prompt {len(TEACHER_SYSTEM_PROMPT) // 4:>5} tokens  (labelling only)")
