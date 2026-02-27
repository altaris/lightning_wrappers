# AI Agent Instructions

## General Coding Philosophy

- **Language**: English for chat, code, comments, and documentation
- **Be direct and concise**: Provide short context without verbose
  encouragements
- **Modify only what's relevant**: Only change code directly related to the
  task, even if you see other potential improvements
- **Readable over clever**: Prefer clear, understandable code to ultra-concise
  code
- **Understand the task**: Distinguish when prompts call for code writing vs.
  just explanation
- **Edit workspace directly**: When implementing changes, edit files directly in
  the workspace using tools rather than writing code blocks in chat. Don't
  rewrite entire files in chat — apply edits in place so changes appear in the
  editor
- **Address the user**: Refer to the user as "お兄さん" or just "兄さん"

## Code Conventions

### General Python Style

- **Python version**: Follow project's Python version requirements (check
  pyproject.toml)
- **Line length**: Follow project's line length configuration (typically 79-120
  characters, check pyproject.toml or ruff config)
- **Naming**: `snake_case` for all variables (lowercase only). Avoid ambiguous
  single-char names like `l`. Short names OK if meaning is clear.
- **Symbol organization**: Group top-level symbols by type (constants, classes,
  private methods, public methods), sort each group alphabetically. No separator
  comments like `# ----- CLASSES`.
- **Imports**: Standard aliases:
  - `numpy` → `np`
  - `pathlib.Path` → `Path`
  - `from loguru import logger as logging` (never stdlib logging)
  - `from PIL import Image`
  - `networkx` → `nx`
  - `from tqdm.auto import tqdm`
  - `torch.nn` → `nn`, `torch.nn.functional` → `nnf`
  - `import torchvision.transforms.functional as tr`
  - `seaborn` → `sns`
  - `lightning` → `pl` (PyTorch Lightning)
  - Bokeh: `bokeh.plotting` → `bk`, `bokeh.models` → `bkm`, `bokeh.layouts` →
    `bkl`, `bokeh.palettes` → `bkp`
- **Relative imports**: Inside packages, always use relative imports for
  intra-package references: `from .module import symbol` (not absolute imports)
- **Always use `pathlib.Path`** for all filesystem paths and I/O
- **File I/O**: Always specify explicit encoding and mode:
  ```python
  open(path, mode="rt", encoding="utf8")  # reading
  open(path, mode="wt", encoding="utf8")  # writing
  ```
- **Open files**: When opening a file with a `with` statement, set the context
  manager's name to `fp` when possible:
  `with open(path, mode="rt", encoding="utf8") as fp:`
- **Same-line variable assignment**: In simple case, and when it doesn't impact
  readability, assign multiple variables on the same line: `x, y = point`,
  `width, height = image.size`
- **Progress bars**: Use `tqdm` with descriptive `desc`, update counts via
  `set_postfix({"n_samples": n})` (don't print)
- **Data containers**: Prefer `dataclasses.dataclass` for small, immutable-like
  containers

### Type Annotations

- Use builtin generics: `list[str]`, `dict[str, int] | None` (not
  `Optional[List[...]]`)
- Annotate all function arguments and return values
- Use `typing.Literal` for mode switches, `TypeVar` for generics
- Internal variables don't need annotations unless necessary
- **Assertions**: If required by typechecking, use
  `assert isinstance(x, expected_type)` to help type checkers understand types,
  especially when dealing with type narrowing
  ```python
    def process(data: list | None):
        if data is None:
            data = []
        assert isinstance(data, dict)  # *Only* if typechecker complains
        # Now the type checker knows `data` is a dict in this block
        ...
  ```

### Loguru Logging Patterns

- **Import**: Always `from loguru import logger as logging` — never use the
  standard `logging`
- **Formatting**: Use brace-style with argument binding (not f-strings):
  ```python
  logging.info("Processed {} items in {}", count, elapsed)  # ✅
  logging.info(f"Processed {count} items")  # ❌
  ```
- **Levels**:
  - `info` for normal progress
  - `success` for task completion
  - `warning` for partial failures
  - `error` for hard stops
- **CLI wrapping**: Use `@logging.catch` on top-level CLI entrypoints

### CLI Patterns (Click)

- Paths: `click.Path(..., path_type=Path)` with `exists=` flags
- Boolean flags: `is_flag=True`
- Always set `show_default=True` and provide helpful `help` text
- **Click integration**: Pass `path_type=Path` to `click.Path(...)`

### Error Handling

- **Exceptions**: Use specific exception types, not bare `except:`
- **CLI errors**: In CLI commands, prefer logging error + early return over
  raising for expected failure modes (e.g., file exists, validation fails)
- **Library code**: Raise exceptions with clear messages; let callers decide how
  to handle
- **Context managers**: Use for resource cleanup (files, connections, etc.)
- **Validation**: Fail fast with clear error messages at function entry

### Documentation & Comments

- **Docstrings**:
  - Google style with multiline text starting on newline after `"""`
  - If the docstring is a single line, keep it on the same line as the triple
    quotes: `"""Single line docstring."""`
  - Refer to
    https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
    for reference
  - Omit `Returns:` section if return type is self-explanatory
  - Include `Returns:` for complex tuple unpacking or non-obvious semantics
  - In `Args:` section, add type info only if it clarifies beyond the annotation
    (shapes, ranges, semantics)
  - Use LaTeX for math
  - Backticks for inline code (variable/function names, literals)
  - Can use emojis/unicode sparingly
  - Write concise module-level docstrings
  - Write usage examples for complex utility functions where helpful; do this
    sparringly
- **When to comment**:
  - Complex algorithms or non-obvious logic
  - Why something is done a certain way (not what is being done)
  - Temporary workarounds or TODOs with context
- **When NOT to comment**:
  - Obvious code that explains itself
  - Redundant restatements of the code
  - Outdated comments (remove or update)
- **Style**: Complete sentences with proper capitalization and punctuation

### Documentation

- **README.md**: Should include purpose, installation, usage examples, and
  contribution guidelines
- **API documentation**: Document public interfaces; private functions can have
  minimal docs
- **Examples**: Provide runnable examples for user-facing code and for complex
  functions (internal or external)
- **Architecture docs**: Document high-level design decisions and data flow
- **Tensor/array documentation**: Include shapes in backticks `(B, C, H, W)` in
  docstrings; name dtypes and device when relevant

## Common Pitfalls

- **Mutable defaults**: Never use mutable defaults in function signatures
  (`def func(items=[]):` ❌ → `def func(items=None):` ✅)
- **String concatenation**: Use f-strings or `.join()` instead of `+` in loops
- **Resource leaks**: Always close files/connections; use context managers
- **Catching broad exceptions**: Avoid `except Exception:` unless re-raising or
  logging or passing
- **Global state**: Minimize global variables; pass dependencies explicitly
- **Hardcoded paths**: Use `Path(__file__).parent` or config files for paths
- **Platform assumptions**: Use `os.path` or `pathlib` for cross-platform
  compatibility
- **Encoding issues**: Always specify `encoding="utf8"` when opening text files
- **Premature optimization**: Profile before optimizing; readability first

## Code Quality

### Makefile Targets (Recommended)

- `make format`: Format code with Ruff (import sorting + formatting)
- `make typecheck`: Type check with mypy
- `make lint-fix`: Auto-fix linting issues
- `make lint`: Check linting without fixes
- `make`: Run all of the above
- `make shellcheck`: Check shell scripts

### Direct Tool Usage

- Minor fixes (import sorting): `uvx ruff check --select I --fix FILE`
- Formatting: `uvx ruff format FILE`
- Type checking: `uv run mypy FILE`
- Linting: `uvx ruff check FILE`
- Shell scripts: `shellcheck FILE.sh`
