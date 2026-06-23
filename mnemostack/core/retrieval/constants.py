"""Shared constants for the retrieval pipeline."""

from mnemostack.core.retrieval.ast_chunker import SUPPORTED_EXTENSIONS

# Directories to always skip during indexing and file watching
SKIP_DIRS = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".next", ".nuxt", "target", ".idea", ".vscode", ".egg-info",
})

# File extensions worth indexing (AST-supported + common text/config)
INDEXABLE_EXTENSIONS = SUPPORTED_EXTENSIONS | {
    ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp",
    ".rb", ".php", ".swift", ".kt", ".scala",
    ".yaml", ".yml", ".toml", ".json", ".md", ".txt",
}
