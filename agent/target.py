from pathlib import Path

# The hardcoded target for manual runs: a local fixture with a known bug. Its code runs in the sandbox like any repo.
REPO = str(Path(__file__).resolve().parents[1] / "examples" / "pagination_repo")

ISSUE = """Pagination skips the first page and drops the last partial page

With 25 orders and 10 per page, `paginate(orders, 1, 10)` returns orders 11-20 instead of 1-10, and page 3 comes back empty instead of holding orders 21-25. `total_pages(25, 10)` also returns 2 when there are clearly 3 pages.

Expected: pages are 1-indexed, page 1 starts at the first item, and total_pages counts a final partial page."""
