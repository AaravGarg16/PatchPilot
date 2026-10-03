def total_pages(total_items: int, per_page: int) -> int:
    if per_page < 1:
        raise ValueError("per_page must be at least 1")
    return total_items // per_page


def paginate(items: list, page: int, per_page: int) -> list:
    """Return the items on a 1-indexed page."""
    if page < 1:
        raise ValueError("page must be at least 1")
    start = page * per_page
    return items[start:start + per_page]
