import pytest

from shop.pagination import paginate, total_pages

ORDERS = list(range(1, 26))


def test_first_page_starts_at_first_item():
    assert paginate(ORDERS, 1, 10) == list(range(1, 11))


def test_last_page_is_partial():
    assert paginate(ORDERS, 3, 10) == [21, 22, 23, 24, 25]


def test_page_past_end_is_empty():
    assert paginate(ORDERS, 4, 10) == []


def test_rejects_page_zero():
    with pytest.raises(ValueError):
        paginate(ORDERS, 0, 10)


def test_total_pages_rounds_up():
    assert total_pages(25, 10) == 3


def test_total_pages_exact_multiple():
    assert total_pages(20, 10) == 2


def test_total_pages_empty():
    assert total_pages(0, 10) == 0


def test_total_pages_rejects_zero_per_page():
    with pytest.raises(ValueError):
        total_pages(25, 0)
