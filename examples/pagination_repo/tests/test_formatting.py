from shop.formatting import format_price


def test_formats_dollars_and_cents():
    assert format_price(1250) == "$12.50"


def test_formats_thousands():
    assert format_price(123456) == "$1,234.56"


def test_formats_negative():
    assert format_price(-5) == "-$0.05"
