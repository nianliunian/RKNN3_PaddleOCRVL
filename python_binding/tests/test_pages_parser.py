"""Tests for parse_pages_param."""
import pytest

from pipeline.pages_parser import parse_pages_param, PageSpecError


class TestParsePagesParam:
    def test_none_returns_all_up_to_max(self):
        assert parse_pages_param(None, total_in_doc=20, max_pages=10) == list(range(10))

    def test_empty_string_returns_all_up_to_max(self):
        assert parse_pages_param("", total_in_doc=20, max_pages=10) == list(range(10))

    def test_single_page(self):
        assert parse_pages_param("3", total_in_doc=20, max_pages=10) == [2]

    def test_comma_separated_pages(self):
        assert parse_pages_param("3,5,7", total_in_doc=20, max_pages=10) == [2, 4, 6]

    def test_range(self):
        assert parse_pages_param("1-3", total_in_doc=20, max_pages=10) == [0, 1, 2]

    def test_mixed_range_and_single(self):
        assert parse_pages_param("1-3,5,8-10", total_in_doc=20, max_pages=10) == [0, 1, 2, 4, 7, 8, 9]

    def test_out_of_range_pages_dropped(self):
        assert parse_pages_param("1-5,30", total_in_doc=20, max_pages=10) == [0, 1, 2, 3, 4]

    def test_max_pages_truncates_selection(self):
        # 1-20 selected but max_pages=10 → first 10 of selection
        result = parse_pages_param("1-20", total_in_doc=50, max_pages=10)
        assert result == list(range(10))
        assert len(result) == 10

    def test_descending_range_raises(self):
        with pytest.raises(PageSpecError, match="descending"):
            parse_pages_param("5-2", total_in_doc=20, max_pages=10)

    def test_non_numeric_raises(self):
        with pytest.raises(PageSpecError):
            parse_pages_param("a-b", total_in_doc=20, max_pages=10)

    def test_zero_page_raises(self):
        with pytest.raises(PageSpecError, match="1-based"):
            parse_pages_param("0", total_in_doc=20, max_pages=10)

    def test_garbage_token_raises(self):
        with pytest.raises(PageSpecError):
            parse_pages_param("1-3,xyz,5", total_in_doc=20, max_pages=10)

    def test_all_pages_in_doc_when_under_max(self):
        # 5-page doc, no pages param → all 5
        assert parse_pages_param(None, total_in_doc=5, max_pages=10) == [0, 1, 2, 3, 4]

    def test_dedup_preserves_order(self):
        # 1,1,2 → [0,1] (dedup, first occurrence order)
        assert parse_pages_param("1,1,2", total_in_doc=20, max_pages=10) == [0, 1]
