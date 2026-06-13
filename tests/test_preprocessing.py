"""Tests for the text preprocessing pipeline."""

import pytest

from src.data.preprocessor import (
    basic_clean,
    normalize_slang,
    handle_negation,
    advanced_clean,
)


class TestBasicClean:
    def test_lowercase(self):
        assert basic_clean("Hello WORLD") == "hello world"

    def test_remove_mentions(self):
        assert basic_clean("hello @user world") == "hello world"

    def test_remove_hashtags(self):
        assert basic_clean("hello #tag world") == "hello world"

    def test_remove_urls(self):
        assert basic_clean("visit http://example.com now") == "visit now"
        assert basic_clean("visit www.example.com now") == "visit now"

    def test_remove_special_chars(self):
        assert basic_clean("hello! world?") == "hello world"

    def test_empty_input(self):
        assert basic_clean("") == ""

    def test_non_string_input(self):
        assert basic_clean(None) == ""
        assert basic_clean(123) == ""

    def test_extra_whitespace(self):
        assert basic_clean("hello   world") == "hello world"


class TestNormalizeSlang:
    def test_slang_replacement(self):
        assert normalize_slang("gak suka") == "tidak suka"
        assert normalize_slang("udah makan") == "sudah makan"
        assert normalize_slang("gw pergi") == "saya pergi"

    def test_no_slang(self):
        assert normalize_slang("saya makan") == "saya makan"

    def test_multiple_slang(self):
        result = normalize_slang("gw gak udah")
        assert result == "saya tidak sudah"

    def test_empty_string(self):
        assert normalize_slang("") == ""


class TestHandleNegation:
    def test_simple_negation(self):
        result = handle_negation("tidak bagus")
        assert "NOT_bagus" in result

    def test_slang_negation(self):
        result = handle_negation("gak enak")
        assert "NOT_enak" in result

    def test_no_negation(self):
        result = handle_negation("bagus sekali")
        assert "NOT_" not in result

    def test_multiple_negation(self):
        result = handle_negation("tidak bagus dan bukan murah")
        assert "NOT_bagus" in result
        assert "NOT_murah" in result

    def test_negation_scope_two_words(self):
        result = handle_negation("tidak sangat bagus")
        assert "NOT_sangat" in result
        assert "NOT_bagus" in result


class TestAdvancedClean:
    def test_full_pipeline_no_tools(self):
        result = advanced_clean("Gak Bagus BGT!", use_stemming=False, use_stopwords=False)
        assert "tidak" in result
        assert "bagus" in result

    def test_empty_input(self):
        assert advanced_clean("", use_stemming=False, use_stopwords=False) == ""

    def test_non_string(self):
        assert advanced_clean(None, use_stemming=False, use_stopwords=False) == ""

    def test_url_and_mention_removal(self):
        result = advanced_clean(
            "@user http://example.com barang bagus",
            use_stemming=False,
            use_stopwords=False,
        )
        assert "@" not in result
        assert "http" not in result

    def test_slang_normalization_in_pipeline(self):
        result = advanced_clean(
            "gw udah makan", use_stemming=False, use_stopwords=False
        )
        assert "saya" in result
        assert "sudah" in result
