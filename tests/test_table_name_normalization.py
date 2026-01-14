"""Tests for table name normalization to fix PGVectorStore double-prefix bug."""

import pytest
from utils.naming import normalize_table_name_for_pgvector, get_actual_table_name


class TestNormalizeTableName:
    """Test table name normalization for PGVectorStore compatibility."""

    def test_strips_data_prefix(self):
        """Test that 'data_' prefix is stripped."""
        assert normalize_table_name_for_pgvector("data_messages_cs500") == "messages_cs500"
        assert normalize_table_name_for_pgvector("data_test_table") == "test_table"

    def test_keeps_non_prefixed_names(self):
        """Test that names without 'data_' prefix are unchanged."""
        assert normalize_table_name_for_pgvector("messages_cs500") == "messages_cs500"
        assert normalize_table_name_for_pgvector("test_table") == "test_table"

    def test_handles_multiple_data_prefixes(self):
        """Test handling of multiple 'data_' prefixes."""
        # Only strip the first one
        assert normalize_table_name_for_pgvector("data_data_table") == "data_table"
        assert normalize_table_name_for_pgvector("data_data_data_table") == "data_data_table"

    def test_data_in_middle_not_affected(self):
        """Test that 'data_' in middle of name is preserved."""
        assert normalize_table_name_for_pgvector("my_data_table") == "my_data_table"
        assert normalize_table_name_for_pgvector("table_data_v2") == "table_data_v2"

    def test_empty_name_raises_error(self):
        """Test that empty table name raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            normalize_table_name_for_pgvector("")

        with pytest.raises(ValueError, match="cannot be empty"):
            normalize_table_name_for_pgvector("   ")

    def test_whitespace_trimmed(self):
        """Test that whitespace is trimmed."""
        assert normalize_table_name_for_pgvector("  data_table  ") == "table"
        assert normalize_table_name_for_pgvector("  table  ") == "table"

    def test_case_sensitivity(self):
        """Test that function is case-sensitive (PostgreSQL convention)."""
        # Should NOT strip "DATA_" (uppercase)
        assert normalize_table_name_for_pgvector("DATA_table") == "DATA_table"
        assert normalize_table_name_for_pgvector("Data_table") == "Data_table"

    def test_hyphens_preserved(self):
        """Test that hyphens in table names are preserved."""
        assert normalize_table_name_for_pgvector("data_messages-text-slim") == "messages-text-slim"
        assert normalize_table_name_for_pgvector("my-table-name") == "my-table-name"

    def test_underscores_preserved(self):
        """Test that underscores are preserved correctly."""
        assert normalize_table_name_for_pgvector("data_my_table_name") == "my_table_name"
        assert normalize_table_name_for_pgvector("my_table_name") == "my_table_name"

    def test_complex_names(self):
        """Test complex real-world table names."""
        # Real example from the bug report
        assert normalize_table_name_for_pgvector("data_messages-text-slim_fast_1883_260108") == "messages-text-slim_fast_1883_260108"

        # Generated table names
        assert normalize_table_name_for_pgvector("data_report_cs700_ov150_bge_260108") == "report_cs700_ov150_bge_260108"


class TestGetActualTableName:
    """Test getting actual PostgreSQL table name after PGVectorStore processing."""

    def test_adds_prefix_when_missing(self):
        """Test that 'data_' prefix is added when missing."""
        assert get_actual_table_name("messages_cs500") == "data_messages_cs500"
        assert get_actual_table_name("test_table") == "data_test_table"

    def test_preserves_existing_prefix(self):
        """Test that existing 'data_' prefix is preserved."""
        assert get_actual_table_name("data_messages_cs500") == "data_messages_cs500"
        assert get_actual_table_name("data_test_table") == "data_test_table"

    def test_no_double_prefix(self):
        """Test that double prefix is never created."""
        # Even if called twice
        name1 = get_actual_table_name("messages_cs500")
        name2 = get_actual_table_name(name1)
        assert name1 == name2 == "data_messages_cs500"

    def test_idempotent_with_normalize(self):
        """Test that normalize → get_actual is idempotent."""
        original = "data_messages_cs500"
        normalized = normalize_table_name_for_pgvector(original)
        actual = get_actual_table_name(normalized)
        assert actual == original

    def test_with_hyphens(self):
        """Test handling of table names with hyphens."""
        assert get_actual_table_name("messages-text-slim") == "data_messages-text-slim"
        assert get_actual_table_name("data_messages-text-slim") == "data_messages-text-slim"


class TestRoundTripConversion:
    """Test round-trip conversion scenarios."""

    def test_user_provides_no_prefix(self):
        """User provides: 'messages_cs500' → PostgreSQL: 'data_messages_cs500'"""
        user_input = "messages_cs500"
        normalized = normalize_table_name_for_pgvector(user_input)
        assert normalized == "messages_cs500"

        actual = get_actual_table_name(normalized)
        assert actual == "data_messages_cs500"

    def test_user_provides_with_prefix(self):
        """User provides: 'data_messages_cs500' → PostgreSQL: 'data_messages_cs500'"""
        user_input = "data_messages_cs500"
        normalized = normalize_table_name_for_pgvector(user_input)
        assert normalized == "messages_cs500"

        actual = get_actual_table_name(normalized)
        assert actual == "data_messages_cs500"

    def test_double_prefix_prevented(self):
        """Verify double prefix is prevented."""
        # Without fix, this would create "data_data_messages_cs500"
        user_input = "data_messages_cs500"
        normalized = normalize_table_name_for_pgvector(user_input)
        actual = get_actual_table_name(normalized)

        # Should NOT be double-prefixed
        assert actual == "data_messages_cs500"
        assert actual.count("data_") == 1

    def test_real_world_bug_case(self):
        """Test the actual bug case from the report."""
        # The bug: user sees "data_messages-text-slim_fast_1883_260108" in list
        # System tries to query it, PGVectorStore creates "data_data_messages..."
        user_table = "data_messages-text-slim_fast_1883_260108"

        # Fix: normalize before passing to PGVectorStore
        normalized = normalize_table_name_for_pgvector(user_table)
        assert normalized == "messages-text-slim_fast_1883_260108"

        # PGVectorStore will add "data_" back, resulting in correct table name
        actual = get_actual_table_name(normalized)
        assert actual == "data_messages-text-slim_fast_1883_260108"

        # Verify no double prefix
        assert "data_data_" not in actual

    def test_multiple_conversions_stable(self):
        """Test that multiple conversions remain stable."""
        start = "my_table"

        # First round
        norm1 = normalize_table_name_for_pgvector(start)
        actual1 = get_actual_table_name(norm1)
        assert actual1 == "data_my_table"

        # Second round (simulating re-reading from database)
        norm2 = normalize_table_name_for_pgvector(actual1)
        actual2 = get_actual_table_name(norm2)
        assert actual2 == "data_my_table"

        # Should be stable
        assert actual1 == actual2


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_character_name(self):
        """Test single character table names."""
        assert normalize_table_name_for_pgvector("a") == "a"
        assert normalize_table_name_for_pgvector("data_a") == "a"
        assert get_actual_table_name("a") == "data_a"

    def test_exactly_five_characters(self):
        """Test 5-char names (edge case for 'data_' prefix length)."""
        assert normalize_table_name_for_pgvector("abcde") == "abcde"
        assert normalize_table_name_for_pgvector("data_") == ""  # Edge case
        # Note: Empty result would fail validation if used

    def test_only_data_prefix(self):
        """Test table name that is exactly 'data_'."""
        with pytest.raises(ValueError, match="cannot be empty"):
            # After stripping "data_", becomes empty
            name = normalize_table_name_for_pgvector("data_")

    def test_numeric_names(self):
        """Test table names with numbers."""
        assert normalize_table_name_for_pgvector("table123") == "table123"
        assert normalize_table_name_for_pgvector("data_123table") == "123table"

    def test_special_characters_after_prefix(self):
        """Test special characters after data_ prefix."""
        # These should be preserved (PostgreSQL allows them with quoting)
        assert normalize_table_name_for_pgvector("data_table-v1") == "table-v1"
        assert normalize_table_name_for_pgvector("data_table_v1") == "table_v1"

    def test_very_long_names(self):
        """Test very long table names."""
        long_name = "a" * 100
        assert normalize_table_name_for_pgvector(long_name) == long_name
        assert normalize_table_name_for_pgvector(f"data_{long_name}") == long_name


class TestIntegrationWithExistingFunctions:
    """Test integration with existing naming functions."""

    def test_with_generate_table_name(self):
        """Test that generated table names work with normalization."""
        from utils.naming import generate_table_name
        from pathlib import Path

        doc_path = Path("test.pdf")
        generated = generate_table_name(doc_path, 700, 150, "BAAI/bge-small-en")

        # Generated names should NOT start with "data_"
        assert not generated.startswith("data_")

        # Normalization should return unchanged
        normalized = normalize_table_name_for_pgvector(generated)
        assert normalized == generated

        # Get actual should add prefix
        actual = get_actual_table_name(normalized)
        assert actual == f"data_{generated}"

    def test_with_sanitize_table_name(self):
        """Test integration with sanitize_table_name."""
        from utils.naming import sanitize_table_name

        # Sanitized names should work with normalization
        sanitized = sanitize_table_name("my-document name")
        normalized = normalize_table_name_for_pgvector(sanitized)

        # Should be compatible
        assert "_" in normalized  # Has underscores from sanitization
        assert "-" not in normalized  # Hyphens removed by sanitize


class TestBugPrevention:
    """Tests specifically to prevent the reported bug from recurring."""

    def test_prevents_zero_results_bug(self):
        """Test that the specific bug scenario is prevented.

        Bug scenario:
        1. Table created: data_messages-text-slim_fast_1883_260108
        2. User selects it from list
        3. Without fix: PGVectorStore queries data_data_messages... (empty!)
        4. With fix: PGVectorStore queries data_messages... (has data)
        """
        # Table as it appears in database
        db_table = "data_messages-text-slim_fast_1883_260108"

        # User selects this from dropdown (web UI)
        selected_table = db_table

        # OLD CODE (bug): Pass directly to PGVectorStore
        # PGVectorStore.from_params(table_name=selected_table)
        # Would query: "data_data_messages-text-slim_fast_1883_260108" ❌

        # NEW CODE (fix): Normalize before passing to PGVectorStore
        normalized = normalize_table_name_for_pgvector(selected_table)
        # PGVectorStore.from_params(table_name=normalized)
        # Will query: "data_messages-text-slim_fast_1883_260108" ✅

        # Verify fix works
        actual = get_actual_table_name(normalized)
        assert actual == db_table
        assert "data_data_" not in actual

    def test_indexing_creates_correct_table(self):
        """Test that indexing creates correct table name.

        Scenario: User wants table "my_index"
        Expected: PostgreSQL table "data_my_index" (not "data_data_my_index")
        """
        user_desired_name = "my_index"

        # Normalize (no change since no prefix)
        normalized = normalize_table_name_for_pgvector(user_desired_name)
        assert normalized == "my_index"

        # PGVectorStore creates
        actual = get_actual_table_name(normalized)
        assert actual == "data_my_index"

        # If user later queries using the actual name
        # (e.g., from list_vector_tables())
        normalized_for_query = normalize_table_name_for_pgvector(actual)
        assert normalized_for_query == "my_index"

        # Should query the same table
        assert get_actual_table_name(normalized_for_query) == actual

    def test_migration_from_broken_tables(self):
        """Test handling of existing broken tables with double prefix."""
        # Some users may have created data_data_* tables
        broken_table = "data_data_messages_cs500"

        # Normalize strips first "data_"
        normalized = normalize_table_name_for_pgvector(broken_table)
        assert normalized == "data_messages_cs500"

        # If used for new table, would create
        actual = get_actual_table_name(normalized)
        assert actual == "data_data_messages_cs500"

        # This matches the broken table, allowing queries to work
        # (though ideally user should rename the table)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
