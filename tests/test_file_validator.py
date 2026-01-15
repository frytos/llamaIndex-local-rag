"""Tests for file validation utilities.

This module tests the file_validator module, including:
- Magic byte detection for different file types
- File extension validation
- Directory scanning for misnamed files
- Validation report generation

Week 1 - Day 2: File validator basic tests (10 tests)
"""

import pytest
from pathlib import Path
from utils.file_validator import (
    read_file_header,
    detect_file_type,
    validate_file_extension,
    scan_directory_for_misnamed_files,
    create_validation_report,
)


# ============================================================================
# Day 2: Magic Byte Detection Tests (5 tests)
# ============================================================================


class TestMagicByteDetection:
    """Test file type detection via magic bytes.

    These tests validate that the file validator correctly:
    - Detects PDF files from magic bytes
    - Detects different JPEG variants
    - Detects PNG files
    - Returns None for unknown types
    - Handles empty files gracefully
    """

    @pytest.mark.unit
    def test_detect_pdf(self, sample_files):
        """Test PDF detection from magic bytes.

        Given: A file starting with %PDF- magic bytes
        When: detect_file_type() is called
        Then: Returns "pdf"
        """
        pdf_file = sample_files['pdf_correct']

        detected = detect_file_type(pdf_file)

        assert detected == "pdf"

    @pytest.mark.unit
    def test_detect_jpeg_variants(self, sample_files):
        """Test detection of different JPEG magic byte variants.

        Given: A file with JPEG JFIF magic bytes (xFF xD8 xFF xE0)
        When: detect_file_type() is called
        Then: Returns "jpeg"
        """
        jpeg_file = sample_files['jpeg_as_pdf']  # This has JPEG bytes

        detected = detect_file_type(jpeg_file)

        assert detected == "jpeg"

    @pytest.mark.unit
    def test_detect_png(self, sample_files):
        """Test PNG detection from magic bytes.

        Given: A file with PNG signature (x89 PNG \r \n x1a \n)
        When: detect_file_type() is called
        Then: Returns "png"
        """
        png_file = sample_files['png_correct']

        detected = detect_file_type(png_file)

        assert detected == "png"

    @pytest.mark.unit
    def test_detect_unknown_type(self, sample_files):
        """Test detection returns None for unknown file types.

        Given: A file with unrecognized magic bytes
        When: detect_file_type() is called
        Then: Returns None
        """
        unknown_file = sample_files['unknown']

        detected = detect_file_type(unknown_file)

        assert detected is None

    @pytest.mark.unit
    def test_detect_empty_file(self, sample_files):
        """Test handling of empty files.

        Given: An empty file (0 bytes)
        When: detect_file_type() is called
        Then: Returns None without crashing
        """
        empty_file = sample_files['empty']

        detected = detect_file_type(empty_file)

        assert detected is None


# ============================================================================
# Day 2: File Validation Tests (3 tests)
# ============================================================================


class TestFileValidation:
    """Test extension vs content validation.

    These tests validate that:
    - Correctly named files pass validation
    - Misnamed files are detected
    - ZIP-based formats (DOCX, PPTX) are handled correctly
    """

    @pytest.mark.unit
    def test_validate_correct_pdf(self, sample_files):
        """Test validation passes for correctly named PDF.

        Given: A PDF file with .pdf extension
        When: validate_file_extension() is called
        Then: Returns (True, "pdf")
        """
        pdf_file = sample_files['pdf_correct']

        is_valid, detected = validate_file_extension(pdf_file)

        assert is_valid is True
        assert detected == "pdf"

    @pytest.mark.unit
    def test_validate_misnamed_jpeg_as_pdf(self, sample_files):
        """Test detection of JPEG file with .pdf extension.

        Given: A JPEG file named with .pdf extension
        When: validate_file_extension() is called
        Then: Returns (False, "jpeg") indicating mismatch
        """
        misnamed_file = sample_files['jpeg_as_pdf']

        is_valid, detected = validate_file_extension(misnamed_file)

        assert is_valid is False
        assert detected == "jpeg"

    @pytest.mark.unit
    def test_validate_docx_as_zip(self, sample_files):
        """Test DOCX file validates correctly (DOCX is ZIP-based).

        Given: A DOCX file (which has ZIP magic bytes)
        When: validate_file_extension() is called
        Then: Returns (True, "zip") because DOCX is a ZIP container
        """
        docx_file = sample_files['docx_correct']

        is_valid, detected = validate_file_extension(docx_file)

        assert is_valid is True
        assert detected == "zip"


# ============================================================================
# Day 2: Directory Scanning Tests (2 tests)
# ============================================================================


class TestDirectoryScanning:
    """Test bulk directory scanning and reporting.

    These tests validate:
    - Directory scanning finds all misnamed files
    - Recursive scanning works in subdirectories
    """

    @pytest.mark.unit
    def test_scan_directory_finds_misnamed(self, test_directory_with_misnamed_files):
        """Test directory scanning detects all misnamed files.

        Given: A directory with 3 misnamed files
        When: scan_directory_for_misnamed_files() is called
        Then: Returns dict with 3 entries
        """
        test_dir, expected_count = test_directory_with_misnamed_files

        misnamed = scan_directory_for_misnamed_files(test_dir)

        assert len(misnamed) == expected_count
        # Verify misnamed files are tracked
        assert any("fake_pdf.pdf" in path for path in misnamed.keys())
        assert any("fake_image.png" in path for path in misnamed.keys())
        assert any("wrong.jpg" in path for path in misnamed.keys())

    @pytest.mark.unit
    def test_scan_directory_recursive(self, test_directory_with_misnamed_files):
        """Test recursive scanning finds files in subdirectories.

        Given: A directory with misnamed files in subdirectories
        When: scan_directory_for_misnamed_files() is called
        Then: Finds files in both root and subdirectories
        """
        test_dir, expected_count = test_directory_with_misnamed_files

        misnamed = scan_directory_for_misnamed_files(test_dir)

        # Check that subdirectory file is found
        assert any("subfolder" in path for path in misnamed.keys())
        assert any("wrong.jpg" in path for path in misnamed.keys())


# ============================================================================
# Test Discovery and Execution Notes
# ============================================================================
"""
Run Day 2 tests:
    pytest tests/test_file_validator.py -v

Run with coverage:
    pytest tests/test_file_validator.py \
        --cov=utils.file_validator \
        --cov-report=term-missing

Expected results:
    - 10 tests passing
    - Coverage increase: +60-70% for utils/file_validator.py
    - Validates file type detection and prevents data corruption
"""
