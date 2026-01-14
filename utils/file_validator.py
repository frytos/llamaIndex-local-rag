"""
File Validation Utilities

Validates file types based on magic bytes (file headers) to detect
misnamed files (e.g., JPEG files with .pdf extension).
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

log = logging.getLogger(__name__)


# Magic bytes for common file types
FILE_SIGNATURES = {
    "pdf": [
        b"%PDF-",  # PDF header
    ],
    "jpeg": [
        b"\xFF\xD8\xFF\xE0",  # JPEG/JFIF
        b"\xFF\xD8\xFF\xE1",  # JPEG/EXIF
        b"\xFF\xD8\xFF\xE2",  # JPEG/Canon
        b"\xFF\xD8\xFF\xE8",  # JPEG/SPIFF
    ],
    "png": [
        b"\x89PNG\r\n\x1a\n",  # PNG signature
    ],
    "gif": [
        b"GIF87a",  # GIF87a
        b"GIF89a",  # GIF89a
    ],
    "zip": [
        b"PK\x03\x04",  # ZIP/DOCX/PPTX (all are ZIP-based)
        b"PK\x05\x06",  # Empty ZIP
        b"PK\x07\x08",  # Spanned ZIP
    ],
}


def read_file_header(file_path: Path, num_bytes: int = 16) -> bytes:
    """
    Read the first N bytes of a file.

    Args:
        file_path: Path to file
        num_bytes: Number of bytes to read (default: 16)

    Returns:
        First N bytes of the file
    """
    try:
        with open(file_path, "rb") as f:
            return f.read(num_bytes)
    except Exception as e:
        log.warning(f"Could not read file header for {file_path}: {e}")
        return b""


def detect_file_type(file_path: Path) -> Optional[str]:
    """
    Detect actual file type based on magic bytes.

    Args:
        file_path: Path to file

    Returns:
        Detected file type (e.g., "pdf", "jpeg", "png") or None if unknown
    """
    header = read_file_header(file_path, num_bytes=16)

    if not header:
        return None

    # Check each known file signature
    for file_type, signatures in FILE_SIGNATURES.items():
        for sig in signatures:
            if header.startswith(sig):
                return file_type

    return None


def validate_file_extension(file_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Validate that file extension matches actual file type.

    Args:
        file_path: Path to file

    Returns:
        Tuple of (is_valid, detected_type)
        - is_valid: True if extension matches detected type
        - detected_type: Actual detected file type
    """
    ext = file_path.suffix.lower().lstrip(".")
    detected = detect_file_type(file_path)

    if detected is None:
        # Unknown file type - can't validate
        return True, None

    # Map ZIP-based formats
    if detected == "zip" and ext in ("docx", "pptx", "xlsx"):
        return True, "zip"

    # Direct match
    if ext == detected or (ext == "jpg" and detected == "jpeg"):
        return True, detected

    # Mismatch
    return False, detected


def scan_directory_for_misnamed_files(directory: Path, extensions: list = None) -> dict:
    """
    Scan directory for files with incorrect extensions.

    Args:
        directory: Directory to scan
        extensions: List of extensions to check (default: ["pdf", "jpg", "jpeg", "png"])

    Returns:
        Dictionary mapping file paths to (claimed_ext, actual_type) tuples
    """
    if extensions is None:
        extensions = ["pdf", "jpg", "jpeg", "png", "docx", "pptx"]

    misnamed = {}

    for file_path in directory.rglob("*"):
        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower().lstrip(".")
        if ext not in extensions:
            continue

        is_valid, detected = validate_file_extension(file_path)
        if not is_valid and detected:
            misnamed[str(file_path)] = (ext, detected)
            log.warning(
                f"  ⚠️  Misnamed file: {file_path.name} "
                f"(claims to be .{ext} but is actually {detected})"
            )

    return misnamed


def create_validation_report(directory: Path, output_file: Optional[Path] = None) -> dict:
    """
    Create a detailed validation report for a directory.

    Args:
        directory: Directory to scan
        output_file: Optional path to write report

    Returns:
        Report dictionary with statistics
    """
    misnamed = scan_directory_for_misnamed_files(directory)

    report = {
        "directory": str(directory),
        "total_misnamed": len(misnamed),
        "files": misnamed,
        "summary_by_type": {},
    }

    # Count by claimed type
    for file_path, (claimed, actual) in misnamed.items():
        key = f"{claimed}_as_{actual}"
        report["summary_by_type"][key] = report["summary_by_type"].get(key, 0) + 1

    if output_file:
        import json
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
        log.info(f"Validation report written to: {output_file}")

    return report


if __name__ == "__main__":
    """
    Command-line interface for file validation.

    Usage:
        python utils/file_validator.py <directory>
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python utils/file_validator.py <directory>")
        sys.exit(1)

    directory = Path(sys.argv[1])
    if not directory.exists():
        print(f"Error: Directory not found: {directory}")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print(f"\n🔍 Scanning directory for misnamed files: {directory}\n")

    report = create_validation_report(directory)

    print(f"\n📊 Validation Summary")
    print(f"{'=' * 60}")
    print(f"Total misnamed files: {report['total_misnamed']}")

    if report["summary_by_type"]:
        print(f"\nBreakdown by type:")
        for type_pair, count in report["summary_by_type"].items():
            print(f"  • {type_pair}: {count} file(s)")

    if report["total_misnamed"] == 0:
        print("\n✅ No misnamed files detected!")
    else:
        print(f"\n⚠️  Found {report['total_misnamed']} misnamed file(s)")
        print(f"\nConsider renaming these files to prevent loading errors.")
