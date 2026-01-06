#!/usr/bin/env python3
"""
Clean Facebook Messenger HTML export data for RAG indexing.

This script:
1. Recursively finds all message HTML files in the messenger export
2. Extracts messages with sender, content, timestamp, and thread info
3. Cleans HTML entities and tags
4. Outputs clean text files organized by conversation thread

Usage:
    python scripts/clean_messenger_html.py

    # Or with custom paths:
    python scripts/clean_messenger_html.py --input data/messenger --output data/messenger_clean

Output format:
    data/messenger_clean/
        inbox/
            thread_name/
                conversation.txt  # All messages in chronological order
        filtered_threads/
            thread_name/
                conversation.txt
        ... (same structure as input)
"""

import argparse
import html
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
from datetime import datetime


def clean_html_content(html_content: str) -> str:
    """Remove HTML tags and decode entities."""
    # Use BeautifulSoup to extract text
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text(separator=' ', strip=True)

    # Decode HTML entities
    text = html.unescape(text)

    # Clean up excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def parse_timestamp(timestamp_str: str) -> Optional[datetime]:
    """Parse Facebook timestamp formats."""
    # Try different formats
    formats = [
        "%b %d, %Y %I:%M:%S %p",  # Jun 20, 2016 5:09:52 pm
        "%d %B %Y, %H:%M UTC%z",  # 17 décembre 2025, 08:40 UTC+01:00
    ]

    for fmt in formats:
        try:
            return datetime.strptime(timestamp_str, fmt)
        except ValueError:
            continue

    # If parsing fails, return None
    return None


def extract_messages_from_html(html_path: Path) -> List[Dict[str, str]]:
    """Extract all messages from a single HTML file.

    Returns:
        List of message dictionaries with keys:
        - sender: Name of the sender (or None for replies)
        - content: Message text content
        - timestamp: Raw timestamp string
        - datetime: Parsed datetime object (if available)
    """
    with open(html_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    # Extract thread title
    thread_title_elem = soup.find('h1')
    thread_title = clean_html_content(str(thread_title_elem)) if thread_title_elem else "Unknown Thread"

    messages = []

    # Find all message sections
    sections = soup.find_all('section', class_='_a6-g')

    for section in sections:
        # Extract sender (if present - indicated by header with sender name)
        sender_elem = section.find('h2', class_='_a6-h')
        sender = clean_html_content(str(sender_elem)) if sender_elem else None

        # Extract message content
        content_elem = section.find('div', class_='_a6-p')
        if not content_elem:
            continue

        # Get text content, handling links and media
        content_parts = []
        for elem in content_elem.descendants:
            if isinstance(elem, str):
                text = elem.strip()
                if text:
                    content_parts.append(text)
            elif elem.name == 'a':
                # Include link text and URL
                link_text = clean_html_content(str(elem))
                if link_text:
                    content_parts.append(link_text)

        content = ' '.join(content_parts)
        content = re.sub(r'\s+', ' ', content).strip()

        if not content:
            continue

        # Extract timestamp
        footer_elem = section.find('footer', class_='_a6-o')
        timestamp_str = None
        timestamp_obj = None

        if footer_elem:
            time_elem = footer_elem.find('time')
            if time_elem:
                # Try to get datetime attribute
                timestamp_str = time_elem.get('datetime', time_elem.get_text(strip=True))
            else:
                # Fallback to footer text
                timestamp_str = clean_html_content(str(footer_elem))

            # Try to parse timestamp
            if timestamp_str:
                timestamp_obj = parse_timestamp(timestamp_str)

        messages.append({
            'sender': sender,
            'content': content,
            'timestamp': timestamp_str,
            'datetime': timestamp_obj,
        })

    return messages, thread_title


def format_conversation(messages: List[Dict[str, str]], thread_title: str) -> str:
    """Format messages into readable conversation text."""
    lines = []
    lines.append(f"# {thread_title}")
    lines.append(f"# Total messages: {len(messages)}")
    lines.append("")

    # Sort by datetime if available, otherwise keep original order
    messages_with_datetime = [m for m in messages if m['datetime']]
    messages_without_datetime = [m for m in messages if not m['datetime']]

    sorted_messages = sorted(messages_with_datetime, key=lambda x: x['datetime'])
    sorted_messages.extend(messages_without_datetime)

    current_sender = None

    for msg in sorted_messages:
        sender = msg['sender'] or "(reply)"
        timestamp = msg['timestamp'] or "unknown time"
        content = msg['content']

        # Add blank line between different senders
        if current_sender and current_sender != sender:
            lines.append("")

        # Format: [Timestamp] Sender: Message
        lines.append(f"[{timestamp}] {sender}: {content}")
        current_sender = sender

    return '\n'.join(lines)


def process_thread_folder(thread_path: Path, output_path: Path, stats: Dict):
    """Process all HTML files in a thread folder."""
    # Find all message HTML files
    html_files = sorted(thread_path.glob("message_*.html"))

    if not html_files:
        return

    all_messages = []
    thread_title = "Unknown Thread"

    # Process each HTML file
    for html_file in html_files:
        try:
            messages, title = extract_messages_from_html(html_file)
            all_messages.extend(messages)
            if title and title != "Unknown Thread":
                thread_title = title
            stats['files_processed'] += 1
            stats['messages_extracted'] += len(messages)
        except Exception as e:
            print(f"Error processing {html_file}: {e}")
            stats['errors'] += 1

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Write conversation to file
    output_file = output_path / "conversation.txt"
    conversation_text = format_conversation(all_messages, thread_title)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(conversation_text)

    stats['threads_processed'] += 1

    # Also save as JSON for potential other uses
    json_file = output_path / "messages.json"
    json_data = {
        'thread_title': thread_title,
        'message_count': len(all_messages),
        'messages': [
            {
                'sender': m['sender'],
                'content': m['content'],
                'timestamp': m['timestamp'],
            }
            for m in all_messages
        ]
    }

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)


def process_messenger_export(input_dir: Path, output_dir: Path):
    """Process entire messenger export directory."""
    print(f"Processing Messenger export from: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()

    stats = {
        'threads_processed': 0,
        'files_processed': 0,
        'messages_extracted': 0,
        'errors': 0,
    }

    # Process each category (inbox, filtered_threads, archived_threads, etc.)
    categories = ['inbox', 'filtered_threads', 'archived_threads', 'message_requests', 'e2ee_cutover']

    for category in categories:
        category_path = input_dir / category

        if not category_path.exists():
            continue

        print(f"Processing {category}...")

        # Find all thread folders
        thread_folders = [d for d in category_path.iterdir() if d.is_dir()]

        for i, thread_folder in enumerate(thread_folders, 1):
            if i % 50 == 0:
                print(f"  Processed {i}/{len(thread_folders)} threads...")

            # Create corresponding output path
            output_thread_path = output_dir / category / thread_folder.name

            process_thread_folder(thread_folder, output_thread_path, stats)

        print(f"  Completed {len(thread_folders)} threads in {category}")
        print()

    # Print summary
    print("=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Threads processed:     {stats['threads_processed']:,}")
    print(f"HTML files processed:  {stats['files_processed']:,}")
    print(f"Messages extracted:    {stats['messages_extracted']:,}")
    print(f"Errors:                {stats['errors']:,}")
    print()
    print(f"Output saved to: {output_dir}")
    print()
    print("You can now index this data with:")
    print(f"  export PDF_PATH={output_dir}")
    print(f"  export PGTABLE=messenger_chat_history")
    print(f"  export RESET_TABLE=1")
    print(f"  python rag_low_level_m1_16gb_verbose.py")


def main():
    parser = argparse.ArgumentParser(
        description="Clean Facebook Messenger HTML export for RAG indexing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--input',
        type=Path,
        default=Path('data/messenger'),
        help='Input directory containing messenger export (default: data/messenger)',
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/messenger_clean'),
        help='Output directory for cleaned data (default: data/messenger_clean)',
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.input.exists():
        print(f"Error: Input directory does not exist: {args.input}")
        return 1

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Process the export
    process_messenger_export(args.input, args.output)

    return 0


if __name__ == '__main__':
    exit(main())
