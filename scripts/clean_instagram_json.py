#!/usr/bin/env python3
"""
Clean Instagram/Messenger JSON exports for RAG indexing.

Extracts only message content, removes metadata, decodes unicode properly.
"""

import json
import re
from pathlib import Path
from typing import List, Dict
from datetime import datetime

def decode_instagram_text(text: str) -> str:
    """
    Instagram exports have double-encoded UTF-8. This fixes it.
    Example: '\u00c3\u00a9' should become 'é'
    """
    if not text:
        return ""

    try:
        # Try to encode as latin-1 then decode as utf-8 (fixes Instagram encoding)
        return text.encode('latin-1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        # If that fails, return as-is
        return text


def clean_message_content(content: str) -> str:
    """Clean message content: decode unicode, remove URLs, extra whitespace."""
    if not content:
        return ""

    # Decode Instagram's weird encoding
    content = decode_instagram_text(content)

    # Remove URLs
    content = re.sub(r'https?://\S+', '', content)

    # Remove extra whitespace
    content = re.sub(r'\s+', ' ', content)
    content = content.strip()

    return content


def extract_messages_from_json(json_path: Path) -> List[Dict]:
    """Extract clean messages from an Instagram JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        messages = []
        participants = [p['name'] for p in data.get('participants', [])]

        for msg in data.get('messages', []):
            content = msg.get('content')
            if not content:
                # Skip messages without text (photos, reactions, etc.)
                continue

            cleaned = clean_message_content(content)
            if not cleaned or len(cleaned) < 3:
                # Skip empty or very short messages
                continue

            messages.append({
                'content': cleaned,
                'sender': decode_instagram_text(msg.get('sender_name', 'Unknown')),
                'timestamp': msg.get('timestamp_ms', 0),
                'conversation': participants
            })

        return messages

    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        print(f"Error parsing {json_path}: {e}")
        return []


def process_inbox_directory(inbox_path: Path, output_path: Path):
    """
    Process all Instagram JSON files in inbox directory.

    Creates one clean text file per conversation with:
    - Decoded unicode
    - Only message content (no metadata)
    - Chronological order
    """
    inbox_path = Path(inbox_path)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    json_files = list(inbox_path.rglob("message_*.json"))
    print(f"Found {len(json_files)} message files")

    conversation_messages = {}

    # Extract all messages
    for json_file in json_files:
        messages = extract_messages_from_json(json_file)
        if not messages:
            continue

        # Group by conversation (parent directory)
        conversation_name = json_file.parent.name
        if conversation_name not in conversation_messages:
            conversation_messages[conversation_name] = []
        conversation_messages[conversation_name].extend(messages)

    print(f"Processing {len(conversation_messages)} conversations...")

    # Write cleaned conversations
    total_messages = 0
    for conv_name, messages in conversation_messages.items():
        if not messages:
            continue

        # Sort by timestamp
        messages.sort(key=lambda m: m['timestamp'])

        # Create output file
        output_file = output_path / f"{conv_name}.txt"

        with open(output_file, 'w', encoding='utf-8') as f:
            # Write conversation metadata
            participants = messages[0]['conversation']
            f.write(f"Conversation: {' & '.join(participants)}\n")
            f.write(f"Messages: {len(messages)}\n")
            f.write("=" * 70 + "\n\n")

            # Write messages
            for msg in messages:
                timestamp = datetime.fromtimestamp(msg['timestamp'] / 1000).strftime('%Y-%m-%d %H:%M')
                f.write(f"[{timestamp}] {msg['sender']}: {msg['content']}\n")

        total_messages += len(messages)
        print(f"  ✓ {conv_name}: {len(messages)} messages")

    print(f"\n✓ Done!")
    print(f"  Conversations: {len(conversation_messages)}")
    print(f"  Total messages: {total_messages}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python clean_instagram_json.py <inbox_path> [output_path]")
        print("\nExample:")
        print("  python clean_instagram_json.py data/inbox data/inbox_clean")
        sys.exit(1)

    inbox_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else inbox_path.parent / "inbox_clean"

    if not inbox_path.exists():
        print(f"Error: {inbox_path} does not exist")
        sys.exit(1)

    process_inbox_directory(inbox_path, output_path)
