#!/usr/bin/env python3
"""
Test the enhanced chat metadata extraction on Facebook Messenger data.
"""

import json
import re
from pathlib import Path
from collections import Counter


def extract_chat_metadata(text: str, file_path: str = "") -> dict:
    """
    Extract metadata from chat log messages.
    Supports multiple formats with enhanced Facebook Messenger detection.
    """
    metadata = {}

    # Check for Facebook Messenger header format
    header_pattern = r'Conversation:\s*(.+?)\s*\nMessages:\s*(\d+)'
    header_match = re.search(header_pattern, text)

    if header_match:
        metadata["platform"] = "facebook_messenger"
        conversation_line = header_match.group(1)
        all_participants = [p.strip() for p in conversation_line.split('&')]
        metadata["conversation_participants"] = all_participants
        metadata["conversation_participant_count"] = len(all_participants)
        total_messages = int(header_match.group(2))
        metadata["total_conversation_messages"] = total_messages
        metadata["conversation_type"] = "group_chat" if len(all_participants) > 2 else "direct_message"

        if file_path:
            filename = file_path.split('/')[-1]
            group_id_match = re.search(r'_(\d{10,})(?:\.txt)?$', filename)
            if group_id_match:
                metadata["group_id"] = group_id_match.group(1)
            group_name_match = re.search(r'/([^/_]+)_\d+(?:\.txt)?$', file_path)
            if group_name_match:
                metadata["group_name"] = group_name_match.group(1)

    # Pattern: [YYYY-MM-DD HH:MM] Name: message
    pattern = r'\[(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})\]\s+([^:]+):'
    matches = re.findall(pattern, text)

    if not matches:
        if metadata:
            metadata["is_chat_log"] = True
            metadata["message_count"] = 0
        return metadata

    participants = []
    dates = []

    for date_str, time_str, participant in matches:
        participants.append(participant.strip())
        dates.append(date_str)

    unique_participants = list(set(participants))
    unique_dates = sorted(set(dates))

    metadata.update({
        "participants": unique_participants,
        "participant_count": len(unique_participants),
        "message_count": len(matches),
        "is_chat_log": True,
    })

    if unique_dates:
        metadata["dates"] = unique_dates
        metadata["earliest_date"] = unique_dates[0]
        metadata["latest_date"] = unique_dates[-1]
        metadata["date_range"] = f"{unique_dates[0]} to {unique_dates[-1]}"

        from datetime import datetime
        try:
            start = datetime.strptime(unique_dates[0], "%Y-%m-%d")
            end = datetime.strptime(unique_dates[-1], "%Y-%m-%d")
            metadata["time_span_days"] = (end - start).days + 1
        except:
            pass

    if participants:
        participant_counts = Counter(participants)
        dominant = participant_counts.most_common(1)[0]
        metadata["dominant_participant"] = dominant[0]
        metadata["dominant_participant_count"] = dominant[1]

    if metadata.get("platform") == "facebook_messenger":
        attachment_patterns = [
            r'a envoyé une pièce jointe',
            r'sent an attachment',
            r'Vous avez envoyé une pièce jointe',
            r'You sent an attachment'
        ]
        attachment_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in attachment_patterns)
        if attachment_count > 0:
            metadata["attachment_count"] = attachment_count
            metadata["has_attachments"] = True

        reaction_pattern = r'reacted\s+[\U0001F300-\U0001F9FF]|a réagi'
        if re.search(reaction_pattern, text):
            metadata["has_reactions"] = True
            reaction_matches = re.findall(reaction_pattern, text)
            metadata["reaction_count"] = len(reaction_matches)

        group_events = []
        event_patterns = [
            (r'added (.+?) to the group', 'member_added'),
            (r'a ajouté (.+?) au groupe', 'member_added'),
            (r'named the group (.+?)\.', 'group_renamed'),
            (r'a nommé le groupe (.+?)\.', 'group_renamed'),
            (r'left the group', 'member_left'),
            (r'a quitté le groupe', 'member_left'),
            (r'removed (.+?) from the group', 'member_removed'),
            (r'changed the group photo', 'photo_changed'),
        ]

        for pattern, event_type in event_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match else ''
                group_events.append({
                    "type": event_type,
                    "detail": match if match else event_type
                })

        if group_events:
            metadata["group_events"] = group_events
            metadata["group_event_count"] = len(group_events)
            metadata["has_group_events"] = True

    if "platform" not in metadata and matches:
        metadata["platform"] = "generic"

    return metadata

def test_file():
    """Test metadata extraction on real Facebook Messenger file."""

    file_path = "/Users/frytos/code/llamaIndex-local-rag/data/inbox_clean/pralololol_5770450729659393.txt"

    # Read first 5000 characters (should include header + some messages)
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read(5000)

    print("="*70)
    print("TESTING ENHANCED CHAT METADATA EXTRACTION")
    print("="*70)
    print(f"\nFile: {file_path}")
    print(f"Text length: {len(text)} characters")
    print("\n" + "="*70)
    print("EXTRACTED METADATA:")
    print("="*70 + "\n")

    # Extract metadata
    metadata = extract_chat_metadata(text, file_path=file_path)

    # Pretty print results
    for key, value in sorted(metadata.items()):
        if key == "dates" and isinstance(value, list) and len(value) > 5:
            # Show first and last dates only
            print(f"  {key}: [{value[0]}, ..., {value[-1]}] ({len(value)} dates)")
        elif key == "conversation_participants" and isinstance(value, list):
            print(f"  {key}:")
            for participant in value:
                print(f"    - {participant}")
        elif key == "group_events" and isinstance(value, list):
            print(f"  {key}:")
            for event in value:
                print(f"    - {event['type']}: {event['detail']}")
        elif isinstance(value, (list, dict)) and len(str(value)) > 100:
            print(f"  {key}: {type(value).__name__} (truncated)")
        else:
            print(f"  {key}: {value}")

    print("\n" + "="*70)
    print("METADATA FIELDS EXTRACTED:")
    print("="*70)
    print(f"Total fields: {len(metadata)}")
    print(f"Fields: {', '.join(sorted(metadata.keys()))}")

    # Save full metadata as JSON for inspection
    output_file = Path(__file__).parent / "test_metadata_output.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Full metadata saved to: {output_file}")

    return metadata

if __name__ == "__main__":
    metadata = test_file()
