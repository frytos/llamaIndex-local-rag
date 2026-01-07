#!/usr/bin/env python3
"""
RAG Visualizer - Simple visualization tool for your RAG pipeline
Shows: retrieval results, similarity scores, and source attribution
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

def load_retrieval_log(log_file: str) -> Dict[str, Any]:
    """Parse the RAG log file to extract retrieval information"""
    data = {
        'query': '',
        'chunks': [],
        'scores': [],
        'sources': [],
        'answer': '',
        'metrics': {}
    }

    with open(log_file, 'r') as f:
        content = f.read()

        # Extract query
        if '❓ Query:' in content:
            query_line = [l for l in content.split('\n') if '❓ Query:' in l][0]
            data['query'] = query_line.split('❓ Query:')[1].strip().strip('"')

        # Extract retrieval results
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'Similarity:' in line and 'Source:' in line:
                try:
                    parts = line.split('|')
                    score = float(parts[0].split('Similarity:')[1].strip())
                    source = parts[1].split('Source:')[1].strip()

                    # Get text from next line
                    if i + 1 < len(lines):
                        text_line = lines[i + 1]
                        text = text_line.split('Text:')[1].strip().strip('"') if 'Text:' in text_line else ''

                        data['chunks'].append(text[:200] + '...' if len(text) > 200 else text)
                        data['scores'].append(score)
                        data['sources'].append(source)
                except (IndexError, ValueError, KeyError) as e:
                    # Skip malformed log entries
                    continue

        # Extract answer
        if '✨ FINAL ANSWER:' in content:
            answer_start = content.find('✨ FINAL ANSWER:')
            answer_section = content[answer_start:answer_start+2000]
            answer_lines = answer_section.split('\n')[2:10]  # Get lines after header
            data['answer'] = ' '.join([l.strip() for l in answer_lines if l.strip()])

        # Extract metrics
        for line in lines:
            if 'Best match score:' in line:
                data['metrics']['best_score'] = float(line.split(':')[1].strip())
            elif 'Average score:' in line:
                data['metrics']['avg_score'] = float(line.split(':')[1].strip())
            elif 'Throughput:' in line and 'nodes/second' in line:
                data['metrics']['throughput'] = float(line.split(':')[1].split('nodes/second')[0].strip())

    return data

def create_visualization(data: Dict[str, Any], output_file: str = 'rag_visualization.png'):
    """Create a comprehensive RAG visualization"""

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('RAG Pipeline Visualization', fontsize=16, fontweight='bold')

    # 1. Query box at top
    ax_query = plt.subplot(4, 1, 1)
    ax_query.axis('off')
    ax_query.text(0.5, 0.5, f"Query: {data['query']}",
                  ha='center', va='center', fontsize=14,
                  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
                  wrap=True)
    ax_query.set_xlim(0, 1)
    ax_query.set_ylim(0, 1)

    # 2. Retrieval scores bar chart
    ax_scores = plt.subplot(4, 2, 3)
    if data['scores']:
        bars = ax_scores.barh(range(len(data['scores'])), data['scores'],
                               color=['#2ecc71' if s > 0.8 else '#f39c12' if s > 0.6 else '#e74c3c'
                                      for s in data['scores']])
        ax_scores.set_yticks(range(len(data['scores'])))
        ax_scores.set_yticklabels([f"Chunk {i+1}\n(Src: {src})"
                                   for i, src in enumerate(data['sources'])])
        ax_scores.set_xlabel('Similarity Score')
        ax_scores.set_title('Retrieved Chunks - Similarity Scores')
        ax_scores.set_xlim(0, 1)
        ax_scores.axvline(x=0.8, color='green', linestyle='--', alpha=0.5, label='Excellent (>0.8)')
        ax_scores.axvline(x=0.6, color='orange', linestyle='--', alpha=0.5, label='Good (>0.6)')
        ax_scores.legend(loc='lower right', fontsize=8)
        ax_scores.grid(axis='x', alpha=0.3)

    # 3. Metrics box
    ax_metrics = plt.subplot(4, 2, 4)
    ax_metrics.axis('off')
    metrics_text = "📊 Metrics:\n\n"
    if 'best_score' in data['metrics']:
        metrics_text += f"Best Match: {data['metrics']['best_score']:.4f}\n"
    if 'avg_score' in data['metrics']:
        metrics_text += f"Avg Score: {data['metrics']['avg_score']:.4f}\n"
    if 'throughput' in data['metrics']:
        metrics_text += f"Throughput: {data['metrics']['throughput']:.1f} nodes/s\n"
    metrics_text += f"Chunks Retrieved: {len(data['chunks'])}\n"

    ax_metrics.text(0.1, 0.5, metrics_text,
                   fontsize=11, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7),
                   family='monospace')
    ax_metrics.set_xlim(0, 1)
    ax_metrics.set_ylim(0, 1)

    # 4. Retrieved chunks display
    ax_chunks = plt.subplot(4, 1, 3)
    ax_chunks.axis('off')
    chunks_text = "📄 Retrieved Chunks:\n\n"
    for i, (chunk, score) in enumerate(zip(data['chunks'], data['scores'])):
        quality = "🟢" if score > 0.8 else "🟡" if score > 0.6 else "🔴"
        chunks_text += f"{quality} Chunk {i+1} (Score: {score:.4f}):\n"
        chunks_text += f"   {chunk}\n\n"

    ax_chunks.text(0.05, 0.95, chunks_text,
                  fontsize=9, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.9),
                  family='monospace', wrap=True)
    ax_chunks.set_xlim(0, 1)
    ax_chunks.set_ylim(0, 1)

    # 5. Answer box
    ax_answer = plt.subplot(4, 1, 4)
    ax_answer.axis('off')
    answer_text = f"✨ Generated Answer:\n\n{data['answer'][:500]}"
    if len(data['answer']) > 500:
        answer_text += "..."

    ax_answer.text(0.05, 0.95, answer_text,
                  fontsize=10, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
                  wrap=True)
    ax_answer.set_xlim(0, 1)
    ax_answer.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_file}")
    print(f"   Open it to see the RAG pipeline in action!")

    return output_file

def create_html_report(data: Dict[str, Any], output_file: str = 'rag_report.html'):
    """Create an interactive HTML report"""

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>RAG Pipeline Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .section {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .query {{
            background: #e3f2fd;
            padding: 15px;
            border-left: 4px solid #2196f3;
            font-size: 1.1em;
            margin: 20px 0;
        }}
        .chunk {{
            background: #f9f9f9;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #4caf50;
            border-radius: 4px;
        }}
        .score {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 20px;
            font-weight: bold;
            color: white;
        }}
        .score-excellent {{ background-color: #4caf50; }}
        .score-good {{ background-color: #ff9800; }}
        .score-poor {{ background-color: #f44336; }}
        .answer {{
            background: #e8f5e9;
            padding: 20px;
            border-left: 4px solid #4caf50;
            font-size: 1.05em;
            line-height: 1.6;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
        }}
        .metric-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 RAG Pipeline Execution Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="section">
        <h2>📝 Query</h2>
        <div class="query">{data['query']}</div>
    </div>

    <div class="section">
        <h2>📊 Performance Metrics</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{data['metrics'].get('best_score', 0):.4f}</div>
                <div class="metric-label">Best Match Score</div>
            </div>
            <div class="metric">
                <div class="metric-value">{data['metrics'].get('avg_score', 0):.4f}</div>
                <div class="metric-label">Average Score</div>
            </div>
            <div class="metric">
                <div class="metric-value">{len(data['chunks'])}</div>
                <div class="metric-label">Chunks Retrieved</div>
            </div>
            <div class="metric">
                <div class="metric-value">{data['metrics'].get('throughput', 0):.1f}</div>
                <div class="metric-label">Nodes/Second</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>🔎 Retrieved Chunks</h2>
"""

    for i, (chunk, score, source) in enumerate(zip(data['chunks'], data['scores'], data['sources'])):
        score_class = 'score-excellent' if score > 0.8 else 'score-good' if score > 0.6 else 'score-poor'
        html += f"""
        <div class="chunk">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <strong>Chunk {i+1} (Source: {source})</strong>
                <span class="score {score_class}">{score:.4f}</span>
            </div>
            <div>{chunk}</div>
        </div>
"""

    html += f"""
    </div>

    <div class="section">
        <h2>✨ Generated Answer</h2>
        <div class="answer">{data['answer']}</div>
    </div>

    <div class="section">
        <h2>🔄 Pipeline Flow</h2>
        <div style="text-align: center; padding: 20px;">
            <div style="display: inline-block; text-align: left;">
                <div style="background: #e3f2fd; padding: 15px; margin: 5px 0; border-radius: 5px;">
                    1️⃣ <strong>Query</strong> → Convert to embedding vector
                </div>
                <div style="text-align: center; font-size: 2em;">↓</div>
                <div style="background: #fff3e0; padding: 15px; margin: 5px 0; border-radius: 5px;">
                    2️⃣ <strong>Retrieval</strong> → Find {len(data['chunks'])} most similar chunks
                </div>
                <div style="text-align: center; font-size: 2em;">↓</div>
                <div style="background: #f3e5f5; padding: 15px; margin: 5px 0; border-radius: 5px;">
                    3️⃣ <strong>Context Building</strong> → Combine chunks with query
                </div>
                <div style="text-align: center; font-size: 2em;">↓</div>
                <div style="background: #e8f5e9; padding: 15px; margin: 5px 0; border-radius: 5px;">
                    4️⃣ <strong>Generation</strong> → LLM synthesizes answer
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

    with open(output_file, 'w') as f:
        f.write(html)

    print(f"✅ HTML report saved to: {output_file}")
    print(f"   Open in browser to see interactive report!")

    return output_file

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_rag.py <log_file>")
        print("\nExample:")
        print("  python visualize_rag.py answer.log")
        sys.exit(1)

    log_file = sys.argv[1]

    if not Path(log_file).exists():
        print(f"❌ Error: Log file not found: {log_file}")
        sys.exit(1)

    print(f"📖 Loading RAG log: {log_file}")
    data = load_retrieval_log(log_file)

    print(f"\n📊 Found:")
    print(f"  • Query: {data['query'][:50]}...")
    print(f"  • Chunks: {len(data['chunks'])}")
    print(f"  • Avg Score: {data['metrics'].get('avg_score', 0):.4f}")

    print("\n🎨 Creating visualizations...")

    # Create matplotlib visualization
    try:
        img_file = create_visualization(data)
    except Exception as e:
        print(f"⚠️  Matplotlib visualization failed: {e}")
        img_file = None

    # Create HTML report
    html_file = create_html_report(data)

    print(f"\n✅ Done! Open these files:")
    if img_file:
        print(f"   📊 {img_file}")
    print(f"   🌐 {html_file}")

if __name__ == "__main__":
    main()
