#!/bin/bash
# Free memory on macOS M1
# Usage: ./scripts/free_memory.sh

echo "🧹 Libération de mémoire..."
echo ""

# 1. Kill RAG processes
echo "1️⃣  Arrêt des processus RAG..."
pkill -f "rag_low_level" 2>/dev/null && echo "   ✓ rag_low_level arrêté" || echo "   - Pas de processus rag_low_level"
pkill -f "rag_interactive" 2>/dev/null && echo "   ✓ rag_interactive arrêté" || echo "   - Pas de processus rag_interactive"

sleep 2

# 2. Clear Python caches
echo ""
echo "2️⃣  Nettoyage des caches Python..."
find ~/.cache -type f -name "*.pyc" -delete 2>/dev/null
echo "   ✓ Caches Python nettoyés"

# 3. Clear old logs
echo ""
echo "3️⃣  Suppression des anciens logs (>7 jours)..."
find ~/Library/Logs -type f -mtime +7 -delete 2>/dev/null
echo "   ✓ Anciens logs supprimés"

# 4. Clear Postgres temp files (if running)
echo ""
echo "4️⃣  Nettoyage des fichiers temporaires PostgreSQL..."
docker exec -it $(docker ps -q --filter ancestor=postgres) \
  sh -c 'rm -rf /tmp/pg_*' 2>/dev/null && \
  echo "   ✓ Fichiers temp PostgreSQL nettoyés" || \
  echo "   - PostgreSQL non trouvé ou déjà propre"

# 5. Memory report
echo ""
echo "📊 État de la mémoire après nettoyage:"
echo ""

# Get memory stats
vm_stat | perl -ne '/page size of (\d+)/ and $size=$1; /Pages\s+([^:]+)[^\d]+(\d+)/ and printf("%-20s % 10.1f MB\n", "$1:", $2 * $size / 1048576);' | grep -E "(free|purgeable|occupied)"

echo ""
echo "✅ Nettoyage terminé !"
echo ""
echo "💡 Pour libérer encore plus :"
echo "   • Fermer Docker:  osascript -e 'quit app \"Docker\"'"
echo "   • Fermer VS Code: osascript -e 'quit app \"Visual Studio Code\"'"
echo "   • Purger cache système (nécessite sudo): sudo purge"
echo ""
