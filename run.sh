#!/usr/bin/env bash
# Script de lancement rapide pour Little AI Client avec uv
set -e

# Détection de l'exécutable uv
if command -v uv &> /dev/null; then
    UV_BIN="uv"
elif [ -f "$HOME/.local/bin/uv" ]; then
    UV_BIN="$HOME/.local/bin/uv"
else
    echo "❌ Erreur : 'uv' n'a pas été trouvé."
    echo "Installez-le avec : curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

exec "$UV_BIN" run client.py "$@"
