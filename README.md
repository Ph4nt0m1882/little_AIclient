# 🤖 Little AI Client - Client Terminal pour vLLM

Un client de terminal interactif, moderne et ultra-rapide en Python pour discuter avec votre serveur **vLLM** local (compatible avec l'API OpenAI), géré avec **uv**.

---

## ⚡ Démarrage Rapide

### 1. Assurez-vous que votre serveur vLLM tourne
Exemple de lancement avec Docker pour **Qwen/Qwen3.8-27B** :

```bash
docker run --gpus all \
  --name vllm-qwen38 \
  -d \
  --restart unless-stopped \
  -p 8000:8000 \
  --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN="ton_token_hf_ici" \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3.8-27B \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90
```

### 2. Lancer le client

Dans le dossier `little_AIclient` :

```bash
./run.sh
```

Ou directement avec `uv` :

```bash
uv run client.py
# ou
uv run chat
```

---

## 🌟 Fonctionnalités

- **Streaming en direct** : Réception et affichage des tokens au fur et à mesure sans délai.
- **Auto-détection du modèle** : Détecte automatiquement le modèle chargé sur `http://localhost:8000/v1` (fallback sur `Qwen/Qwen3.8-27B`).
- **Métriques en temps réel** : Affiche la latence (TTFT - *Time To First Token*), la vitesse de génération en **tokens/seconde** (`tok/s`) et le nombre total de tokens.
- **Historique interactif** : Mémorise le contexte de la discussion, navigation avec les flèches Haut/Bas, et persistance dans `~/.vllm_client_history`.
- **Commandes slash intégrées** : Autocomplétion avec `Tab` pour toutes les commandes (`/help`, `/clear`, etc.).
- **Mode multi-lignes** : Collez directement du code ou des textes longs, ou basculez en mode multi-lignes avec `/multiline` (envoi avec `Esc + Entrée` ou `Alt + Entrée`).
- **Support des pipes et requêtes directes** :
  ```bash
  # Requête one-shot
  uv run client.py -p "Explique-moi la différence entre un thread et un processus."

  # Analyse d'un fichier avec stdin
  cat mon_script.py | uv run client.py -p "Trouve les erreurs dans ce code :"
  ```

---

## 💬 Commandes interactives dans le chat

Tapez ces commandes directement à l'invite `Vous > ` :

| Commande | Description |
|---|---|
| `/help` | Affiche l'aide des commandes |
| `/clear` | Efface le terminal et réinitialise la mémoire de la conversation |
| `/reset` | Réinitialise la conversation sans effacer l'écran |
| `/model [nom]` | Affiche les modèles vLLM détectés ou bascule vers un autre modèle |
| `/system [prompt]` | Affiche ou modifie le prompt système à la volée |
| `/params [options]` | Modifie les paramètres de génération (`/params temp=0.8 max=2048`) |
| `/retry` | Relance la génération de la dernière réponse de l'assistant |
| `/save [fichier.md]` | Exporte toute la discussion dans un fichier Markdown structuré |
| `/history` | Affiche les métriques de la session (nombre d'échanges, estimation de tokens) |
| `/multiline` | Active / désactive le mode de saisie multi-lignes |
| `/exit` ou `/quit` | Quitte le client (raccourci : `Ctrl+D`) |

---

## ⚙️ Options de ligne de commande

```bash
uv run client.py [OPTIONS]
```

- `--url <URL>` : URL de l'endpoint vLLM (défaut : `http://localhost:8000/v1` ou variable `VLLM_BASE_URL`)
- `--model, -m <MODELE>` : Forcer le nom du modèle (ex : `Qwen/Qwen3.8-27B`)
- `--system, -s <PROMPT>` : Définir le prompt système initial
- `--temperature, -t <FLOAT>` : Température d'échantillonnage (défaut : `0.7`)
- `--top-p <FLOAT>` : Nucleus sampling (défaut : `0.9`)
- `--max-tokens <INT>` : Nombre maximum de tokens en réponse (défaut : `4096`)
- `--no-stats` : Masque la ligne des métriques de vitesse (`tok/s`, durée)
- `-p, --prompt <TEXTE>` : Envoie le prompt directement sans démarrer la session interactive

---

## 📁 Structure du projet

- `client.py` : Script principal du client interactif.
- `run.sh` : Lanceur bash rapide avec détection automatique de `uv`.
- `pyproject.toml` : Fichier de projet `uv` avec les dépendances (`openai`, `rich`, `prompt-toolkit`).
- `.env` : Fichier de configuration d'environnement par défaut.
