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

Au lancement interactif, un prompt vous invite à confirmer ou personnaliser l'URL du serveur (appuyez sur `Entrée` pour garder l'adresse par défaut `http://localhost:8000/v1`).

---

## 🌟 Fonctionnalités

- **Formatage ANSI & Markdown en direct** :
  - **Réflexion en gris** : Les blocs de raisonnement `<think>...</think>` (ou `reasoning_content`) s'affichent en **gris** (`\033[90m`) sous un en-tête `💭 [Réflexion]`, avant de basculer sur `💡 [Réponse]`.
  - **Gras & styles Markdown** : `**texte en gras**` mis en valeur avec ANSI bold, code inline `` `code` ``, titres `#`, `##`, `###` colorés, puces `•`, et blocs de code encadrés avec coloration syntaxique.
- **Invite de configuration de l'URL au démarrage** :
  - Vous pouvez valider l'URL par défaut avec `Entrée` ou entrer une IP/port sur votre réseau local (`http://192.168.1.50:8000`).
  - L'URL est automatiquement mémorisée dans le fichier `.env`.
  - La commande `/url` permet aussi de changer de serveur à chaud pendant la session.
- **Streaming ultra-fluide** : Affichage token par token sans saccades ni duplication de lignes.
- **Auto-détection du modèle** : Détecte automatiquement le modèle chargé sur votre instance vLLM (fallback sur `Qwen/Qwen3.8-27B`).
- **Métriques de performance** : Affiche à chaque fin de réponse le **TTFT** (*Time To First Token*), la vitesse en **tokens/seconde** (`tok/s`) et le total de tokens.
- **Historique & saisie riche** : Mémorisation du contexte, navigation avec flèches `Haut`/`Bas`, persistance dans `~/.vllm_client_history`.
- **Mode multi-lignes** : Collez du texte long directement ou activez `/multiline` (`Esc + Entrée` pour envoyer).
- **Requêtes directes & pipes Unix** :
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
| `/url [lien]` | Affiche ou modifie l'URL du serveur vLLM à chaud |
| `/model [nom]` | Affiche les modèles vLLM détectés ou bascule vers un autre modèle |
| `/clear` | Efface le terminal et réinitialise la mémoire de la conversation |
| `/reset` | Réinitialise la conversation sans effacer l'écran |
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
