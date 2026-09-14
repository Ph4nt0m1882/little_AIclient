#!/usr/bin/env python3
"""
Little AI Client - Client de terminal interactif pour vLLM (compatible API OpenAI).
Optimisé pour les modèles locaux comme Qwen/Qwen3.8-27B.
Supporte le formatage ANSI dynamique, la réflexion en gris et la configuration de l'URL du serveur.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
import time
from typing import Any

import openai
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

DEFAULT_URL = os.getenv("VLLM_BASE_URL", os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"))
DEFAULT_MODEL = "Qwen/Qwen3.8-27B"
DEFAULT_SYSTEM_PROMPT = (
    "Tu es un assistant IA serviable, concis et précis. "
    "Réponds en français avec du code bien formaté en Markdown lorsque c'est pertinent."
)
HISTORY_FILE = Path.home() / ".vllm_client_history"

COMMANDS = [
    "/help",
    "/url",
    "/model",
    "/system",
    "/clear",
    "/reset",
    "/params",
    "/retry",
    "/save",
    "/history",
    "/multiline",
    "/exit",
    "/quit",
]


class AnsiMarkdownStreamer:
    """
    Streamer en temps réel qui convertit à la volée le Markdown et les balises
    de réflexion (<think>) en séquences de couleurs et styles ANSI.
    """
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREY = "\033[90m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    CYAN = "\033[36m"
    BRIGHT_CYAN = "\033[96m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    CODE_BG = "\033[48;5;236m\033[38;5;252m"

    def __init__(self, out=None):
        self.out = out or sys.stdout
        self.buffer = ""
        self.in_think_tag = False
        self.in_reasoning_field = False
        self.in_bold = False
        self.in_italic = False
        self.in_inline_code = False
        self.in_code_block = False
        self.code_block_lang = ""
        self.at_line_start = True
        self.has_printed_think_header = False
        self.has_printed_response_header = False

    @property
    def in_think(self) -> bool:
        return self.in_think_tag or self.in_reasoning_field

    def write_reasoning(self, text: str) -> None:
        """Gère le streaming dédié au champ de raisonnement (delta.reasoning_content)."""
        if not self.in_reasoning_field:
            self.in_reasoning_field = True
            if not self.has_printed_think_header:
                self.out.write(f"\n{self.GREY}💭 [Réflexion]{self.RESET}\n")
                self.has_printed_think_header = True
        self.out.write(f"{self.GREY}{text}")
        self.out.flush()

    def end_reasoning(self) -> None:
        """Clôture la phase de raisonnement issue de delta.reasoning_content."""
        if self.in_reasoning_field:
            self.in_reasoning_field = False
            self.out.write(self.RESET)
            if not self.has_printed_response_header:
                self.out.write(f"\n\n{self.BOLD}{self.BRIGHT_CYAN}💡 [Réponse]{self.RESET}\n")
                self.has_printed_response_header = True
            self.at_line_start = True
            self.out.flush()

    def write(self, text: str) -> None:
        """Ajoute du texte au buffer et traite les balises & styles Markdown."""
        self.buffer += text
        self._process(flush_all=False)

    def flush(self) -> None:
        """Vide le buffer restant et réinitialise tous les attributs ANSI."""
        self._process(flush_all=True)
        if (
            self.in_think
            or self.in_bold
            or self.in_italic
            or self.in_inline_code
            or self.in_code_block
        ):
            self.out.write(self.RESET)
        self.out.flush()

    def _process(self, flush_all: bool = False) -> None:
        i = 0
        n = len(self.buffer)
        while i < n:
            rem = self.buffer[i:]

            # 1. Détection de la balise <think>
            if not self.in_think_tag and rem.startswith("<think>"):
                self.in_think_tag = True
                if not self.has_printed_think_header:
                    self.out.write(f"\n{self.GREY}💭 [Réflexion]{self.RESET}\n{self.GREY}")
                    self.has_printed_think_header = True
                else:
                    self.out.write(self.GREY)
                i += 7
                continue

            # 2. Détection de la balise </think>
            if self.in_think_tag and rem.startswith("</think>"):
                self.in_think_tag = False
                self.out.write(self.RESET)
                if not self.has_printed_response_header:
                    self.out.write(f"\n\n{self.BOLD}{self.BRIGHT_CYAN}💡 [Réponse]{self.RESET}\n")
                    self.has_printed_response_header = True
                i += 8
                self.at_line_start = True
                continue

            # Si on est dans la balise <think>, afficher en gris
            if self.in_think_tag:
                if not flush_all and "</think>".startswith(rem):
                    break
                self.out.write(self.buffer[i])
                i += 1
                continue

            # 3. Blocs de code (```)
            if rem.startswith("```"):
                if not self.in_code_block:
                    nl = self.buffer.find("\n", i + 3)
                    if nl == -1 and not flush_all:
                        break
                    lang = self.buffer[i + 3 : nl].strip() if nl != -1 else ""
                    self.in_code_block = True
                    self.code_block_lang = lang
                    lang_label = f"({lang}) " if lang else ""
                    self.out.write(f"\n{self.MAGENTA}─── Code {lang_label}────────────────────────{self.RESET}\n{self.CYAN}")
                    i = nl + 1 if nl != -1 else n
                    self.at_line_start = True
                    continue
                else:
                    self.in_code_block = False
                    self.out.write(f"{self.RESET}\n{self.MAGENTA}────────────────────────────────────────{self.RESET}\n")
                    i += 3
                    if i < n and self.buffer[i] == "\n":
                        i += 1
                    self.at_line_start = True
                    continue

            if self.in_code_block:
                if not flush_all and "```".startswith(rem):
                    break
                self.out.write(self.buffer[i])
                i += 1
                continue

            # 4. Code en ligne (`...`)
            if self.buffer[i] == "`":
                self.in_inline_code = not self.in_inline_code
                self.out.write(self.CODE_BG if self.in_inline_code else self.RESET)
                i += 1
                continue

            if self.in_inline_code:
                self.out.write(self.buffer[i])
                i += 1
                continue

            # 5. Gras (**...**)
            if rem.startswith("**"):
                self.in_bold = not self.in_bold
                self.out.write(self.BOLD if self.in_bold else self.RESET)
                i += 2
                continue

            # 6. Éléments de début de ligne
            if self.at_line_start:
                if rem.startswith("### "):
                    self.out.write(f"{self.BOLD}{self.YELLOW}")
                    i += 4
                    self.at_line_start = False
                    continue
                elif rem.startswith("## "):
                    self.out.write(f"{self.BOLD}{self.BRIGHT_CYAN}")
                    i += 3
                    self.at_line_start = False
                    continue
                elif rem.startswith("# "):
                    self.out.write(f"{self.BOLD}{self.UNDERLINE}{self.BRIGHT_CYAN}")
                    i += 2
                    self.at_line_start = False
                    continue
                elif rem.startswith("- ") or rem.startswith("* "):
                    self.out.write(f"{self.CYAN}•{self.RESET} ")
                    i += 2
                    self.at_line_start = False
                    continue
                elif rem.startswith("> "):
                    self.out.write(f"{self.GREEN}▎{self.RESET} \033[3m")
                    i += 2
                    self.at_line_start = False
                    continue
                elif rem.startswith("---") or rem.startswith("***"):
                    nl = self.buffer.find("\n", i)
                    if nl == -1 and not flush_all:
                        break
                    self.out.write(f"{self.GREY}────────────────────────────────────────{self.RESET}\n")
                    i = nl + 1 if nl != -1 else n
                    self.at_line_start = True
                    continue

            # Éviter de couper les balises partielles en streaming
            if not flush_all:
                tags = ("<think>", "</think>", "```", "**", "### ", "## ", "# ", "---", "***")
                if any(t.startswith(rem) for t in tags):
                    break

            ch = self.buffer[i]
            self.out.write(ch)
            if ch == "\n":
                self.at_line_start = True
                if self.in_bold:
                    self.out.write(self.BOLD)
            else:
                self.at_line_start = False
            i += 1

        self.buffer = self.buffer[i:]
        self.out.flush()


def normalize_url(url: str) -> str:
    """Normalise l'URL du serveur vLLM."""
    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        url = f"http://{url}"
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


def update_env_file(url: str) -> None:
    """Sauvegarde l'URL choisie dans le fichier .env."""
    try:
        env_file = Path(".env")
        lines = []
        found = False
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("VLLM_BASE_URL="):
                    lines.append(f"VLLM_BASE_URL={url}")
                    found = True
                else:
                    lines.append(line)
        if not found:
            lines.append(f"VLLM_BASE_URL={url}")
            lines.append("OPENAI_API_KEY=EMPTY")
        env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        pass


def prompt_for_server_url(default_url: str) -> str:
    """Demande à l'utilisateur de confirmer ou personnaliser l'URL du serveur au démarrage."""
    console = Console()
    console.print()
    console.print(
        Panel(
            f"[white]Appuyez sur [bold cyan]Entrée[/bold cyan] pour utiliser : [bold green]{default_url}[/bold green]\n"
            "Ou saisissez une nouvelle URL (ex: [cyan]http://localhost:8000[/cyan] ou [cyan]http://192.168.1.50:8000[/cyan]) :[/white]",
            title="[bold blue]🔗 Connexion au serveur vLLM[/bold blue]",
            border_style="blue",
            padding=(0, 2),
        )
    )

    session = PromptSession()
    try:
        style = Style.from_dict({"prompt": "ansicyan bold"})
        user_url = session.prompt(
            f"URL du serveur [{default_url}] > ",
            style=style,
        ).strip()

        if not user_url:
            return default_url

        valid_url = normalize_url(user_url)
        update_env_file(valid_url)
        console.print(f"[green]✓ URL sélectionnée : [bold]{valid_url}[/bold][/green]\n")
        return valid_url
    except (KeyboardInterrupt, EOFError):
        console.print(f"\n[dim]Utilisation de l'URL par défaut : {default_url}[/dim]\n")
        return default_url


class ChatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str = "EMPTY",
        model: str | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 4096,
        show_stats: bool = True,
    ):
        self.console = Console()
        self.base_url = normalize_url(base_url)
        self.api_key = api_key
        self.client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)

        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.show_stats = show_stats
        self.multiline_mode = False

        self.system_prompt = system_prompt
        self.messages: list[dict[str, str]] = []
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

        # Détection ou attribution du modèle
        self.available_models = self.fetch_available_models()
        if model:
            self.model = model
        elif self.available_models:
            self.model = self.available_models[0]
        else:
            self.model = DEFAULT_MODEL

    def set_url(self, new_url: str) -> None:
        """Met à jour l'URL du serveur et réinitialise le client."""
        self.base_url = normalize_url(new_url)
        self.client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.available_models = self.fetch_available_models()
        if self.available_models:
            self.model = self.available_models[0]
        update_env_file(self.base_url)

    def fetch_available_models(self) -> list[str]:
        """Récupère la liste des modèles servis par l'instance vLLM."""
        try:
            res = self.client.models.list()
            return [m.id for m in res.data]
        except Exception:
            return []

    def print_welcome_banner(self) -> None:
        """Affiche la bannière d'accueil et l'état de la connexion."""
        status_color = "green" if self.available_models else "yellow"
        status_text = "Connecté" if self.available_models else "Non joignable (chargement en cours ?)"

        info_table = Table.grid(padding=(0, 2))
        info_table.add_column(style="bold cyan", justify="right")
        info_table.add_column(style="white")

        info_table.add_row("Serveur :", f"{self.base_url} ([{status_color}]{status_text}[/{status_color}])")
        info_table.add_row("Modèle :", f"[bold green]{self.model}[/bold green]")
        info_table.add_row("Paramètres :", f"temp={self.temperature} | top_p={self.top_p} | max_tokens={self.max_tokens}")
        info_table.add_row("Raccourcis :", "[dim]Tapez /help pour les commandes, Ctrl+C pour annuler, Ctrl+D pour quitter[/dim]")

        self.console.print(
            Panel(
                info_table,
                title="[bold blue]🤖 Little AI Client - vLLM Terminal[/bold blue]",
                border_style="blue",
                padding=(1, 2),
            )
        )

        if not self.available_models:
            self.console.print(
                "[yellow]⚠️ Note : Impossible de contacter le serveur vLLM sur "
                f"{self.base_url}.\n"
                "Vérifiez que votre conteneur Docker est bien en cours d'exécution.\n"
                "Tapez [bold]/url[/bold] pour changer d'adresse ou [bold]/model[/bold] pour réessayer.[/yellow]\n"
            )

    def print_help(self) -> None:
        """Affiche l'aide des commandes disponibles."""
        table = Table(title="Commandes disponibles", border_style="dim")
        table.add_column("Commande", style="bold cyan", no_wrap=True)
        table.add_column("Description", style="white")

        table.add_row("/help", "Afficher ce message d'aide")
        table.add_row("/url [lien]", "Afficher ou modifier l'URL du serveur vLLM")
        table.add_row("/model [nom]", "Afficher ou changer de modèle actif (recherche auto)")
        table.add_row("/clear", "Effacer l'écran et réinitialiser l'historique de discussion")
        table.add_row("/reset", "Réinitialiser l'historique sans effacer l'écran")
        table.add_row("/system [texte]", "Afficher ou modifier le prompt système")
        table.add_row("/params [options]", "Afficher ou modifier les paramètres (temp=0.7 max=2048)")
        table.add_row("/retry", "Relancer la dernière réponse de l'assistant")
        table.add_row("/save [fichier.md]", "Sauvegarder l'historique dans un fichier Markdown")
        table.add_row("/history", "Afficher le résumé des échanges actuels")
        table.add_row("/multiline", "Basculer le mode multi-lignes (activer/désactiver)")
        table.add_row("/exit, /quit", "Quitter l'application (ou Ctrl+D)")
        self.console.print(table)
        self.console.print("[dim]Astuce : vous pouvez aussi coller directement du texte sur plusieurs lignes.[/dim]\n")

    def handle_command(self, cmd_line: str) -> bool:
        """Traite une commande commençant par '/'. Retourne False si l'utilisateur souhaite quitter."""
        parts = cmd_line.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/exit", "/quit", "/q"):
            self.console.print("[bold yellow]Au revoir ![/bold yellow]")
            return False

        elif cmd == "/help":
            self.print_help()

        elif cmd == "/url":
            if not arg:
                self.console.print(f"URL actuelle du serveur : [bold green]{self.base_url}[/bold green]")
                self.console.print("[dim]Pour la changer : /url http://nouvelle_adresse:8000[/dim]\n")
            else:
                self.set_url(arg)
                status = "Connecté" if self.available_models else "Non joignable"
                color = "green" if self.available_models else "yellow"
                self.console.print(f"[{color}]✓ URL mise à jour : {self.base_url} ({status})[/{color}]\n")

        elif cmd == "/clear":
            os.system("clear" if os.name == "posix" else "cls")
            self.messages = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
            self.print_welcome_banner()
            self.console.print("[green]✓ Écran et historique réinitialisés.[/green]\n")

        elif cmd == "/reset":
            self.messages = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
            self.console.print("[green]✓ Historique de conversation réinitialisé.[/green]\n")

        elif cmd == "/multiline":
            self.multiline_mode = not self.multiline_mode
            status = "[bold green]activé[/bold green] (utilisez Esc+Entrée pour envoyer)" if self.multiline_mode else "[bold yellow]désactivé[/bold yellow] (Entrée pour envoyer)"
            self.console.print(f"Mode multi-lignes {status}.\n")

        elif cmd == "/system":
            if not arg:
                self.console.print(Panel(self.system_prompt or "[italic dim]Aucun[/italic dim]", title="Prompt système actuel", border_style="cyan"))
                self.console.print("[dim]Pour le changer : /system <nouveau texte>[/dim]\n")
            else:
                self.system_prompt = arg
                if self.messages and self.messages[0]["role"] == "system":
                    self.messages[0]["content"] = arg
                else:
                    self.messages.insert(0, {"role": "system", "content": arg})
                self.console.print("[green]✓ Prompt système mis à jour.[/green]\n")

        elif cmd == "/model":
            self.available_models = self.fetch_available_models()
            if not arg:
                self.console.print(f"Modèle actuel : [bold green]{self.model}[/bold green]")
                if self.available_models:
                    self.console.print(f"Modèles détectés sur vLLM ({len(self.available_models)}) :")
                    for m in self.available_models:
                        marker = "★" if m == self.model else " "
                        self.console.print(f"  {marker} [cyan]{m}[/cyan]")
                else:
                    self.console.print("[yellow]Aucun modèle détecté sur le serveur (vérifiez le conteneur Docker).[/yellow]")
                self.console.print("[dim]Pour changer : /model <nom_du_modele>[/dim]\n")
            else:
                self.model = arg
                self.console.print(f"[green]✓ Modèle actif changé pour : [bold]{self.model}[/bold][/green]\n")

        elif cmd == "/params":
            if not arg:
                table = Table(title="Paramètres de génération", border_style="dim")
                table.add_column("Paramètre", style="cyan")
                table.add_column("Valeur", style="white")
                table.add_row("Temperature", str(self.temperature))
                table.add_row("Top-P", str(self.top_p))
                table.add_row("Max Tokens", str(self.max_tokens))
                table.add_row("Afficher les stats", "Oui" if self.show_stats else "Non")
                self.console.print(table)
                self.console.print("[dim]Syntaxe pour modifier : /params temp=0.8 max=2048 top_p=0.95 stats=off[/dim]\n")
            else:
                for token in arg.split():
                    if "=" in token:
                        key, val = token.split("=", 1)
                        key = key.lower().strip()
                        val = val.strip()
                        try:
                            if key in ("temp", "temperature"):
                                self.temperature = float(val)
                            elif key in ("top_p", "topp"):
                                self.top_p = float(val)
                            elif key in ("max", "max_tokens", "maxtokens"):
                                self.max_tokens = int(val)
                            elif key == "stats":
                                self.show_stats = val.lower() in ("1", "true", "on", "oui", "yes")
                        except ValueError as e:
                            self.console.print(f"[red]Valeur invalide pour {key}: {e}[/red]")
                self.console.print(f"[green]✓ Paramètres mis à jour : temp={self.temperature}, top_p={self.top_p}, max={self.max_tokens}, stats={self.show_stats}[/green]\n")

        elif cmd == "/retry":
            if len(self.messages) >= 2 and self.messages[-1]["role"] == "assistant":
                self.messages.pop()
                last_user_msg = self.messages.pop()
                self.console.print("[yellow]↻ Nouvelle génération de la dernière réponse...[/yellow]\n")
                self.generate_response(last_user_msg["content"])
            elif len(self.messages) >= 1 and self.messages[-1]["role"] == "user":
                last_user_msg = self.messages.pop()
                self.generate_response(last_user_msg["content"])
            else:
                self.console.print("[yellow]Aucun message précédent à relancer.[/yellow]\n")

        elif cmd == "/history":
            total_msgs = len(self.messages)
            user_msgs = sum(1 for m in self.messages if m["role"] == "user")
            asst_msgs = sum(1 for m in self.messages if m["role"] == "assistant")
            char_count = sum(len(m["content"]) for m in self.messages)
            approx_tokens = char_count // 4

            table = Table(title="Historique de la conversation", border_style="dim")
            table.add_column("Métrique", style="cyan")
            table.add_column("Valeur", style="white")
            table.add_row("Messages totaux", str(total_msgs))
            table.add_row("Tours de parole (User / IA)", f"{user_msgs} / {asst_msgs}")
            table.add_row("Caractères cumulés", f"{char_count:,}")
            table.add_row("Tokens estimés dans le contexte", f"~{approx_tokens:,}")
            self.console.print(table)
            self.console.print()

        elif cmd == "/save":
            filename = arg if arg else f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            save_path = Path(filename)
            try:
                lines = [
                    f"# Conversation Exportée - {self.model}",
                    f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
                ]
                for msg in self.messages:
                    role = msg["role"]
                    content = msg["content"]
                    if role == "system":
                        lines.append(f"> **System Prompt** : {content}\n")
                    elif role == "user":
                        lines.append(f"### 👤 Utilisateur\n\n{content}\n")
                    elif role == "assistant":
                        lines.append(f"### 🤖 {self.model}\n\n{content}\n")
                save_path.write_text("\n".join(lines), encoding="utf-8")
                self.console.print(f"[green]✓ Conversation sauvegardée dans [bold]{save_path.resolve()}[/bold][/green]\n")
            except Exception as e:
                self.console.print(f"[red]Erreur lors de la sauvegarde : {e}[/red]\n")

        else:
            self.console.print(f"[red]Commande inconnue : {cmd}. Tapez /help pour la liste des commandes.[/red]\n")

        return True

    def generate_response(self, user_prompt: str) -> None:
        """Envoie le message à vLLM et affiche la réponse en streaming avec formatage ANSI."""
        self.messages.append({"role": "user", "content": user_prompt})

        self.console.print(f"\n[bold green]🤖 {self.model}[/bold green] :")

        full_response = ""
        start_time = time.perf_counter()
        first_token_time = None
        completion_tokens = 0
        prompt_tokens = 0

        streamer = AnsiMarkdownStreamer()

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,  # type: ignore[arg-type]
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )

            for chunk in stream:
                if hasattr(chunk, "usage") and chunk.usage:
                    completion_tokens = chunk.usage.completion_tokens or completion_tokens
                    prompt_tokens = chunk.usage.prompt_tokens or prompt_tokens

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # 1. Gérer delta.reasoning_content si présent (modèles de réflexion vLLM)
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    streamer.write_reasoning(reasoning)
                    completion_tokens += 1
                    full_response += reasoning

                # 2. Gérer delta.content standard (peut aussi contenir <think>...</think>)
                content = getattr(delta, "content", None)
                if content:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    if streamer.in_reasoning_field:
                        streamer.end_reasoning()
                    full_response += content
                    completion_tokens += 1
                    streamer.write(content)

            streamer.flush()
            sys.stdout.write("\n")
            sys.stdout.flush()

            total_elapsed = time.perf_counter() - start_time
            self.messages.append({"role": "assistant", "content": full_response})

            # Ligne de métriques
            if self.show_stats and full_response:
                ttft = (first_token_time - start_time) if first_token_time else total_elapsed
                gen_time = (total_elapsed - ttft) if first_token_time else total_elapsed
                tok_per_sec = (completion_tokens / gen_time) if gen_time > 0 else 0

                stats_str = f"⏱ {total_elapsed:.2f}s (TTFT: {ttft:.2f}s) | ⚡ {tok_per_sec:.1f} tok/s | 📝 {completion_tokens} tokens"
                if prompt_tokens:
                    stats_str += f" (prompt: {prompt_tokens})"
                self.console.print(f"[dim]{stats_str}[/dim]")

        except KeyboardInterrupt:
            streamer.flush()
            sys.stdout.write("\n")
            self.console.print("[yellow]⚠️ Génération interrompue par l'utilisateur.[/yellow]")
            if full_response:
                self.messages.append({"role": "assistant", "content": full_response})
        except openai.APIConnectionError as e:
            streamer.flush()
            self.console.print(f"\n[bold red]❌ Erreur de connexion au serveur vLLM :[/bold red] {e}")
            self.console.print(f"[yellow]Vérifiez que le serveur écoute sur {self.base_url} ou tapez [bold]/url[/bold] pour modifier l'adresse.[/yellow]")
            self.messages.pop()
        except openai.APIStatusError as e:
            streamer.flush()
            self.console.print(f"\n[bold red]❌ Erreur API ({e.status_code}) :[/bold red] {e.message}")
            self.messages.pop()
        except Exception as e:
            streamer.flush()
            self.console.print(f"\n[bold red]❌ Erreur inattendue :[/bold red] {e}")
            self.messages.pop()

        self.console.print()

    def run_interactive(self) -> None:
        """Lance la boucle de chat interactive."""
        self.print_welcome_banner()

        completer = WordCompleter(COMMANDS, sentence=True)
        style = Style.from_dict({
            "prompt": "ansicyan bold",
            "arrow": "ansibrightyellow bold",
        })

        session: PromptSession[str] = PromptSession(
            history=FileHistory(str(HISTORY_FILE)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=completer,
            style=style,
        )

        while True:
            try:
                prompt_prefix = "[multi] " if self.multiline_mode else ""
                prompt_text = f"{prompt_prefix}Vous > "
                user_input = session.prompt(
                    prompt_text,
                    multiline=self.multiline_mode,
                ).strip()

                if not user_input:
                    continue

                if user_input.startswith("/"):
                    should_continue = self.handle_command(user_input)
                    if not should_continue:
                        break
                    continue

                self.generate_response(user_input)

            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[bold yellow]Fermeture du client. À bientôt ![/bold yellow]")
                break


def run_one_shot(client: ChatClient, prompt: str) -> None:
    """Mode prompt unique (non interactif ou pipe stdin) avec formatage ANSI."""
    try:
        stream = client.client.chat.completions.create(
            model=client.model,
            messages=[
                {"role": "system", "content": client.system_prompt},
                {"role": "user", "content": prompt},
            ],  # type: ignore[arg-type]
            temperature=client.temperature,
            top_p=client.top_p,
            max_tokens=client.max_tokens,
            stream=True,
        )
        streamer = AnsiMarkdownStreamer()
        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    streamer.write_reasoning(reasoning)
                content = getattr(delta, "content", None)
                if content:
                    if streamer.in_reasoning_field:
                        streamer.end_reasoning()
                    streamer.write(content)
        streamer.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()
    except Exception as e:
        sys.stderr.write(f"Erreur vLLM : {e}\n")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Little AI Client - Client de terminal pour vLLM (OpenAI-compatible)"
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"URL de base du serveur vLLM (défaut: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help=f"Nom du modèle (défaut: auto-détection ou {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--system",
        "-s",
        default=DEFAULT_SYSTEM_PROMPT,
        help="Prompt système par défaut",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=0.7,
        help="Température de génération (défaut: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling (défaut: 0.9)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Nombre maximal de tokens générés (défaut: 4096)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", "EMPTY"),
        help="Clé API si nécessaire (défaut: EMPTY)",
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Désactiver l'affichage des métriques de génération (tok/s, durée)",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="Mode requête directe : envoie ce prompt, affiche la réponse et quitte",
    )

    args = parser.parse_args()

    # Si stdin est redirigé (pipe), on exécute directement sans poser de question
    if not sys.stdin.isatty():
        piped_input = sys.stdin.read().strip()
        full_prompt = f"{args.prompt}\n\n{piped_input}" if args.prompt else piped_input
        if full_prompt:
            chat_client = ChatClient(
                base_url=args.url,
                api_key=args.api_key,
                model=args.model,
                system_prompt=args.system,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                show_stats=not args.no_stats,
            )
            run_one_shot(chat_client, full_prompt)
            return

    # Si l'argument -p est passé, on exécute directement
    if args.prompt:
        chat_client = ChatClient(
            base_url=args.url,
            api_key=args.api_key,
            model=args.model,
            system_prompt=args.system,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            show_stats=not args.no_stats,
        )
        run_one_shot(chat_client, args.prompt)
        return

    # En mode interactif : demander confirmation ou personnalisation de l'URL du serveur
    selected_url = prompt_for_server_url(args.url)

    chat_client = ChatClient(
        base_url=selected_url,
        api_key=args.api_key,
        model=args.model,
        system_prompt=args.system,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        show_stats=not args.no_stats,
    )

    chat_client.run_interactive()


if __name__ == "__main__":
    main()
