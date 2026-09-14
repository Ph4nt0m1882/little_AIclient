#!/usr/bin/env python3
"""
Little AI Client - Client de terminal interactif pour vLLM (compatible API OpenAI).
Optimisé pour les modèles locaux comme Qwen/Qwen3.8-27B.
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
    "/clear",
    "/reset",
    "/system",
    "/model",
    "/params",
    "/retry",
    "/save",
    "/history",
    "/multiline",
    "/exit",
    "/quit",
]


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
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url = f"{self.base_url}/v1"

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

        # Detect or set model
        self.available_models = self.fetch_available_models()
        if model:
            self.model = model
        elif self.available_models:
            self.model = self.available_models[0]
        else:
            self.model = DEFAULT_MODEL

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
        status_text = "Connecté" if self.available_models else "Non détecté (démarrage en cours ?)"

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
                "Tapez [bold]/model[/bold] pour réessayer une fois le conteneur prêt.[/yellow]\n"
            )

    def print_help(self) -> None:
        """Affiche l'aide des commandes disponibles."""
        table = Table(title="Commandes disponibles", border_style="dim")
        table.add_column("Commande", style="bold cyan", no_wrap=True)
        table.add_column("Description", style="white")

        table.add_row("/help", "Afficher ce message d'aide")
        table.add_row("/clear", "Effacer l'écran et réinitialiser l'historique de discussion")
        table.add_row("/reset", "Réinitialiser l'historique sans effacer l'écran")
        table.add_row("/model [nom]", "Afficher ou changer de modèle actif (recherche auto)")
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
        """Envoie le message à vLLM et affiche la réponse en streaming."""
        self.messages.append({"role": "user", "content": user_prompt})

        self.console.print(f"\n[bold green]🤖 {self.model}[/bold green] :")

        full_response = ""
        start_time = time.perf_counter()
        first_token_time = None
        completion_tokens = 0
        prompt_tokens = 0

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
                content = delta.content
                if content:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    full_response += content
                    completion_tokens += 1
                    sys.stdout.write(content)
                    sys.stdout.flush()

            sys.stdout.write("\n")
            sys.stdout.flush()

            total_elapsed = time.perf_counter() - start_time
            self.messages.append({"role": "assistant", "content": full_response})

            # Stats line
            if self.show_stats and full_response:
                ttft = (first_token_time - start_time) if first_token_time else total_elapsed
                gen_time = (total_elapsed - ttft) if first_token_time else total_elapsed
                tok_per_sec = (completion_tokens / gen_time) if gen_time > 0 else 0

                stats_str = f"⏱ {total_elapsed:.2f}s (TTFT: {ttft:.2f}s) | ⚡ {tok_per_sec:.1f} tok/s | 📝 {completion_tokens} tokens"
                if prompt_tokens:
                    stats_str += f" (prompt: {prompt_tokens})"
                self.console.print(f"[dim]{stats_str}[/dim]")

        except KeyboardInterrupt:
            sys.stdout.write("\n")
            self.console.print("[yellow]⚠️ Génération interrompue par l'utilisateur.[/yellow]")
            if full_response:
                self.messages.append({"role": "assistant", "content": full_response})
        except openai.APIConnectionError as e:
            self.console.print(f"\n[bold red]❌ Erreur de connexion au serveur vLLM :[/bold red] {e}")
            self.console.print(f"[yellow]Vérifiez que le serveur écoute sur {self.base_url} et que le conteneur Docker est actif.[/yellow]")
            self.messages.pop()
        except openai.APIStatusError as e:
            self.console.print(f"\n[bold red]❌ Erreur API ({e.status_code}) :[/bold red] {e.message}")
            self.messages.pop()
        except Exception as e:
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
    """Mode prompt unique (non interactif ou pipe stdin)."""
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
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                sys.stdout.write(chunk.choices[0].delta.content)
                sys.stdout.flush()
        sys.stdout.write("\n")
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

    if not sys.stdin.isatty():
        piped_input = sys.stdin.read().strip()
        full_prompt = f"{args.prompt}\n\n{piped_input}" if args.prompt else piped_input
        if full_prompt:
            run_one_shot(chat_client, full_prompt)
            return

    if args.prompt:
        run_one_shot(chat_client, args.prompt)
        return

    chat_client.run_interactive()


if __name__ == "__main__":
    main()
