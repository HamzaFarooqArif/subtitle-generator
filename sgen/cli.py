"""sgen — offline subtitle generation.

    sgen run VIDEO...            transcribe and write .srt/.vtt
    sgen ui                      start the web UI (stops a previous one first)
    sgen stop                    stop running servers
    sgen config                  show settings; --init creates the file
    sgen models pull             download weights (the only online command)
    sgen models verify           prove the offline path resolves
    sgen reformat SIDECAR        rebuild cues without re-transcribing
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import serverctl, settings
from .config import Config, WORK_DIR, enforce_offline

app = typer.Typer(add_completion=False, help=__doc__, no_args_is_help=True)
models_app = typer.Typer(help="Manage local model weights.", no_args_is_help=True)
app.add_typer(models_app, name="models")

def _force_utf8_output() -> None:
    """Stop non-ASCII output from killing the process on a legacy console.

    Windows consoles often default to cp1252, which cannot encode the arrows and
    warning signs used below — printing them raised UnicodeEncodeError and took
    the command down with it. That was worst for `sgen run`, which crashed at the
    moment it tried to warn that a result looked wrong, and for `sgen ui`, which
    died before the server started.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


_force_utf8_output()
console = Console()

VIDEO_SUFFIXES = {
    ".mp4", ".mkv", ".mov", ".avi", ".m4v", ".wmv", ".flv", ".webm", ".mpg",
    ".mpeg", ".mts", ".m2ts", ".3gp", ".ts", ".vob",
}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}


def _skip_finished(files: list[Path], cfg: Config, out_dir: Optional[Path]):
    """Split a batch into what still needs doing and what is already done.

    The judgement comes from the subtitle files themselves, so a batch of fifty
    survives a reboot without any stored progress: re-run the same command and it
    continues. A file whose subtitles look truncated is redone rather than
    trusted.
    """
    from . import resume as resume_mod

    target = cfg.translate_target if cfg.translate_to_english else None
    todo, done = [], []
    for source in files:
        status = resume_mod.classify(
            source, cfg, out_dir=out_dir, translate_target=target
        )
        (todo if status.needs_work else done).append(
            source if status.needs_work else status
        )
    return todo, done


def _settings() -> settings.Settings:
    """Read settings.local.yaml, complaining about a broken one but continuing."""
    user = settings.load_or_default()
    if user.error:
        console.print(f"[yellow]! settings ignored:[/] {user.error}")
    return user


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("faster_whisper", "urllib3", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _expand(inputs: list[Path]) -> list[Path]:
    """Expand directories into media files, recursively."""
    allowed = VIDEO_SUFFIXES | AUDIO_SUFFIXES
    found: list[Path] = []
    for item in inputs:
        if item.is_dir():
            found.extend(
                p for p in sorted(item.rglob("*"))
                if p.is_file() and p.suffix.lower() in allowed
            )
        elif item.is_file():
            found.append(item)
        else:
            console.print(f"[yellow]skipping missing path:[/] {item}")
    return found


@app.command()
def run(
    inputs: list[Path] = typer.Argument(..., help="Video/audio files or directories."),
    out_dir: Optional[Path] = typer.Option(
        None, "--out-dir", "-o",
        help="Where to write subtitles. Default: alongside each source file.",
    ),
    profile: Optional[str] = typer.Option(None, "--profile", "-p"),
    model: Optional[str] = typer.Option(None, "--model", help="Override ASR model."),
    language: Optional[str] = typer.Option(
        None, "--language", "-l",
        help="Pin the language (e.g. en, de, es). Default: detect per file.",
    ),
    hotwords: Optional[str] = typer.Option(
        None, "--hotwords",
        help="Comma-separated names/jargon to bias decoding toward.",
    ),
    batch_size: Optional[int] = typer.Option(None, "--batch-size"),
    keep_suppressed: Optional[bool] = typer.Option(
        None, "--keep-suppressed/--drop-suppressed",
        help="Include gated segments in output, for reviewing what was cut.",
    ),
    romanize: Optional[bool] = typer.Option(
        None, "--romanize/--no-romanize",
        help="Also write Latin-script subtitles for non-Latin languages "
             "(नमस्ते -> namaste), as <name>.<lang>-Latn.srt.",
    ),
    translate: bool = typer.Option(
        False, "--translate",
        help="Also write translated subtitles as <name>.<target>.srt.",
    ),
    translate_engine: Optional[str] = typer.Option(
        None, "--translate-engine",
        help="'auto' (default) picks by how punctuated the transcript is. "
             "'nllb' translates text (better on speech). 'whisper' translates "
             "audio (better on unpunctuated lyrics, no extra model).",
    ),
    translate_target: Optional[str] = typer.Option(
        None, "--translate-to",
        help="Target language code for translation (default: en). "
             "Only 'nllb' supports targets other than English.",
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Redo completed work."),
    resume: bool = typer.Option(
        True, "--resume/--no-resume",
        help="Skip files that already have complete subtitles for these settings. "
             "Judged from the subtitle files on disk, so an interrupted batch "
             "picks up where it stopped.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Transcribe files and write subtitles."""
    _setup_logging(verbose)
    enforce_offline()

    # Precedence: a flag given here, then settings.local.yaml, then the profile.
    user = _settings()
    cfg = Config.load(profile or user.defaults.profile)
    settings.apply_defaults(cfg, user)

    if model:
        cfg.asr.model = model
    if language:
        cfg.asr.language = language
    if hotwords:
        cfg.asr.hotwords = hotwords
    if batch_size:
        cfg.asr.batch_size = batch_size
    if keep_suppressed is not None:
        cfg.gating.drop_suppressed = not keep_suppressed
    if romanize is not None:
        cfg.romanize = romanize
    if out_dir is None and user.defaults.out_dir:
        out_dir = Path(user.defaults.out_dir)
    if translate:
        cfg.translate_to_english = True
    if translate_engine:
        if translate_engine not in ("auto", "nllb", "whisper"):
            console.print("[red]--translate-engine must be 'auto', 'nllb' or 'whisper'[/]")
            raise typer.Exit(2)
        cfg.translate_engine = translate_engine
    if translate_target:
        cfg.translate_target = translate_target
        cfg.translate_to_english = True

    files = _expand(inputs)
    if not files:
        console.print("[red]No media files found.[/]")
        raise typer.Exit(1)

    found = len(files)
    if resume and not overwrite:
        files, skipped = _skip_finished(files, cfg, out_dir)
        if skipped:
            console.print(
                f"[dim]{len(skipped)} of {found} already have subtitles for these "
                f"settings — skipping. Use --no-resume to redo them.[/]"
            )
            for status in skipped[:5]:
                console.print(f"  [dim]· {status.source.name}: {status.reason}[/]")
            if len(skipped) > 5:
                console.print(f"  [dim]· … and {len(skipped) - 5} more[/]")
        if not files:
            console.print("[green]Nothing left to do.[/]")
            return

    console.print(
        f"[bold]{len(files)}[/] file(s) · profile [cyan]{cfg.profile}[/] · "
        f"model [cyan]{cfg.asr.model}[/] ({cfg.asr.compute_type})"
    )

    # Import late so `sgen models verify` works before torch is importable.
    from .pipeline import Pipeline

    results = []
    with Pipeline(cfg) as pipeline:
        for i, path in enumerate(files, 1):
            console.print(f"\n[bold cyan]({i}/{len(files)})[/] {path.name}")
            started = time.perf_counter()
            try:
                result = pipeline.process(path, out_dir=out_dir, overwrite=overwrite)
            except Exception as exc:  # one bad file must not kill a batch
                logging.getLogger(__name__).debug("failure", exc_info=True)
                console.print(f"  [red]failed:[/] {exc}")
                results.append((path, None, str(exc)))
                continue

            elapsed = time.perf_counter() - started
            speed = (result.duration / elapsed) if elapsed > 0 else 0
            console.print(
                f"  [green]{len(result.cues)} cues[/] · {result.language} "
                f"({result.language_probability:.0%}) · {result.gate_summary}"
            )
            console.print(f"  {elapsed:.1f}s ({speed:.0f}x realtime)")

            # A file that essentially failed must not read as a success.
            if result.verdict and result.verdict.suspect:
                console.print(
                    f"  [bold yellow]⚠ RESULT LOOKS WRONG[/] "
                    f"[dim]({', '.join(result.verdict.warnings)})[/]"
                )
                for note in result.verdict.notes:
                    console.print(f"    [yellow]·[/] {note}")
            elif result.verdict:
                console.print(
                    f"  [dim]coverage {result.verdict.coverage:.0%} of audio[/]"
                )

            flagged = [c for c in result.cues if c.warnings]
            if flagged:
                counts: dict[str, int] = {}
                for cue in flagged:
                    for warning in cue.warnings:
                        key = warning.split("_")[0] if warning.startswith("cps") else warning
                        counts[key] = counts.get(key, 0) + 1
                detail = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
                console.print(f"  [yellow]{len(flagged)} cue(s) flagged:[/] {detail}")
            for out in result.outputs:
                console.print(f"  [dim]->[/] {out}")
            results.append((path, result, None))

    _summary(results)


def _summary(results: list) -> None:
    failures = [(p, e) for p, r, e in results if r is None]
    if len(results) > 1:
        table = Table(title="Summary", show_lines=False)
        table.add_column("File", overflow="fold")
        table.add_column("Cues", justify="right")
        table.add_column("Lang")
        table.add_column("Gated", justify="right")
        for path, result, error in results:
            if result is None:
                table.add_row(path.name, "-", "-", "[red]failed[/]")
            else:
                table.add_row(
                    path.name,
                    str(len(result.cues)),
                    result.language,
                    f"{result.suppressed_count}",
                )
        console.print()
        console.print(table)
    if failures:
        console.print(f"\n[red]{len(failures)} file(s) failed.[/]")
        raise typer.Exit(1)


@app.command()
def reformat(
    sidecar: Path = typer.Argument(..., help="A .sgen.json file."),
    profile: Optional[str] = typer.Option(None, "--profile", "-p"),
    max_chars: Optional[int] = typer.Option(None, "--max-chars"),
    target_cps: Optional[float] = typer.Option(None, "--target-cps"),
) -> None:
    """Rebuild cues and subtitle files from a sidecar. No GPU, no re-transcribe."""
    _setup_logging(False)
    from .pipeline import reformat_from_sidecar

    user = _settings()
    cfg = Config.load(profile or user.defaults.profile)
    settings.apply_defaults(cfg, user)
    if max_chars:
        cfg.cues.max_chars_per_line = max_chars
    if target_cps:
        cfg.cues.target_cps = target_cps

    outputs = reformat_from_sidecar(sidecar, cfg)
    for out in outputs:
        console.print(f"[green]wrote[/] {out}")


@models_app.command("pull")
def models_pull(
    include_turbo: bool = typer.Option(True, "--turbo/--no-turbo"),
    include_align: bool = typer.Option(True, "--align/--no-align"),
    translation: bool = typer.Option(
        False, "--translation",
        help="Also fetch and convert the NLLB translation model (~5.5 GB "
             "download, ~1.4 GB converted).",
    ),
    translation_model: str = typer.Option(
        "nllb-1.3b", "--translation-model",
        help="nllb-1.3b (better) or nllb-600m (smaller, faster).",
    ),
) -> None:
    """Download model weights. The only command that touches the network."""
    _setup_logging(True)

    if translation:
        from .download import _prepare_env, pull_translation

        _prepare_env()
        pull_translation(translation_model, console=console)
        return

    from .download import pull_all

    pull_all(include_turbo=include_turbo, include_align=include_align, console=console)


@models_app.command("verify")
def models_verify() -> None:
    """Check every model resolves locally with the network disabled."""
    enforce_offline()
    from . import models as registry

    table = Table(title="Local models")
    table.add_column("Model")
    table.add_column("Status")
    ok = True
    for name, present in registry.available().items():
        required = name in registry.REQUIRED
        if present:
            status = "[green]ok[/]"
        elif required:
            status = "[red]missing[/]"
            ok = False
        else:
            status = "[dim]not installed (optional)[/]"
        table.add_row(name, status)
    console.print(table)
    if not ok:
        console.print("[yellow]Run:[/] sgen models pull")
        raise typer.Exit(1)
    console.print("[green]All models resolve offline.[/]")


@app.command("export-text")
def export_text_cmd(
    sidecar: Path = typer.Argument(..., help="A .sgen.json sidecar under work/."),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Write to a file."),
) -> None:
    """Export numbered subtitle text for an external translator."""
    _setup_logging(False)
    from . import roundtrip
    from .pipeline import _cues_from_sidecar

    cues = _cues_from_sidecar(sidecar)
    text = roundtrip.export_text(cues)
    if out:
        out.write_text(text, encoding="utf-8")
        console.print(f"[green]wrote[/] {out} ({len(cues)} lines)")
        # Everything else here is offline; this is the one file meant to leave.
        console.print(
            "\n[yellow]![/] This file contains the transcript in plain text. "
            "Pasting it into an online\n  translator sends it to that service — "
            "the only point in this pipeline where\n  anything leaves the machine."
        )
    else:
        print(text)


@app.command("import-text")
def import_text_cmd(
    sidecar: Path = typer.Argument(..., help="The .sgen.json this text came from."),
    translated: Path = typer.Argument(..., help="File containing the translation."),
    language: str = typer.Option("en", "--language", "-l", help="Target language code."),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir", "-o"),
    drop_untranslated: bool = typer.Option(False, "--drop-untranslated"),
) -> None:
    """Apply an external translation onto the original timings."""
    _setup_logging(False)
    from . import roundtrip
    from .pipeline import _cues_from_sidecar
    from .write import load_sidecar, write_subtitles

    cues = _cues_from_sidecar(sidecar)
    text = translated.read_text(encoding="utf-8-sig")
    applied, report = roundtrip.apply_translation(
        cues, text, keep_untranslated=not drop_untranslated
    )
    if not report.ok:
        console.print(
            "[red]Could not match any translated line to a cue.[/] Keep the line "
            "numbers, or keep one line per cue."
        )
        raise typer.Exit(1)

    data = load_sidecar(sidecar)
    cfg = Config.load(data.get("config", {}).get("profile") or "home-video")
    rebroken = roundtrip.rebreak(applied, cfg.cues)

    source = Path(data["source"]["path"])
    base = (out_dir or source.parent) / source.stem
    written = write_subtitles(rebroken, base, cfg.formats, language, cfg.encoding)

    console.print(f"[green]{report.summary()}[/]")
    for path in written:
        console.print(f"  [dim]->[/] {path}")


@app.command()
def ui(
    host: Optional[str] = typer.Option(
        None, "--host", help="Bind address. Localhost only by default."
    ),
    port: Optional[int] = typer.Option(None, "--port", "-p"),
    open_browser: Optional[bool] = typer.Option(None, "--open/--no-open"),
    replace: Optional[bool] = typer.Option(
        None, "--replace/--no-replace",
        help="Stop servers from a previous launch first (default: yes).",
    ),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes (development)."),
) -> None:
    """Launch the local web UI."""
    _setup_logging(False)
    enforce_offline()

    server = _settings().server
    host = host or server.host
    port = port or server.port
    if open_browser is None:
        open_browser = server.open_browser
    if replace is None:
        replace = server.replace_running

    if replace:
        _replace_running(port)
    elif serverctl.port_is_open(port):
        console.print(
            f"[red]Port {port} is already in use.[/] Something is listening there "
            f"already — run [bold]sgen stop[/] first, or pass --port."
        )
        raise typer.Exit(1)

    try:
        import uvicorn
    except ImportError:
        console.print("[red]uvicorn is not installed.[/] Run: pip install -r requirements.txt")
        raise typer.Exit(1)

    url = f"http://{host}:{port}"
    console.print(f"\n  [bold cyan]sgen ui[/] → [link={url}]{url}[/]")
    console.print("  [dim]Localhost only. Media is read in place, never uploaded.[/]\n")

    if open_browser:
        import threading
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    if reload:
        uvicorn.run("sgen.server.app:create_app", factory=True, host=host, port=port,
                    reload=True, log_level="warning")
    else:
        from .server.app import create_app

        uvicorn.run(create_app(), host=host, port=port, log_level="warning")


def _replace_running(port: int) -> None:
    """Stop previous servers before binding, so they cannot pile up.

    Anything on the port that is *not* sgen is refused rather than killed — the
    port number is not a licence to terminate somebody else's process.
    """
    stopped, others = serverctl.stop_running()
    for instance in stopped:
        console.print(f"  [dim]stopped previous server on {instance.describe()}[/]")

    blocking = [i for i in others if i.port == port]
    if blocking:
        console.print(
            f"[red]Port {port} is held by {blocking[0].describe()}, which is not "
            f"sgen.[/] Close it or start on another port with --port."
        )
        raise typer.Exit(1)
    for instance in others:
        console.print(f"  [yellow]left alone:[/] {instance.describe()}")


@app.command()
def stop(
    port: Optional[int] = typer.Option(
        None, "--port", "-p", help="Only stop this port. Default: every sgen server."
    ),
) -> None:
    """Stop running sgen servers."""
    _force_utf8_output()
    ports = [port] if port else serverctl.PORT_SCAN
    stopped, others = serverctl.stop_running(ports)

    for instance in stopped:
        console.print(f"[green]stopped[/] {instance.describe()}")
    for instance in others:
        console.print(f"[yellow]left alone[/] {instance.describe()}")
    if not stopped and not others:
        console.print("[dim]No servers were running.[/]")


@app.command()
def config(
    init: bool = typer.Option(
        False, "--init", help="Create settings.local.yaml from the template."
    ),
    set_: Optional[list[str]] = typer.Option(
        None, "--set",
        help="Set a property, e.g. --set api_keys.google=AIza… "
             "--set defaults.profile=music. Repeatable.",
    ),
    edit: bool = typer.Option(False, "--edit", help="Open the file in your editor."),
) -> None:
    """Show or change local settings (API keys, defaults, UI port)."""
    _force_utf8_output()
    path = settings.settings_path()

    if init:
        path, created = settings.ensure_file(path)
        console.print(
            f"[green]created[/] {path}" if created else f"[yellow]exists already:[/] {path}"
        )

    if set_:
        try:
            changes = dict(settings.parse_assignment(item) for item in set_)
            settings.set_values(changes, path)
        except settings.SettingsError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(2)
        for key in changes:
            console.print(f"[green]set[/] {key}")

    if edit:
        settings.ensure_file(path)
        typer.launch(str(path))

    user = settings.load_or_default(path)
    console.print(
        f"\n[bold]{path}[/]" + ("" if user.exists else "  [dim](does not exist yet)[/]")
    )
    if user.error:
        console.print(f"[red]{user.error}[/]")
    if not user.exists:
        console.print("[dim]Run[/] sgen config --init [dim]to create it.[/]")

    table = Table(show_header=True, header_style="dim")
    table.add_column("Property")
    table.add_column("Value", overflow="fold")
    table.add_column("From", style="dim")

    for name in ("google", "deepl"):
        value = getattr(user.api_keys, name)
        source = user.key_source.get(name, "unset")
        table.add_row(
            f"api_keys.{name}",
            # Never print a key, not even locally: terminals get pasted into
            # chats and scrollback outlives the session.
            "[green]set[/]" if value else "[yellow]not set[/]",
            "environment" if source in settings.ENV_KEYS.values() else
            ("file" if value else "—"),
        )
    table.add_row("api_keys.deepl_plan", user.api_keys.deepl_plan, "")

    d = user.defaults
    rows = [
        ("defaults.profile", d.profile),
        ("defaults.language", d.language or "detect"),
        ("defaults.hotwords", d.hotwords or "—"),
        ("defaults.romanize", str(d.romanize).lower()),
        ("defaults.keep_suppressed", str(d.keep_suppressed).lower()),
        ("defaults.formats", ", ".join(d.formats)),
        ("defaults.out_dir", d.out_dir or "next to each source file"),
        ("defaults.translate.provider", d.translate.provider),
        ("defaults.translate.target", d.translate.target),
        ("server.host", user.server.host),
        ("server.port", str(user.server.port)),
        ("server.open_browser", str(user.server.open_browser).lower()),
    ]
    for key, value in rows:
        table.add_row(key, value, "file" if user.given(key) else "default")
    console.print(table)
    console.print(
        "[dim]Edit the file directly, or:[/] "
        "sgen config --set defaults.profile=music"
    )


@app.command()
def doctor() -> None:
    """Check the environment: ffmpeg, CUDA, VRAM, models."""
    table = Table(title="Environment")
    table.add_column("Check")
    table.add_column("Result", overflow="fold")

    from . import ffmpeg as ff

    try:
        table.add_row("ffmpeg", ff.resolve("ffmpeg"))
        table.add_row("ffprobe", ff.resolve("ffprobe"))
    except Exception as exc:
        table.add_row("ffmpeg", f"[red]{exc}[/]")

    try:
        import torch

        table.add_row("torch", torch.__version__)
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            table.add_row("gpu", f"{props.name} · {props.total_memory / 1e9:.1f} GB")
        else:
            table.add_row("gpu", "[red]CUDA not available[/]")
    except ImportError:
        table.add_row("torch", "[red]not installed[/]")

    from . import cuda

    registered = cuda.prepare()
    table.add_row("cuda dll dirs", str(len(registered)) if registered else "[yellow]none[/]")

    try:
        import ctranslate2

        table.add_row("ctranslate2", ctranslate2.__version__)
        devices = ctranslate2.get_cuda_device_count()
        table.add_row(
            "ct2 cuda devices",
            str(devices) if devices else "[red]0 — cuDNN/cuBLAS not loadable[/]",
        )
    except Exception as exc:
        table.add_row("ctranslate2", f"[red]{exc}[/]")

    enforce_offline()
    from . import models as registry

    for name, present in registry.available().items():
        table.add_row(f"model:{name}", "[green]ok[/]" if present else "[yellow]missing[/]")

    table.add_row("work dir", str(WORK_DIR))
    console.print(table)


if __name__ == "__main__":
    app()
