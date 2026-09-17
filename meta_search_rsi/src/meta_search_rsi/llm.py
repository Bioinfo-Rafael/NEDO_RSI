# ORIGIN: ORIGINAL
# IMPLEMENTATION_NOTE: Sole optional LLM boundary; Codex CLI only, no provider SDK.
"""Structured Codex proposal transport, isolated from offline research logic."""
import os
from pathlib import Path
import signal
import subprocess


def codex_command(directory: Path, executable: str, model: str | None) -> list[str]:
    """Construct argv against local codex exec 0.154.0; never use shell=True."""
    command = [executable, 'exec', '--sandbox', 'read-only', '--ephemeral',
               '--ignore-user-config', '--skip-git-repo-check', '--color', 'never', '--json',
               '-c', 'cli_auth_credentials_store="file"', '-c', 'forced_login_method="chatgpt"',
               '--cd', str(directory), '--output-schema', str(directory / 'schema.json'),
               '--output-last-message', str(directory / 'proposal.json')]
    if model:
        command += ['--model', model]
    return command + ['-']


def invoke_codex(command: list[str], prompt: str, directory: Path,
                 codex_home: Path, timeout: float) -> None:
    """Call only Codex; kill its process group on timeout or user interruption."""
    codex_home.mkdir(parents=True, exist_ok=True, mode=0o700)
    environment = dict(os.environ, CODEX_HOME=str(codex_home), PYTHONDONTWRITEBYTECODE='1')
    for name in ('OPENAI_API_KEY', 'CODEX_API_KEY', 'ANTHROPIC_API_KEY'):
        environment.pop(name, None)
    with (directory / 'events.jsonl').open('w') as events, (directory / 'stderr.log').open('w') as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=events, stderr=errors,
                                   text=True, env=environment, cwd=directory, start_new_session=True)
        try:
            process.communicate(prompt, timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
    if process.returncode:
        raise RuntimeError(f'Codex exited {process.returncode}; see {directory / "stderr.log"}')


