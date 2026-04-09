# TOOLS.md - Local Notes

Skills define _how_ tools work. This file is for _your_ specifics — the stuff that's unique to your setup.

## What Goes Here

Things like:

- Camera names and locations
- SSH hosts and aliases
- Preferred voices for TTS
- Speaker/room names
- Device nicknames
- Anything environment-specific

## Examples

```markdown
### Cameras

- living-room → Main area, 180° wide angle
- front-door → Entrance, motion-triggered

### SSH

- home-server → 192.168.1.100, user: admin

### TTS

- Preferred voice: "Nova" (warm, slightly British)
- Default speaker: Kitchen HomePod
```

## SSH Playbook

Use this section to store the minimum info I need to work safely over SSH.

### SSH Defaults

- Start with read-only discovery commands first (`pwd`, `whoami`, `hostname`, `ls`, `git status`, `nvidia-smi`)
- Do not delete, reset, overwrite, or stop services without Leonardo's explicit approval
- Ask before using `sudo`, package installs, firewall changes, service restarts, or destructive repo commands
- Prefer long jobs inside `tmux` or another resumable session
- Prefer host aliases in `~/.ssh/config` over raw IPs in day-to-day use

### Server Template

Copy and fill one block per server:

```markdown
#### <alias>
- Purpose:
- HostName:
- User:
- Port: 22
- Auth: key / password / other
- Key path:
- ProxyJump:
- Project path:
- Repo path:
- Python env / conda env:
- Common start command:
- Common status command:
- Common log command:
- GPU check command:
- Notes:
```

### Known Servers

#### RXL
- Purpose: remote server accessed over SSH
- HostName: 10.12.208.90
- User: rxl
- Port: 1207
- Auth: unknown yet
- Key path: unknown yet
- ProxyJump: none configured
- Project path: unknown yet
- Repo path: unknown yet
- Python env / conda env: unknown yet
- Common start command: unknown yet
- Common status command: unknown yet
- Common log command: unknown yet
- GPU check command: `nvidia-smi`
- Notes: available via local SSH alias `RXL` from `~/.ssh/config`

### What I need from Leonardo for a new server

- SSH alias or hostname/IP
- username
- port if not 22
- auth method (ideally SSH key)
- remote project directory
- preferred way to run long jobs (`tmux`, `nohup`, scheduler, etc.)
- any commands to avoid

### Safe first test

```bash
ssh <alias> 'pwd && whoami && hostname && uname -a'
```

## Why Separate?

Skills are shared. Your setup is yours. Keeping them apart means you can update skills without losing your notes, and share skills without leaking your infrastructure.

---

Add whatever helps you do your job. This is your cheat sheet.
