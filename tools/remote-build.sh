#!/usr/bin/env bash
# =============================================================================
# tools/remote-build.sh — build & push runpod-workers images on the Hetzner
# build box straight from your laptop. No Hetzner console, no manual SSH.
#
#   ./tools/remote-build.sh                   pick a target from a menu
#   ./tools/remote-build.sh wan-animate       build + push one model
#   ./tools/remote-build.sh base | all        rebuild base / everything
#   ./tools/remote-build.sh logs              reattach to the live log
#   ./tools/remote-build.sh status | stop     inspect / cancel the running build
#
# Host + credentials come from <repo>/.env (copy .env.example). The LOCAL
# checkout's build.sh is shipped to the box and run there in a detached
# session, so the build survives Ctrl-C, laptop sleep and dropped Wi-Fi.
# Secrets travel over the SSH channel into a 0600 tmpfs file that the remote
# wrapper sources and deletes before the build starts — never argv, never
# shell history, never on disk. See --help for all commands and options.
#
# Must stay bash 3.2 compatible (macOS /bin/bash).
# =============================================================================
set -euo pipefail

# --- askpass mode: OpenSSH re-invokes this script to obtain the password ----
# (SSH_ASKPASS_REQUIRE=force, OpenSSH >= 8.4; no sshpass needed.)
if [ -n "${RPW_ASKPASS_MODE:-}" ]; then
    case "${1:-}" in
        *"(yes/no"*) echo yes ;;
        *) printf '%s\n' "${RPW_ASKPASS_SECRET:-}" ;;
    esac
    exit 0
fi

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO="$(cd "$(dirname "$SELF")/.." && pwd)"
PROG="tools/remote-build.sh"

if [ -t 2 ]; then
    C_B=$'\033[1m'; C_R=$'\033[31m'; C_G=$'\033[32m'; C_Y=$'\033[33m'; C_0=$'\033[0m'
else
    C_B=""; C_R=""; C_G=""; C_Y=""; C_0=""
fi
info() { printf '%s==>%s %s\n' "$C_B" "$C_0" "$*" >&2; }
note() { printf '    %s\n' "$*" >&2; }
warn() { printf '%sWARNING:%s %s\n' "$C_Y" "$C_0" "$*" >&2; }
die()  { printf '%sERROR:%s %s\n' "$C_R" "$C_0" "$1" >&2; exit "${2:-1}"; }

usage() {
    cat <<EOF
Usage: $PROG [options] [command] [args]

Build runpod-workers images on the Hetzner box from your laptop. Reads the
host and all secrets from <repo>/.env (see .env.example).

Commands:
  build [TARGET]     preflight, start the build detached, stream the log (default)
  start [TARGET]     start the build detached and return immediately
  logs | attach      stream the current (or last) build log; Ctrl-C only detaches
  status             what is running, since when, last log lines, disk free
  stop [--force]     cancel the running build (TERM; KILL with --force)
  preflight          check connection, remote tools, disk, and whether a build runs
  models             list build targets
  ssh [CMD...]       shell (or one command) on the box, credentials handled for you
  setup [PUBKEY]     install your SSH public key on the box (to stop using the password)

TARGET is base, all, or a model directory name (see 'models'). A bare TARGET as
the first argument means 'build TARGET'. With no TARGET you get a menu.

Options:
  --env-file PATH        credentials file (default: <repo>/.env)
  --host H               override HETZNER_HOST      --port P    override HETZNER_PORT
  --user U               override HETZNER_USER
  --branch B             branch the box clones and builds (default: your current branch)
  --no-push              build only, do not push (NO_PUSH=1)
  --base-tag T           pin the base image tag (BASE_TAG)
  --namespace NS         Docker Hub namespace (IMAGE_NAMESPACE)
  --comfyui-version V    ComfyUI version for base builds (COMFYUI_VERSION)
  --dry-run              run the whole pipeline, but the box only echoes and sleeps
  --dry-run-seconds N    how long the dry run pretends to build (default 10)
  --no-watch             with build: same as start
  --force                with stop: SIGKILL if TERM does not work within 30 s
  --reset-hostkey        forget the box's SSH host key first (after a reinstall/rescue)
  -y, --yes              skip the confirmation prompt
  -h, --help             this help

.env keys: HETZNER_HOST HETZNER_PORT HETZNER_USER HETZNER_PASSWORD HETZNER_SSH_KEY
           DOCKERHUB_USERNAME DOCKERHUB_TOKEN HUGGINGFACE_ACCESS_TOKEN CIVITAI_TOKEN
           BRANCH IMAGE_NAMESPACE BASE_TAG COMFYUI_VERSION NO_PUSH REPO_URL
EOF
}

# =============================================================================
# Config
# =============================================================================
KNOWN_KEYS="HETZNER_HOST HETZNER_PORT HETZNER_USER HETZNER_PASSWORD HETZNER_SSH_KEY \
DOCKERHUB_USERNAME DOCKERHUB_TOKEN HUGGINGFACE_ACCESS_TOKEN CIVITAI_TOKEN \
BRANCH IMAGE_NAMESPACE BASE_TAG COMFYUI_VERSION NO_PUSH REPO_URL"
# Forwarded to the box (as a 0600 tmpfs env file the runner sources + deletes).
FORWARD_KEYS="RPW_RUN_ID RPW_DRY_RUN MODEL BRANCH REPO_URL IMAGE_NAMESPACE BASE_TAG \
COMFYUI_VERSION NO_PUSH DOCKERHUB_USERNAME DOCKERHUB_TOKEN HUGGINGFACE_ACCESS_TOKEN CIVITAI_TOKEN"

# Start every known key from the process environment (or empty). Precedence:
# CLI flag > process env > .env file.
for k in $KNOWN_KEYS; do eval "$k=\"\${$k:-}\""; done
MODEL=""; RPW_RUN_ID=""; RPW_DRY_RUN=""

is_known_key() { case " $KNOWN_KEYS " in *" $1 "*) return 0 ;; esac; return 1; }
var() { eval "printf '%s' \"\${$1:-}\""; }

# KEY=VALUE parser (no `source`, so $ and backticks in tokens are never expanded).
# Lines: optional 'export ', '#' comments, values optionally "quoted"/'quoted'.
load_env_file() {
    local f="$1" line key val perms
    [ -f "$f" ] || return 0
    perms="$(stat -f '%Lp' "$f" 2>/dev/null || stat -c '%a' "$f" 2>/dev/null || echo "")"
    case "$perms" in ""|*00) ;; *) warn "$f is mode $perms and holds secrets; run: chmod 600 $f" ;; esac
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line#"${line%%[![:space:]]*}"}"
        case "$line" in ''|'#'*) continue ;; esac
        line="${line#export }"
        case "$line" in *=*) ;; *) warn "$f: ignoring line without '=': ${line%% *}"; continue ;; esac
        key="${line%%=*}"; val="${line#*=}"
        key="${key%"${key##*[![:space:]]}"}"
        case "$key" in ''|*[!A-Za-z0-9_]*) warn "$f: ignoring malformed key '$key'"; continue ;; esac
        val="${val%"${val##*[![:space:]]}"}"
        val="${val#"${val%%[![:space:]]*}"}"
        case "$val" in
            \"*\") val="${val#\"}"; val="${val%\"}" ;;
            \'*\') val="${val#\'}"; val="${val%\'}" ;;
        esac
        if ! is_known_key "$key"; then warn "$f: unknown key '$key' ignored"; continue; fi
        if [ -z "$(var "$key")" ]; then eval "$key=\$val"; fi
    done < "$f"
}

# =============================================================================
# Targets
# =============================================================================
list_targets() {
    echo base; echo all
    local d
    for d in "$REPO"/models/*/; do [ -f "$d/model.yaml" ] && basename "$d"; done
    return 0
}
is_target() { local t; for t in $(list_targets); do [ "$t" = "$1" ] && return 0; done; return 1; }

choose_target() {
    [ -t 0 ] || die "no TARGET given and stdin is not a terminal (pass one: $PROG build <target>)" 2
    local targets=() t x
    for t in $(list_targets); do targets+=("$t"); done
    printf '%sSelect a build target%s (number or name, Ctrl-D to abort):\n' "$C_B" "$C_0" >&2
    PS3="Target [1-${#targets[@]}]: "
    select t in "${targets[@]}"; do
        if [ -n "$t" ]; then MODEL="$t"; break; fi
        for x in "${targets[@]}"; do
            if [ "$REPLY" = "$x" ]; then MODEL="$x"; break 2; fi
        done
        echo "invalid choice: $REPLY" >&2
    done
    echo >&2
    [ -n "$MODEL" ] || die "no target selected" 2
}

# =============================================================================
# SSH plumbing
# =============================================================================
SSH_OPTS=()
AUTH_MODE=""

setup_ssh() {
    [ -n "$HETZNER_HOST" ] || die "HETZNER_HOST is not set (put it in $ENV_FILE or pass --host)" 2
    mkdir -p "$HOME/.ssh"; chmod 700 "$HOME/.ssh" 2>/dev/null || true
    SSH_OPTS=(
        -p "$HETZNER_PORT"
        -o StrictHostKeyChecking=accept-new
        -o ConnectTimeout=15
        -o ServerAliveInterval=15
        -o ServerAliveCountMax=4
        -o LogLevel=ERROR
        -o ControlMaster=auto
        -o ControlPath="$HOME/.ssh/rpw-%C"
        -o ControlPersist=600
    )
    if [ -n "$HETZNER_SSH_KEY" ]; then
        local key="${HETZNER_SSH_KEY/#\~/$HOME}"
        [ -r "$key" ] || die "HETZNER_SSH_KEY=$HETZNER_SSH_KEY is not readable" 2
        SSH_OPTS+=(-i "$key" -o IdentitiesOnly=yes -o BatchMode=yes)
        AUTH_MODE="key $key"
    elif [ -n "$HETZNER_PASSWORD" ]; then
        [ -x "$SELF" ] || die "$SELF must be executable for password auth (run: chmod +x $SELF)" 2
        SSH_OPTS+=(-o PreferredAuthentications=password,keyboard-interactive
                   -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1)
        AUTH_MODE="password from $ENV_FILE"
    else
        die "set HETZNER_PASSWORD or HETZNER_SSH_KEY in $ENV_FILE" 2
    fi
}

# ssh to the box. The password is handed to OpenSSH through the askpass hook,
# scoped to this one invocation (never exported globally, never in argv).
rssh() {
    if [ -n "$HETZNER_SSH_KEY" ]; then
        ssh "${SSH_OPTS[@]}" "$HETZNER_USER@$HETZNER_HOST" "$@"
    else
        SSH_ASKPASS="$SELF" SSH_ASKPASS_REQUIRE=force RPW_ASKPASS_MODE=1 \
        RPW_ASKPASS_SECRET="$HETZNER_PASSWORD" \
            ssh "${SSH_OPTS[@]}" "$HETZNER_USER@$HETZNER_HOST" "$@"
    fi
}

explain_ssh_error() {
    local rc="$1" err="$2"
    [ -z "$err" ] || printf '%s\n' "$err" >&2
    case "$err" in
        *"HOST IDENTIFICATION HAS CHANGED"*|*"Host key verification failed"*)
            die "the SSH host key of $HETZNER_HOST changed (reinstalled or rescued box?). If that is expected, rerun with --reset-hostkey" ;;
        *"Permission denied"*)
            if [ -n "$HETZNER_SSH_KEY" ]; then
                die "$HETZNER_HOST rejected the key $HETZNER_SSH_KEY. Check the box's ~/.ssh/authorized_keys"
            else
                die "$HETZNER_HOST rejected the password. Check HETZNER_PASSWORD in $ENV_FILE. If the message lists only 'publickey', password login is disabled on the box: set HETZNER_SSH_KEY=~/.ssh/id_ed25519 instead"
            fi ;;
        *"Connection refused"*|*"timed out"*|*"No route to host"*|*"Could not resolve"*|*"Network is unreachable"*)
            die "cannot reach $HETZNER_HOST:$HETZNER_PORT. Is the server up and HETZNER_HOST/HETZNER_PORT right?" ;;
        *)  die "ssh to $HETZNER_USER@$HETZNER_HOST failed (exit $rc)" ;;
    esac
}

# Run a remote bash script given on stdin. stdout -> $REMOTE_OUT. An ssh-level
# failure (255) is explained and fatal; the script's own exit code is returned.
REMOTE_OUT=""
remote_capture() {
    local err rc=0
    err="$(mktemp)"
    REMOTE_OUT="$(rssh bash -s 2>"$err")" || rc=$?
    if [ "$rc" -eq 255 ]; then explain_ssh_error "$rc" "$(cat "$err")"; fi
    [ -s "$err" ] && cat "$err" >&2
    rm -f "$err"
    return "$rc"
}

field() { printf '%s\n' "$1" | sed -n "s/^$2=//p" | head -n 1; }

reset_hostkey() {
    ssh-keygen -R "$HETZNER_HOST" >/dev/null 2>&1 || true
    ssh-keygen -R "[$HETZNER_HOST]:$HETZNER_PORT" >/dev/null 2>&1 || true
    ssh -o ControlPath="$HOME/.ssh/rpw-%C" -p "$HETZNER_PORT" -O exit \
        "$HETZNER_USER@$HETZNER_HOST" >/dev/null 2>&1 || true
    info "Forgot the SSH host key for $HETZNER_HOST (it will be re-accepted on the next connection)."
}

# =============================================================================
# Branch checks (the box clones origin/$BRANCH; build.sh itself comes from here)
# =============================================================================
BRANCH_WARNINGS=()

resolve_branch() {
    if [ -z "$BRANCH" ]; then
        BRANCH="$(git -C "$REPO" branch --show-current 2>/dev/null || true)"
        [ -n "$BRANCH" ] || BRANCH=main
    fi
}

check_branch() {
    if ! git -C "$REPO" fetch -q origin "refs/heads/$BRANCH" 2>/dev/null; then
        die "branch '$BRANCH' does not exist on origin (push it first, or pass --branch main)" 2
    fi
    local local_branch ahead
    local_branch="$(git -C "$REPO" branch --show-current 2>/dev/null || true)"
    if [ "$local_branch" = "$BRANCH" ]; then
        ahead="$(git -C "$REPO" rev-list --count FETCH_HEAD..HEAD 2>/dev/null || echo 0)"
        [ "$ahead" = "0" ] || BRANCH_WARNINGS+=("local $BRANCH is $ahead commit(s) ahead of origin/$BRANCH; push first or the box builds the older commit")
        [ -z "$(git -C "$REPO" status --porcelain -- models base schema 2>/dev/null)" ] \
            || BRANCH_WARNINGS+=("uncommitted changes under models/, base/ or schema/ are NOT built (the box clones origin/$BRANCH)")
    fi
    git -C "$REPO" diff --quiet HEAD -- build.sh 2>/dev/null \
        || BRANCH_WARNINGS+=("build.sh has local modifications; the box will run this modified copy")
    case "$MODEL" in
        base|all) ;;
        *) git -C "$REPO" cat-file -e "FETCH_HEAD:models/$MODEL/model.yaml" 2>/dev/null \
            || BRANCH_WARNINGS+=("models/$MODEL/model.yaml is not on origin/$BRANCH; the build will fail on the box") ;;
    esac
}

# =============================================================================
# Preflight
# =============================================================================
PRE_STATE=""; PRE_META=""

run_preflight() {
    info "Connecting to $HETZNER_USER@$HETZNER_HOST:$HETZNER_PORT ($AUTH_MODE)..."
    remote_capture <<'EOF' || die "preflight failed on $HETZNER_HOST"
set -u
RPW="$HOME/.runpod-workers-remote"; mkdir -p "$RPW/logs"
missing=""
for t in git curl flock setsid tail df python3; do
    command -v "$t" >/dev/null 2>&1 || missing="$missing $t"
done
# Need tail -F/--pid (GNU coreutils or the uutils rewrite on Ubuntu >= 25.10).
tail --help 2>&1 | grep -q -- '--pid' || missing="$missing tail(--pid)"
py=""
python3 -c 'import yaml' 2>/dev/null || py="$py python3-yaml"
python3 -c 'import jsonschema' 2>/dev/null || py="$py python3-jsonschema"
if [ -n "$py" ] && command -v apt-get >/dev/null 2>&1; then
    echo "installing=$py"
    export DEBIAN_FRONTEND=noninteractive
    apt-get install -y -q $py </dev/null >/dev/null 2>&1 \
        || { apt-get update -q </dev/null >/dev/null 2>&1; apt-get install -y -q $py </dev/null >/dev/null 2>&1; } \
        || true
    py=""
    python3 -c 'import yaml' 2>/dev/null || py="$py python3-yaml"
    python3 -c 'import jsonschema' 2>/dev/null || py="$py python3-jsonschema"
fi
[ -z "$py" ] || missing="$missing$py"
echo "missing=${missing# }"
echo "os=$( { . /etc/os-release 2>/dev/null && printf '%s' "$PRETTY_NAME"; } || uname -s)"
echo "kernel=$(uname -r)"
echo "uptime=$(uptime -p 2>/dev/null || uptime 2>/dev/null || echo "uptime n/a")"
echo "docker=$(docker --version 2>/dev/null || echo 'not installed (build.sh installs it)')"
echo "disk=$( { df -h /var/lib/docker 2>/dev/null || df -h /; } | awk 'NR==2{print $4" free of "$2" ("$5" used) on "$6}')"
if flock -n "$RPW/lock" true 2>/dev/null; then
    echo "state=idle"
else
    echo "state=busy"
    tr '\n' ' ' < "$RPW/build.meta" 2>/dev/null | sed 's/^/meta=/'; echo
fi
EOF
    local out="$REMOTE_OUT" missing installing
    note "box:     $(field "$out" os) ($(field "$out" kernel)), $(field "$out" uptime)"
    note "docker:  $(field "$out" docker)"
    note "disk:    $(field "$out" disk)"
    installing="$(field "$out" installing)"
    [ -z "$installing" ] || note "installed missing python packages:$installing"
    missing="$(field "$out" missing)"
    [ -z "$missing" ] || die "the box is missing required tools: $missing  (install them with: $PROG ssh apt-get install -y ...)"
    PRE_STATE="$(field "$out" state)"
    PRE_META="$(field "$out" meta)"
    if [ "$PRE_STATE" = busy ]; then
        note "state:   ${C_Y}a build is RUNNING${C_0} ($PRE_META)"
    else
        note "state:   idle"
    fi
}

# =============================================================================
# Start: ship build.sh + runner + secrets over stdin, launch detached
# =============================================================================

# The wrapper that runs detached on the box. Sources the tmpfs env file,
# deletes it, records its process group (what `stop` kills and `tail --pid`
# follows), runs build.sh and writes the exit sentinel.
emit_runner() {
    cat <<'RUNNER'
#!/usr/bin/env bash
# AUTO-GENERATED by tools/remote-build.sh — detached build wrapper. Do not edit.
set -uo pipefail     # no -e on purpose: build.sh's exit code must be captured
ENVF="$1"
RPW="$HOME/.runpod-workers-remote"
trap 'rm -f "$ENVF"' EXIT
set -a; . "$ENVF"; set +a
rm -f "$ENVF"
export BUILDKIT_PROGRESS=plain DOCKER_CLI_HINTS=false
pgid="$(sed 's/^.*) //' "/proc/$$/stat" 2>/dev/null | cut -d' ' -f3)"
[ -n "$pgid" ] || pgid="$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')"
printf '%s\n' "$pgid" > "$RPW/build.pid"
printf 'run_id=%s\nmodel=%s\nbranch=%s\nstarted=%s\npgid=%s\n' \
    "${RPW_RUN_ID:-?}" "${MODEL:-?}" "${BRANCH:-main}" "$(date -u +%s)" "$pgid" > "$RPW/build.meta"
echo "__RPW_START__ run=${RPW_RUN_ID:-?} model=${MODEL:-?} branch=${BRANCH:-main} host=${HOSTNAME:-?} pgid=${pgid} epoch=$(date -u +%s) at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ -n "${RPW_DRY_RUN:-}" ]; then
    echo "[dry-run] would run build.sh with MODEL=${MODEL:-?} BRANCH=${BRANCH:-main} NO_PUSH=${NO_PUSH:-} IMAGE_NAMESPACE=${IMAGE_NAMESPACE:-} BASE_TAG=${BASE_TAG:-}"
    echo "[dry-run] secrets received:${DOCKERHUB_TOKEN:+ dockerhub}${HUGGINGFACE_ACCESS_TOKEN:+ huggingface}${CIVITAI_TOKEN:+ civitai}"
    i=0
    while [ "$i" -lt "${RPW_DRY_RUN}" ]; do
        i=$((i + 1)); echo "[dry-run] tick $i/${RPW_DRY_RUN}"; sleep 1
    done
    rc=0
else
    bash "$RPW/build.sh"
    rc=$?
    docker logout >/dev/null 2>&1 || true
fi
echo "__RPW_EXIT__=${rc}"
rm -f "$RPW/build.pid"
exit "$rc"
RUNNER
}

# Emits the launcher script that is streamed to `bash -s` on the box. Only
# generated identifiers (hex delimiter, run id) are interpolated; every token
# is inside a single-quoted heredoc, and values are %q-quoted for `source`.
generate_start_script() {
    local d k v
    d="RPW$(od -An -N4 -tx1 /dev/urandom | tr -d ' \n')"
    cat <<'PART1'
set -u; umask 077
RPW="$HOME/.runpod-workers-remote"
mkdir -p "$RPW/logs" /dev/shm/rpw || exit 1
flock -n "$RPW/lock" true 2>/dev/null || { echo __RPW_BUSY__; exit 3; }
PART1
    printf 'cat > "$RPW/build.sh" <<'"'"'%sA'"'"'\n' "$d"
    cat "$REPO/build.sh"
    printf '\n%sA\n' "$d"
    printf 'cat > "$RPW/runner.sh" <<'"'"'%sB'"'"'\n' "$d"
    emit_runner
    printf '\n%sB\n' "$d"
    printf 'ENVF="/dev/shm/rpw/env.%s"\n' "$RPW_RUN_ID"
    printf 'cat > "$ENVF" <<'"'"'%sC'"'"'\n' "$d"
    for k in $FORWARD_KEYS; do
        v="$(var "$k")"
        [ -z "$v" ] || printf 'export %s=%q\n' "$k" "$v"
    done
    printf '%sC\n' "$d"
    printf 'LOG="$RPW/logs/%s.log"\n' "$RPW_RUN_ID"
    printf 'ln -sfn "logs/%s.log" "$RPW/current.log"\n' "$RPW_RUN_ID"
    cat <<'PART2'
: > "$LOG"
rm -f "$RPW/build.pid"
setsid flock -n "$RPW/lock" bash "$RPW/runner.sh" "$ENVF" >"$LOG" 2>&1 </dev/null &
i=0
while [ "$i" -lt 50 ]; do
    [ -s "$RPW/build.pid" ] && break
    sleep 0.1; i=$((i + 1))
done
if [ -s "$RPW/build.pid" ]; then
    echo "__RPW_LAUNCHED__ pgid=$(cat "$RPW/build.pid")"
else
    echo "__RPW_LAUNCH_FAILED__"
    rm -f "$ENVF"
    tail -n 20 "$LOG" 2>/dev/null
    exit 1
fi
PART2
}

start_build() {
    RPW_RUN_ID="$(date -u +%Y%m%d-%H%M%S)-${MODEL}"
    RPW_DRY_RUN=""; [ -z "$DRY_RUN" ] || RPW_DRY_RUN="$DRY_RUN_SECONDS"
    info "Starting build '$MODEL' on $HETZNER_HOST (run $RPW_RUN_ID)..."
    local rc=0
    # process substitution, not a pipeline: remote_capture must run in this shell
    remote_capture < <(generate_start_script) || rc=$?
    case "$REMOTE_OUT" in
        *__RPW_BUSY__*) die "a build is already running on $HETZNER_HOST. Attach: $PROG logs   Cancel: $PROG stop" 3 ;;
        *__RPW_LAUNCHED__*) ;;
        *) printf '%s\n' "$REMOTE_OUT" >&2; die "could not start the build on $HETZNER_HOST (exit $rc)" ;;
    esac
    [ "$rc" -eq 0 ] || die "could not start the build on $HETZNER_HOST (exit $rc)"
    local pg
    pg="$(printf '%s\n' "$REMOTE_OUT" | sed -n 's/^__RPW_LAUNCHED__ //p' | head -n 1)"
    note "started detached ($pg); log on the box: ~/.runpod-workers-remote/logs/$RPW_RUN_ID.log"
}

print_detach_hints() {
    note "reattach: $PROG logs"
    note "status:   $PROG status"
    note "cancel:   $PROG stop"
}

# =============================================================================
# Watch: stream the log; Ctrl-C detaches; reconnect on drops; exit with rc
# =============================================================================
TAG_RE='^[[:space:]]+[A-Za-z0-9._-]+/[A-Za-z0-9._-]+:[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{4}-[0-9a-f]+[[:space:]]*$'

fmt_duration() {
    local s="$1"
    if [ "$s" -ge 3600 ]; then printf '%dh%02dm' $((s / 3600)) $((s % 3600 / 60))
    elif [ "$s" -ge 60 ]; then printf '%dm%02ds' $((s / 60)) $((s % 60))
    else printf '%ds' "$s"; fi
}

on_detach() {
    trap - INT
    printf '\n%s[detached]%s the build keeps running on %s.\n' "$C_Y" "$C_0" "$HETZNER_HOST" >&2
    print_detach_hints
    exit 130
}

remote_tail_script() {   # $1 = first line number to send (1 = whole log)
    cat <<EOF
RPW="\$HOME/.runpod-workers-remote"
[ -e "\$RPW/current.log" ] || { echo __RPW_NOLOG__; exit 0; }
pid="\$(cat "\$RPW/build.pid" 2>/dev/null || true)"
if [ -n "\$pid" ] && kill -0 -- "-\$pid" 2>/dev/null; then
    exec tail -n +$1 -F --pid="\$pid" "\$RPW/current.log" 2>/dev/null
else
    exec tail -n +$1 "\$RPW/current.log"
fi
EOF
}

remote_state() {   # prints running | idle | unreachable (never fatal)
    local out
    out="$(rssh bash -s 2>/dev/null <<'EOF'
RPW="$HOME/.runpod-workers-remote"; mkdir -p "$RPW"
if flock -n "$RPW/lock" true 2>/dev/null; then echo idle; else echo running; fi
EOF
)" || out=""
    echo "${out:-unreachable}"
}

run_watch() {
    local count=0 rc="" attempts=0 final_replay=0 tags="" line t start_epoch="" state got delay
    info "Streaming the build log from $HETZNER_HOST (Ctrl-C detaches; the build keeps running)"
    trap on_detach INT
    while :; do
        got=0
        while IFS= read -r line; do
            count=$((count + 1)); got=1
            case "$line" in
                __RPW_EXIT__=*) rc="${line#__RPW_EXIT__=}" ;;
                __RPW_NOLOG__)  trap - INT; die "no build has ever been run from this tool on $HETZNER_HOST" ;;
                __RPW_START__*)
                    start_epoch="${line##*epoch=}"; start_epoch="${start_epoch%% *}"
                    printf '%s[remote]%s %s\n' "$C_B" "$C_0" "${line#__RPW_START__ }" ;;
                *)
                    printf '%s\n' "$line"
                    if [[ "$line" =~ $TAG_RE ]]; then
                        t="${line//[[:space:]]/}"
                        case " $tags " in *" $t "*) ;; *) tags="$tags $t" ;; esac
                    fi ;;
            esac
        done < <(rssh bash -s <<<"$(remote_tail_script "$((count + 1))")")
        [ -z "$rc" ] || break
        # Stream ended without the exit sentinel: connection drop, or the build died.
        state="$(remote_state)"
        case "$state" in
            idle)
                if [ "$final_replay" = 1 ]; then
                    trap - INT
                    die "the build on $HETZNER_HOST ended without an exit marker (OOM kill? reboot?). Inspect with: $PROG status"
                fi
                final_replay=1 ;;
            *)
                [ "$got" = 0 ] || attempts=0
                attempts=$((attempts + 1))
                if [ "$attempts" -gt 40 ]; then
                    trap - INT
                    warn "lost the connection to $HETZNER_HOST; the build is still running there."
                    print_detach_hints
                    exit 130
                fi
                delay=$(( attempts < 5 ? attempts * 3 : 15 ))
                warn "connection to $HETZNER_HOST dropped ($state); reconnecting in ${delay}s (attempt $attempts)..."
                sleep "$delay" ;;
        esac
    done
    trap - INT
    local elapsed=""
    [ -z "$start_epoch" ] || elapsed=" in $(fmt_duration $(( $(date +%s) - start_epoch )))"
    echo "============================================="
    if [ "$rc" = 0 ]; then
        printf ' %s✓ Build finished%s (exit 0)%s\n' "$C_G" "$C_0" "$elapsed"
    else
        printf ' %s✗ Build FAILED%s (exit %s)%s\n' "$C_R" "$C_0" "$rc" "$elapsed"
    fi
    for t in $tags; do echo "     image: $t"; done
    echo "============================================="
    return "$rc"
}

# =============================================================================
# status / stop / ssh / setup
# =============================================================================
cmd_status() {
    remote_capture <<'EOF' || die "could not query $HETZNER_HOST"
RPW="$HOME/.runpod-workers-remote"; mkdir -p "$RPW"
[ ! -f "$RPW/build.meta" ] || sed 's/^/meta_/' "$RPW/build.meta"
if flock -n "$RPW/lock" true 2>/dev/null; then echo state=idle; else echo state=running; fi
echo "log=$(readlink "$RPW/current.log" 2>/dev/null || true)"
echo "lines=$(wc -l < "$RPW/current.log" 2>/dev/null | tr -d ' ' || true)"
echo "exit=$(grep -m1 '^__RPW_EXIT__=' "$RPW/current.log" 2>/dev/null | cut -d= -f2 || true)"
echo "disk=$( { df -h /var/lib/docker 2>/dev/null || df -h /; } | awk 'NR==2{print $4" free of "$2" ("$5" used) on "$6}')"
echo "__TAIL__"
tail -n 8 "$RPW/current.log" 2>/dev/null || true
EOF
    local out="$REMOTE_OUT" state started elapsed=""
    state="$(field "$out" state)"
    started="$(field "$out" meta_started)"
    [ -z "$started" ] || elapsed="$(fmt_duration $(( $(date +%s) - started )))"
    printf '%sBuild box%s %s@%s\n' "$C_B" "$C_0" "$HETZNER_USER" "$HETZNER_HOST"
    if [ "$state" = running ]; then
        printf '  state:    %sRUNNING%s for %s (pgid %s)\n' "$C_G" "$C_0" "$elapsed" "$(field "$out" meta_pgid)"
    elif [ -n "$(field "$out" meta_run_id)" ]; then
        printf '  state:    idle (last build exited %s, started %s ago)\n' "$(field "$out" exit)" "$elapsed"
    else
        printf '  state:    idle (no build has been run from this tool yet)\n'
    fi
    [ -z "$(field "$out" meta_run_id)" ] || {
        printf '  target:   %s   branch: %s\n' "$(field "$out" meta_model)" "$(field "$out" meta_branch)"
        printf '  run:      %s   (%s log lines)\n' "$(field "$out" meta_run_id)" "$(field "$out" lines)"
    }
    printf '  disk:     %s\n' "$(field "$out" disk)"
    if printf '%s\n' "$out" | grep -q '^__TAIL__$'; then
        printf '  %s--- last log lines ---%s\n' "$C_B" "$C_0"
        printf '%s\n' "$out" | sed -n '/^__TAIL__$/,$p' | sed '1d; s/^/  | /'
    fi
}

cmd_stop() {
    info "Stopping the build on $HETZNER_HOST..."
    local force="${FORCE:-0}"
    remote_capture <<EOF || true
RPW="\$HOME/.runpod-workers-remote"
pid="\$(cat "\$RPW/build.pid" 2>/dev/null || true)"
if [ -z "\$pid" ] || ! kill -0 -- "-\$pid" 2>/dev/null; then echo __RPW_NOTRUNNING__; exit 0; fi
kill -TERM -- "-\$pid" 2>/dev/null || true
i=0
while [ "\$i" -lt 30 ] && kill -0 -- "-\$pid" 2>/dev/null; do sleep 1; i=\$((i + 1)); done
if kill -0 -- "-\$pid" 2>/dev/null; then
    if [ "$force" = 1 ]; then kill -KILL -- "-\$pid" 2>/dev/null || true; sleep 1
    else echo __RPW_STILLRUNNING__; exit 0; fi
fi
echo "[stopped by $PROG at \$(date -u +%Y-%m-%dT%H:%M:%SZ)]" >> "\$RPW/current.log"
echo "__RPW_EXIT__=143" >> "\$RPW/current.log"
rm -f "\$RPW/build.pid"
echo __RPW_STOPPED__
EOF
    case "$REMOTE_OUT" in
        *__RPW_NOTRUNNING__*)   note "nothing is running." ;;
        *__RPW_STILLRUNNING__*) die "the build ignored SIGTERM for 30 s. Rerun with: $PROG stop --force" ;;
        *__RPW_STOPPED__*)      note "stopped. Docker Hub drops any half-pushed layers on its own." ;;
        *) die "unexpected reply from $HETZNER_HOST: $REMOTE_OUT" ;;
    esac
}

cmd_setup() {
    local pub="${ARGS[0]:-$HOME/.ssh/id_ed25519.pub}"
    pub="${pub/#\~/$HOME}"
    [ -f "$pub" ] || die "public key not found: $pub  (generate one with: ssh-keygen -t ed25519)" 2
    info "Installing $pub into $HETZNER_USER@$HETZNER_HOST:~/.ssh/authorized_keys"
    rssh 'umask 077; mkdir -p ~/.ssh; k="$(cat)"; touch ~/.ssh/authorized_keys;
          grep -qxF "$k" ~/.ssh/authorized_keys || printf "%s\n" "$k" >> ~/.ssh/authorized_keys; echo ok' < "$pub" \
        | grep -q '^ok$' || die "could not install the key on $HETZNER_HOST"
    note "done. To stop using the password, set in $ENV_FILE:  HETZNER_SSH_KEY=${pub%.pub}"
}

# =============================================================================
# Build orchestration
# =============================================================================
yn() { if [ -n "$1" ]; then echo set; else echo "${C_Y}MISSING${C_0}"; fi; }

print_summary() {
    local push="yes" mode="real build" w
    [ -z "$NO_PUSH" ] || push="no (NO_PUSH=1, build only)"
    [ -z "$DRY_RUN" ] || mode="${C_Y}DRY RUN${C_0} (${DRY_RUN_SECONDS}s of fake work, nothing is built)"
    printf '\n%sRemote build%s\n' "$C_B" "$C_0" >&2
    note "host:       $HETZNER_USER@$HETZNER_HOST:$HETZNER_PORT  [$AUTH_MODE]"
    note "target:     $MODEL"
    note "branch:     $BRANCH"
    note "namespace:  ${IMAGE_NAMESPACE:-jmendapara}"
    [ "$MODEL" = base ] || note "base tag:   ${BASE_TAG:-auto-discover newest on Docker Hub}"
    [ "$MODEL" != base ] && [ "$MODEL" != all ] || note "comfyui:    ${COMFYUI_VERSION:-latest}"
    note "push:       $push"
    note "secrets:    dockerhub=$(yn "$DOCKERHUB_TOKEN")  huggingface=$(yn "$HUGGINGFACE_ACCESS_TOKEN")  civitai=$(yn "$CIVITAI_TOKEN")"
    note "mode:       $mode"
    for w in ${BRANCH_WARNINGS[@]+"${BRANCH_WARNINGS[@]}"}; do warn "$w"; done
    echo >&2
}

confirm() {
    [ -z "$YES" ] || return 0
    [ -t 0 ] || die "stdin is not a terminal; pass -y to skip the confirmation" 2
    local a
    read -r -p "Proceed? [Y/n] " a || a=n
    case "$a" in ''|y|Y|yes|YES) ;; *) die "aborted" 130 ;; esac
}

cmd_build() {
    MODEL="${ARGS[0]:-}"
    [ -n "$MODEL" ] || choose_target
    is_target "$MODEL" || die "unknown target '$MODEL'. See: $PROG models" 2
    if [ -z "$DRY_RUN" ] && { [ -z "$DOCKERHUB_USERNAME" ] || [ -z "$DOCKERHUB_TOKEN" ]; }; then
        die "DOCKERHUB_USERNAME and DOCKERHUB_TOKEN must be set in $ENV_FILE (build.sh logs into Docker Hub with them)" 2
    fi
    resolve_branch
    check_branch
    run_preflight
    print_summary
    if [ "$PRE_STATE" = busy ]; then
        warn "a build is already running on $HETZNER_HOST — only one at a time."
        if [ -n "$YES" ] || [ ! -t 0 ]; then
            die "not starting a second build. Attach: $PROG logs   Cancel: $PROG stop" 3
        fi
        local a
        read -r -p "Attach to the running build instead? [Y/n] " a || a=n
        case "$a" in
            ''|y|Y|yes|YES) run_watch || exit $?; exit 0 ;;
            *) die "aborted. Cancel the running build with: $PROG stop" 3 ;;
        esac
    fi
    confirm
    start_build
    if [ -n "$NO_WATCH" ] || [ "$COMMAND" = start ]; then
        print_detach_hints
        exit 0
    fi
    run_watch || exit $?
}

# =============================================================================
# Main
# =============================================================================
# RPW_LIBRARY_MODE=1 lets a test harness `source` this file for its functions.
if [ -n "${RPW_LIBRARY_MODE:-}" ]; then return 0 2>/dev/null || exit 0; fi

ENV_FILE="${RPW_ENV_FILE:-$REPO/.env}"
CLI_HOST=""; CLI_PORT=""; CLI_USER=""; CLI_BRANCH=""; CLI_NO_PUSH=""
CLI_BASE_TAG=""; CLI_NAMESPACE=""; CLI_COMFYUI=""
DRY_RUN=""; DRY_RUN_SECONDS=10; NO_WATCH=""; FORCE=""; RESET_HOSTKEY=""; YES=""
POSITIONAL=()

need_arg() { [ $# -ge 2 ] && [ -n "$2" ] || die "option $1 needs a value" 2; }

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)          usage; exit 0 ;;
        --env-file)         need_arg "$@"; ENV_FILE="$2"; shift 2 ;;
        --host)             need_arg "$@"; CLI_HOST="$2"; shift 2 ;;
        --port)             need_arg "$@"; CLI_PORT="$2"; shift 2 ;;
        --user)             need_arg "$@"; CLI_USER="$2"; shift 2 ;;
        --branch)           need_arg "$@"; CLI_BRANCH="$2"; shift 2 ;;
        --no-push)          CLI_NO_PUSH=1; shift ;;
        --base-tag)         need_arg "$@"; CLI_BASE_TAG="$2"; shift 2 ;;
        --namespace)        need_arg "$@"; CLI_NAMESPACE="$2"; shift 2 ;;
        --comfyui-version)  need_arg "$@"; CLI_COMFYUI="$2"; shift 2 ;;
        --dry-run)          DRY_RUN=1; shift ;;
        --dry-run-seconds)  need_arg "$@"; DRY_RUN=1; DRY_RUN_SECONDS="$2"; shift 2 ;;
        --no-watch)         NO_WATCH=1; shift ;;
        --force)            FORCE=1; shift ;;
        --reset-hostkey)    RESET_HOSTKEY=1; shift ;;
        -y|--yes)           YES=1; shift ;;
        --)                 shift; while [ $# -gt 0 ]; do POSITIONAL+=("$1"); shift; done ;;
        -*)                 die "unknown option: $1 (see --help)" 2 ;;
        *)
            POSITIONAL+=("$1"); shift
            if [ "${POSITIONAL[0]}" = ssh ]; then   # everything after 'ssh' is the remote command
                while [ $# -gt 0 ]; do POSITIONAL+=("$1"); shift; done
            fi ;;
    esac
done

COMMAND="${POSITIONAL[0]:-build}"
ARGS=()
i=1
while [ "$i" -lt "${#POSITIONAL[@]}" ]; do ARGS+=("${POSITIONAL[$i]}"); i=$((i + 1)); done

case "$COMMAND" in
    build|start|logs|attach|status|stop|preflight|models|ssh|setup) ;;
    *)
        if is_target "$COMMAND"; then
            ARGS=("$COMMAND" ${ARGS[@]+"${ARGS[@]}"}); COMMAND=build
        else
            die "unknown command or target '$COMMAND' (see --help, or: $PROG models)" 2
        fi ;;
esac

if [ "$COMMAND" = models ]; then list_targets; exit 0; fi

load_env_file "$ENV_FILE"
[ -z "$CLI_HOST" ]      || HETZNER_HOST="$CLI_HOST"
[ -z "$CLI_PORT" ]      || HETZNER_PORT="$CLI_PORT"
[ -z "$CLI_USER" ]      || HETZNER_USER="$CLI_USER"
[ -z "$CLI_BRANCH" ]    || BRANCH="$CLI_BRANCH"
[ -z "$CLI_NO_PUSH" ]   || NO_PUSH=1
[ -z "$CLI_BASE_TAG" ]  || BASE_TAG="$CLI_BASE_TAG"
[ -z "$CLI_NAMESPACE" ] || IMAGE_NAMESPACE="$CLI_NAMESPACE"
[ -z "$CLI_COMFYUI" ]   || COMFYUI_VERSION="$CLI_COMFYUI"
HETZNER_PORT="${HETZNER_PORT:-22}"
HETZNER_USER="${HETZNER_USER:-root}"

if [ ! -f "$ENV_FILE" ] && [ -z "$HETZNER_HOST" ]; then
    die "no $ENV_FILE found. Create it with:  cp .env.example .env && chmod 600 .env  (then fill it in)" 2
fi

setup_ssh
[ -z "$RESET_HOSTKEY" ] || reset_hostkey

case "$COMMAND" in
    build|start)  cmd_build ;;
    logs|attach)  run_watch || exit $? ;;
    status)       cmd_status ;;
    stop)         cmd_stop ;;
    preflight)    run_preflight ;;
    setup)        cmd_setup ;;
    ssh)
        if [ "${#ARGS[@]}" -eq 0 ]; then rssh -t; else rssh -t "${ARGS[@]}"; fi ;;
esac
