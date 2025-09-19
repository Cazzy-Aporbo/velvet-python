#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  Cazzy Bashwild: The Uncanny Terminal by Cazandra Aporbo. it will be fine. 
#  The most obscure Bash puzzle game I could dream up. Just a random way to learn bash
#  You type real commands. I judge in a sandbox. Win glory. Or /hint. Ha. (!!)
# ─────────────────────────────────────────────────────────────────────────────
set -Eeuo pipefail

# --- tiny style system -------------------------------------------------------
_has_tput() { command -v tput >/dev/null 2>&1; }
if _has_tput; then
  RED=$(tput setaf 1); GRN=$(tput setaf 2); YEL=$(tput setaf 3)
  BLU=$(tput setaf 4); MAG=$(tput setaf 5); CYN=$(tput setaf 6)
  DIM=$(tput dim); BLD=$(tput bold); RST=$(tput sgr0)
else
  RED=;GRN=;YEL=;BLU=;MAG=;CYN=;DIM=;BLD=;RST=
fi

bar() { printf "%s%s%s\n" "$MAG" "──────────────────────────────────────────────────────────────────────────────" "$RST"; }
say() { printf "%b\n" "$*"; }
title(){ bar; say "${CYN}${BLD}$*${RST}"; bar; }

# --- game state --------------------------------------------------------------
SANDBOX="$(mktemp -d -t cazzy-bashwild.XXXXXX)"
SCORE=0
STREAK=0
MIS_TOTAL=0
declare -A SEEN   # tracks which missions have been served
declare -A ENVINFO=([shell]="$BASH" [ver]="${BASH_VERSION:-?}" [host]="$HOSTNAME")

cleanup(){
  trap - SIGINT SIGTERM EXIT
  rm -rf "$SANDBOX" 2>/dev/null || true
}
trap cleanup SIGINT SIGTERM EXIT

# --- HUD ---------------------------------------------------------------------
hud(){
  local elapsed="$(( $(date +%s) - START ))"
  printf "%sScore:%s %s%d%s  %sStreak:%s %s%d%s  %sTime:%s %s%ds%s  %sSandbox:%s %s\n" \
    "$BLD" "$RST" "$GRN" "$SCORE" "$RST" \
    "$BLD" "$RST" "$YEL" "$STREAK" "$RST" \
    "$BLD" "$RST" "$DIM" "$elapsed" "$RST" \
    "$BLD" "$RST" "$DIM$SANDBOX$RST"
}

# --- dialog ------------------------------------------------------------------
intro(){
  title "Cazzy Bashwild: The Uncanny Terminal — by Cazandra Aporbo"
  cat <<'TXT'
You will solve absurdly specific Bash feats inside a sealed playground.
Nothing outside your sandbox is touched. Your weapon: real commands.

Meta-commands (type them alone):
  :hint       → get a nudge (-1 point)
  :skip       → skip this trial (-2 points, resets streak)
  :lshelp     → list a few useful command families
  :reset      → rebuild a fresh sandbox
  :quit       → leave with honor

Grading is automatic; after each command I check the ritual conditions.
TXT
  bar
}

lshelp(){
  cat <<'HINT'
Useful families to keep in mind:
  files/dirs:   pwd, ls -la, tree, touch, mkdir -p, cp -R, mv, rm -rf
  find/grep:    find . -name, -type, -mtime; grep -Rni --color; sed -i 's/x/y/g'
  text pipes:   tr -sc 'A-Za-z' '\n' | sort | uniq -c | sort -nr | head
  archives:     tar -czf out.tgz dir/    ; unzip/zip
  perms/links:  chmod u+x file; ln -s target link
  misc:         diff <(sort a) <(sort b)   # process substitution
HINT
}

# --- utilities ---------------------------------------------------------------
sandbox_reset(){
  rm -rf "$SANDBOX"; mkdir -p "$SANDBOX"
  # Seed playful files
  mkdir -p "$SANDBOX/lab/alpha" "$SANDBOX/lab/beta" "$SANDBOX/logs"
  printf "the quick red fox jumps over the lazy red dog\n" > "$SANDBOX/lab/alpha/story.txt"
  printf "alpha\nbeta\ngamma\nalpha\nbeta\n" > "$SANDBOX/lab/beta/letters.txt"
  printf "oak\npine\nbirch\nmaple\n" > "$SANDBOX/lab/alpha/trees.a"
  printf "pine\nmaple\nfir\n" > "$SANDBOX/lab/beta/trees.b"
  : > "$SANDBOX/logs/app.log"
  (cd "$SANDBOX/logs"; for i in {1..5}; do printf "log-$i\n" > "trace$i.log"; done)
  # age two traces for a time-travel find puzzle
  (touch -d "2 days ago" "$SANDBOX/logs/trace4.log" || touch -t 202001010101 "$SANDBOX/logs/trace4.log") 2>/dev/null || true
  (touch -d "3 days ago" "$SANDBOX/logs/trace5.log" || touch -t 202001020101 "$SANDBOX/logs/trace5.log") 2>/dev/null || true
}

run_user_cmd(){
  local cmd="$1"
  # Execute inside sandbox, with a minimal PATH and safe IFS
  ( IFS=$' \t\n'; cd "$SANDBOX"; PATH="/usr/bin:/bin:/usr/sbin:/sbin"; bash -O extglob -O nocasematch -c "$cmd" ) \
    1>"$SANDBOX/.stdout" 2>"$SANDBOX/.stderr" || true
}

# judge helpers
file_has(){ grep -qE "$2" "$1" 2>/dev/null; }
exists(){ [ -e "$1" ]; }
is_exec(){ [ -x "$1" ]; }
is_link_to(){ [ -L "$1" ] && [ "$(readlink "$1")" = "$2" ]; }
archive_contains(){ tar -tf "$1" 2>/dev/null | grep -qxF "$2"; }
same_sorted(){ diff <(sort -u "$1") <(sort -u "$2") >/dev/null 2>&1; }

# --- mission registry --------------------------------------------------------
# Each mission is a function whose name starts with M_, sets:
#   GOAL (one-line), HINT (one-line), CHECK (function returning 0/1)
# and prepares any initial files it needs inside $SANDBOX.

M_sed_inplace(){
  GOAL="Transfigure the fox: in lab/alpha/story.txt, replace 'red' with 'blue' everywhere, editing in place."
  HINT="sed -i 's/red/blue/g' lab/alpha/story.txt  # mind BSD vs GNU sed"
  CHECK(){ file_has "$SANDBOX/lab/alpha/story.txt" '^the quick blue fox .* blue dog$'; }
}

M_wordfreq_top3(){
  GOAL="From lab/beta/letters.txt, compute top 3 words (freq then word) into file top3.txt."
  HINT="tr -sc 'A-Za-z' '\\n' < file | sort | uniq -c | sort -nr | head -n 3 > top3.txt"
  CHECK(){ exists "$SANDBOX/top3.txt" && wc -l < "$SANDBOX/top3.txt" | grep -qx '3'; }
}

M_find_older_delete(){
  GOAL="Inside logs/, delete ONLY files older than 1 day. Leave newer traces untouched."
  HINT="find logs -type f -mtime +1 -delete"
  CHECK(){ ! exists "$SANDBOX/logs/trace4.log" && ! exists "$SANDBOX/logs/trace5.log" && exists "$SANDBOX/logs/trace1.log"; }
}

M_tar_logs(){
  GOAL="Archive logs/ into logs.tgz (gzip). Ensure file logs/app.log is included."
  HINT="tar -czf logs.tgz logs"
  CHECK(){ exists "$SANDBOX/logs.tgz" && archive_contains "$SANDBOX/logs.tgz" "logs/app.log"; }
}

M_symlink(){
  GOAL="Create a symbolic link link.story pointing to lab/alpha/story.txt."
  HINT="ln -s lab/alpha/story.txt link.story"
  CHECK(){ is_link_to "$SANDBOX/link.story" "lab/alpha/story.txt"; }
}

M_chmod_exec(){
  GOAL="Create script runme.sh that echoes 'ok' and make it executable."
  HINT="printf '#!/usr/bin/env bash\necho ok\n' > runme.sh && chmod u+x runme.sh && ./runme.sh"
  CHECK(){ is_exec "$SANDBOX/runme.sh"; }
}

M_diff_process_sub(){
  GOAL="Prove that trees.a and trees.b share identical unique sets. Use process substitution with diff; save output to verdict.txt."
  HINT="diff <(sort -u lab/alpha/trees.a) <(sort -u lab/beta/trees.b) > verdict.txt || true"
  CHECK(){ exists "$SANDBOX/verdict.txt"; }
}

M_make_dir_cast(){
  GOAL="Conjure nested dirs arc/{east,west}/gate in one incantation."
  HINT="mkdir -p arc/{east,west}/gate"
  CHECK(){ exists "$SANDBOX/arc/east/gate" && exists "$SANDBOX/arc/west/gate"; }
}

M_move_copy(){
  GOAL="Copy lab/beta/letters.txt to arc/east/letters.copy, then move it to arc/west/letters.moved"
  HINT="cp lab/beta/letters.txt arc/east/letters.copy && mv arc/east/letters.copy arc/west/letters.moved"
  CHECK(){ exists "$SANDBOX/arc/west/letters.moved" && ! exists "$SANDBOX/arc/east/letters.copy"; }
}

M_echo_date_file(){
  GOAL="Write the current date (not the word 'date') into now.txt using command substitution."
  HINT="echo \"$(date)\" > now.txt"
  CHECK(){ file_has "$SANDBOX/now.txt" '^[A-Z][a-z]{2} [A-Z][a-z]{2}'; }
}

M_grep_numbered(){
  GOAL="Search story.txt for 'blue' with line numbers; save output to hits.txt."
  HINT="grep -n 'blue' lab/alpha/story.txt > hits.txt"
  CHECK(){ exists "$SANDBOX/hits.txt" && file_has "$SANDBOX/hits.txt" '^[0-9]+:'; }
}

M_rsync_rename_hint(){
  GOAL="(Trick) Do a fast copy of story.txt to rs.story using rsync (works even if unchanged)."
  HINT="rsync -v lab/alpha/story.txt rs.story"
  CHECK(){ exists "$SANDBOX/rs.story"; }
}

MISSIONS=( sed_inplace wordfreq_top3 find_older_delete tar_logs symlink chmod_exec diff_process_sub make_dir_cast move_copy echo_date_file grep_numbered rsync_rename_hint )

# --- engine ------------------------------------------------------------------
pick_mission(){
  # pick an unseen mission at random; reset if all seen
  if [ "${#SEEN[@]}" -ge "${#MISSIONS[@]}" ]; then
    SEEN=()  # reset cycle
  fi
  local idx
  while :; do
    idx=$(( RANDOM % ${#MISSIONS[@]} ))
    local key="${MISSIONS[$idx]}"
    [[ -n "${SEEN[$key]:-}" ]] || { echo "$key"; return; }
  done
}

render_goal(){
  say "${BLD}${GRN}Trial #$MIS_TOTAL${RST} — ${BLU}$GOAL${RST}"
}

judge(){
  if CHECK; then
    ((SCORE+=3, STREAK++, MIS_TOTAL++))
    say "${GRN}✔ consecrated. +3 points. streak $STREAK.${RST}"
    return 0
  else
    return 1
  fi
}

loop(){
  START="$(date +%s)"
  sandbox_reset
  while :; do
    hud
    local key; key="$(pick_mission)"; SEEN["$key"]=1
    "M_${key}"
    render_goal
    while :; do
      printf "%s\n" "${DIM}[$(date +%H:%M:%S)] in $SANDBOX${RST}"
      read -r -e -p "${YEL}cazzy❯ ${RST}" CMD || { say; say "${RED}bye.${RST}"; exit 0; }
      case "$CMD" in
        ":quit" ) say "${RED}bye.${RST}"; exit 0 ;;
        ":hint" ) say "${DIM}hint:${RST} $HINT"; ((SCORE--)); continue ;;
        ":skip" ) say "${YEL}skipped.${RST}"; ((SCORE-=2)); STREAK=0; MIS_TOTAL=$((MIS_TOTAL+1)); break ;;
        ":lshelp" ) lshelp; continue ;;
        ":reset" ) sandbox_reset; say "${CYN}sandbox rebuilt.${RST}"; continue ;;
        "" ) continue ;;
      esac
      run_user_cmd "$CMD"
      if judge; then break; else
        # print stderr softly to aid debugging
        if [ -s "$SANDBOX/.stderr" ]; then
          say "${DIM}stderr:${RST}"; sed 's/^/  /' "$SANDBOX/.stderr" | tail -n 6
        fi
        say "${RED}✘ not yet. try again or :hint / :skip.${RST}"
      fi
    done
    bar
  done
}

# --- launch ------------------------------------------------------------------
intro
loop
