#!/bin/bash
# [telecom_boed] Stage Qwen3-8B weights from NFS to NODE-LOCAL disk (never mmap weights from /common/users NFS: SIGBUS).
# Run ON a GPU node (it is called by slurm/local_tau2_vllm.sh; can also be run alone). Idempotent, safe to run concurrently:
#   - SRC blobs may be symlinks into a shared xet chunk store (hub/blobs); cp follows them, so DST gets plain files
#   - copies hub/models--Qwen--Qwen3-8B refs + the pinned snapshot + only the blobs that snapshot references
#   - every blob is verified against its name (HF names LFS blobs by sha256, 64 hex; small files by git sha1, 40 hex);
#     a verified blob gets a <blob>.verified marker (size) and is skipped next time; a mismatch deletes the copy and fails
#   - stdout: ONLY the node-local HF_HOME path (progress goes to stderr), so:  export HF_HOME=$(bash stage_weights_local.sh)
# Knobs: SRC_HF_HOME (/filer/tmp1/ss5772/huggingface), DST_HF_HOME (/freespace/local/ss5772/hf_cache), MODEL (Qwen/Qwen3-8B),
#   REVISION (b968826d...), DRY_RUN=1 (copy+verify only blobs <= DRY_MAX_BYTES (1 MiB); for a CPU-side check of the logic
#   against a scratch DST_HF_HOME, e.g.  DRY_RUN=1 DST_HF_HOME=$SCRATCHDIR bash stage_weights_local.sh).
# Inside the GPU hold:  ssh rlab1 "srun --jobid=526576 --overlap --gres=gpu:a100:4 --mem=8G -t 00:20:00 \
#   bash /common/users/ss5772/ddl/telecom_boed/scripts/stage_weights_local.sh"      (needs no GPU; srun form only for node placement
#   on rlab2 - /freespace is node-local, so staging must happen on the node that will run vLLM)
set -euo pipefail
SRC_HF_HOME="${SRC_HF_HOME:-/filer/tmp1/ss5772/huggingface}"
DST_HF_HOME="${DST_HF_HOME:-/freespace/local/ss5772/hf_cache}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
REVISION="${REVISION:-b968826d9c46dd6066d109eabc6255188de91218}"
DRY_MAX_BYTES="${DRY_MAX_BYTES:-1048576}"
REPO="models--${MODEL//\//--}"
S="${SRC_HF_HOME}/hub/${REPO}"; D="${DST_HF_HOME}/hub/${REPO}"
log() { echo "[stage $(date +%T)] $*" >&2; }
[[ -d "${S}/snapshots/${REVISION}" ]] || { echo "source snapshot missing: ${S}/snapshots/${REVISION}" >&2; exit 1; }
mkdir -p "${D}/blobs" "${D}/snapshots" "${D}/refs"
exec 8> "${DST_HF_HOME}/.stage.lock"; flock 8           # serialize concurrent steps on the same node

blob_hash() {  # $1=file $2=expected-name -> prints hash of the kind implied by the name length
  case "${#2}" in
    64) sha256sum "$1" | cut -d' ' -f1 ;;
    40) { printf 'blob %s\0' "$(stat -c %s "$1")"; cat "$1"; } | sha1sum | cut -d' ' -f1 ;;
    *)  echo "unrecognized blob name length: $2" >&2; return 1 ;;
  esac
}

n=0; skipped=0; copied=0; bytes=0
while IFS= read -r link; do
  b="$(basename "$(readlink "${link}")")"; src="${S}/blobs/${b}"; dst="${D}/blobs/${b}"; sz="$(stat -L -c %s "${src}")"
  n=$((n+1))
  if [[ -n "${DRY_RUN:-}" && "${sz}" -gt "${DRY_MAX_BYTES}" ]]; then log "dry-run: skip large blob ${b:0:12} (${sz} B)"; continue; fi
  if [[ -f "${dst}.verified" && "$(cat "${dst}.verified")" == "${sz}" && "$(stat -c %s "${dst}" 2>/dev/null || echo -1)" == "${sz}" ]]; then
    skipped=$((skipped+1)); continue
  fi
  log "copy $(basename "${link}") ${b:0:12} ${sz} B"
  rm -f "${dst}.verified"; tmp="${dst}.tmp.$$"
  cp "${src}" "${tmp}"
  got="$(blob_hash "${tmp}" "${b}")"
  if [[ "${got}" != "${b}" ]]; then rm -f "${tmp}"; echo "HASH MISMATCH for ${link}: expected ${b} got ${got}" >&2; exit 2; fi
  mv "${tmp}" "${dst}"; echo "${sz}" > "${dst}.verified"; copied=$((copied+1)); bytes=$((bytes+sz))
done < <(find "${S}/snapshots/${REVISION}" -type l | sort)

printf %s "${REVISION}" > "${D}/refs/main"   # offline resolution of "${MODEL}" (no --revision) goes through refs/main
mkdir -p "${D}/snapshots/${REVISION}"
cp -RP --no-preserve=all "${S}/snapshots/${REVISION}/." "${D}/snapshots/${REVISION}/"   # relative symlinks into ../../blobs (no mode preservation: /freespace rejects it)
log "files=${n} copied=${copied} (${bytes} B) already_verified=${skipped} -> ${D}"
if [[ -z "${DRY_RUN:-}" ]]; then   # every link in the local snapshot must resolve
  while IFS= read -r l; do [[ -s "${l}" ]] || { echo "dangling/empty: ${l}" >&2; exit 3; }; done < <(find "${D}/snapshots/${REVISION}" -type l)
fi
echo "${DST_HF_HOME}"
