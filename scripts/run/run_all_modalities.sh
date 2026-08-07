#!/bin/bash
set -e

# Run song-stratified cross-validation for all 4 modality configurations sequentially.
#
# Usage:
#   ./run_all_modalities.sh                        # all 4 modalities, all folds
#   ./run_all_modalities.sh --folds ch_1v2t,sg_1v2t  # specific folds only
#   ./run_all_modalities.sh --sleep 10             # longer GPU cooldown between folds
#
# Modalities:
#   1. chroma      — source-separated audio (Audio/)
#   2. cqt         — source-separated audio (Audio/)
#   3. mel_ss      — mel on source-separated audio (mel_base config + Audio/ override)
#   4. mel_orig    — mel on original audio (mel_base config + Audio_Original/)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PANSORI_DATA_ROOT="${PANSORI_DATA_ROOT:-$REPO_ROOT/../Pansori_Data}"
cd "$REPO_ROOT"


SS_AUDIO="$PANSORI_DATA_ROOT/Audio"
OG_AUDIO="$PANSORI_DATA_ROOT/Audio_Original"

# Forward any flags (--folds, --sleep, etc.) to each run_song_stratified.sh call
PASS_ARGS=("$@")

run_modality() {
    local label="$1"; shift
    echo ""
    echo "########################################"
    echo "  $label"
    echo "########################################"
    bash "$REPO_ROOT/run_song_stratified.sh" "$@" "${PASS_ARGS[@]}"
    echo "  $label complete."
}

run_modality "1/4  chroma  (source-separated)" \
    --config_path frame --config_name chroma_base

run_modality "2/4  cqt     (source-separated)" \
    --config_path frame --config_name cqt_base

#run_modality "3/4  mel     (source-separated)" \
#    --config_path frame --config_name mel_base \
#    --extra "data.data_dir=$SS_AUDIO"

run_modality "4/4  mel     (original)" \
    --config_path frame --config_name mel_base \
    --extra "data.data_dir=$OG_AUDIO"

echo ""
echo "All modalities complete."
