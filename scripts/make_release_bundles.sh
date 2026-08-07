#!/bin/bash
# Build one GitHub Release asset per representation, plus checksums.
#
#   bash scripts/make_release_bundles.sh [OUT_DIR]
#
# Each bundle holds only what inference needs: fold*_best_model.pt plus the
# config.yaml that was saved next to it. Posteriorgram PNGs, periodic snapshots
# and W&B caches are deliberately excluded — they are regenerable and would
# multiply the download size for no benefit.
#
# GitHub caps a single Release asset at 2 GiB. Every bundle below is well under
# that; the script prints sizes so a future run that crosses the line is visible
# rather than failing at upload time.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT="${1:-release}"
mkdir -p "$OUT"

# bundle name : source directories
bundle() {
    local name="$1"; shift
    local dirs=()
    for d in "$@"; do [ -d "$d" ] && dirs+=("$d"); done
    if [ ${#dirs[@]} -eq 0 ]; then
        echo "  skip  $name (no source directory present)"
        return
    fi
    tar czf "$OUT/$name.tar.gz" \
        --exclude='*_iter.pt' --exclude='step*.pt' \
        --exclude='*.png' --exclude='wandb' --exclude='*.log' \
        "${dirs[@]}"
    printf "  built %-26s %8s   from: %s\n" \
        "$name.tar.gz" "$(du -h "$OUT/$name.tar.gz" | cut -f1)" "${dirs[*]}"
}

echo "Building release bundles into $OUT/"

bundle mel_weights      weights/frame/Mel_Original_Version \
                        weights/frame/Mel_Original_Song_Stratified
bundle mel_sep_weights  weights/frame/Mel_Sep_Version \
                        weights/frame/Mel_Sep_Stratified
bundle pesto_weights    weights/frame/Pesto_Version \
                        weights/frame/Pesto_Song_Stratified
bundle midi_weights     weights/midi/MIDI_Version \
                        weights/midi/MIDI_Song_Stratified
# CultureMERT checkpoints are ~253 MB each (the backbone dominates), so all ten
# song-stratified folds together exceed GitHub's 2 GiB per-asset cap. Ship the
# version-split model on its own — that is the one most people want — and split
# the 10-fold set into two halves.
bundle cmert_version_weights weights/cmert/layer08_10k_version

CMERT_SS=weights/cmert/layer08_song_stratified
if [ -d "$CMERT_SS" ]; then
    mapfile -t CMERT_RUNS < <(find "$CMERT_SS" -mindepth 1 -maxdepth 1 -type d -not -name test_metrics | sort)
    half=$(( (${#CMERT_RUNS[@]} + 1) / 2 ))
    bundle cmert_song_stratified_weights_part1 "${CMERT_RUNS[@]:0:$half}" "$CMERT_SS/test_metrics"
    bundle cmert_song_stratified_weights_part2 "${CMERT_RUNS[@]:$half}"
fi
bundle cqt_weights      weights/frame/CQT_Version weights/frame/CQT_Song_Stratified
bundle chroma_weights   weights/frame/Chroma_Version weights/frame/Chroma_Song_Stratified
bundle crepe_weights    weights/frame/Crepe_Version weights/frame/Crepe_Song_Stratified

# annotations are small and useful on their own
tar czf "$OUT/labels.tar.gz" data/Label
printf "  built %-26s %8s   from: data/Label\n" "labels.tar.gz" "$(du -h "$OUT/labels.tar.gz" | cut -f1)"

( cd "$OUT" && sha256sum ./*.tar.gz > SHA256SUMS.txt )

echo
echo "Bundles in $OUT/:"
du -h "$OUT"/*.tar.gz | sort -h | sed 's/^/  /'
echo "  total: $(du -sh "$OUT" | cut -f1)"
echo
over=$(find "$OUT" -name '*.tar.gz' -size +2G | wc -l)
if [ "$over" -gt 0 ]; then
    echo "WARNING: $over bundle(s) exceed GitHub's 2 GiB per-asset limit:"
    find "$OUT" -name '*.tar.gz' -size +2G -exec du -h {} \;
else
    echo "All bundles are under GitHub's 2 GiB per-asset limit."
fi
echo
echo "Upload with:"
echo "  gh release create v1.0 $OUT/*.tar.gz $OUT/SHA256SUMS.txt \\"
echo "     --title 'Pretrained weights' --notes 'Per-representation checkpoints.'"
