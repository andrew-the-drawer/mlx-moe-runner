#!/bin/sh
# Tags a release and updates the formula's url/sha256 for it.
# Usage: scripts/release.sh v0.1.0   (run from the repo root, after committing)
set -eu
tag="${1:?usage: release.sh vX.Y.Z}"
repo="lantrungseo/mlx-moe-runner"
git tag -a "$tag" -m "$tag"
tarball="https://github.com/$repo/archive/refs/tags/$tag.tar.gz"
tmp=$(mktemp)
git archive --format=tar.gz --prefix="mlx-moe-runner-${tag#v}/" "$tag" > "$tmp"
sha=$(shasum -a 256 "$tmp" | cut -d' ' -f1)
sed -i '' -e "s|url \".*\"|url \"$tarball\"|" -e "s|sha256 \".*\"|sha256 \"$sha\"|" Formula/mlx-moe-runner.rb
rm -f "$tmp"
echo "Formula updated for $tag (sha256 $sha). Push the tag and commit the formula:"
echo "  git push origin main $tag"
