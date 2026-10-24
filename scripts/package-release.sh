#!/bin/bash
# Build the release assets: dist/Kashiko-<version>-macos-arm64.zip,
# dist/install.sh and dist/SHA256SUMS for <tag>. The system dictionary is a
# release of its own (DICT_RELEASE in the Makefile), pinned in install.sh.
# Usage: scripts/package-release.sh <owner/repo> <tag>
set -euo pipefail
cd "$(dirname "$0")/.."

RELEASE_REPO="${1:?release repository, e.g. owner/name}"
TAG="${2:?release tag, e.g. v0.1.0}"
VERSION="$(/usr/libexec/PlistBuddy -c 'Print CFBundleShortVersionString' macos/Info.plist)"
[ "$TAG" = "v$VERSION" ] || { echo "tag $TAG does not match Info.plist version $VERSION" >&2; exit 1; }
APP_ZIP="Kashiko-$VERSION-macos-arm64.zip"
DIST=dist

make bundle
# The app must run on any Apple Silicon Mac: only system libraries may be
# linked (a Homebrew libomp, say, would not exist on the user's machine).
for bin in build/Kashiko.app/Contents/MacOS/*; do
  if otool -L "$bin" | tail -n +2 | awk '{print $1}' | grep -vE '^(/usr/lib/|/System/Library/)'; then
    echo "$bin links a non-system library" >&2
    exit 1
  fi
done
rm -rf "$DIST" && mkdir -p "$DIST/pkg"
cp -a build/Kashiko.app "$DIST/pkg/"
cp LICENSE-MIT LICENSE-APACHE THIRD_PARTY_LICENSES "$DIST/pkg/"
python3 scripts/third-party-rust.py > "$DIST/pkg/THIRD_PARTY_RUST_LICENSES"
(cd "$DIST/pkg" && ditto -c -k --keepParent Kashiko.app "../$APP_ZIP" \
  && zip -q "../$APP_ZIP" LICENSE-MIT LICENSE-APACHE THIRD_PARTY_LICENSES THIRD_PARTY_RUST_LICENSES)
rm -rf "$DIST/pkg"

sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
DICT_BIN_SHA256="$(sed -n 's/^DICT_BIN_SHA256 = //p' Makefile)"
DICT_TGZ_SHA256="$(sed -n 's/^DICT_TGZ_SHA256 = //p' Makefile)"

DICT_TAG="$(sed -n 's/^DICT_RELEASE ?= //p' Makefile)"
sed -e "s|@RELEASE_REPO@|$RELEASE_REPO|" -e "s|@TAG@|$TAG|" -e "s|@DICT_TAG@|$DICT_TAG|" \
    -e "s|@APP_ZIP@|$APP_ZIP|" -e "s|@APP_ZIP_SHA256@|$(sha "$DIST/$APP_ZIP")|" \
    -e "s|@DICT_TGZ_SHA256@|$DICT_TGZ_SHA256|" -e "s|@DICT_BIN_SHA256@|$DICT_BIN_SHA256|" \
    scripts/install.sh.in > "$DIST/install.sh"
chmod +x "$DIST/install.sh"
(cd "$DIST" && shasum -a 256 "$APP_ZIP" install.sh > SHA256SUMS)
ls -la "$DIST"
