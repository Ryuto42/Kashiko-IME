#!/bin/bash
# Build the release assets for the public distribution repository:
#   dist/Kashiko-<version>-macos-arm64.zip and dist/install.sh for <tag>, and
#   dist/dict.tgz for the dictionary release named in the Makefile.
# Usage: scripts/package-release.sh <release-repo> <tag>
set -euo pipefail
cd "$(dirname "$0")/.."

RELEASE_REPO="${1:?release repository, e.g. owner/name}"
TAG="${2:?release tag, e.g. v0.1.0}"
VERSION="$(/usr/libexec/PlistBuddy -c 'Print CFBundleShortVersionString' Info.plist)"
APP_ZIP="Kashiko-$VERSION-macos-arm64.zip"
DIST=dist

make bundle
rm -rf "$DIST" && mkdir -p "$DIST/pkg"
cp -a out/Kashiko.app "$DIST/pkg/"
cp ../LICENSE-MIT ../LICENSE-APACHE ../THIRD_PARTY_LICENSES "$DIST/pkg/"
(cd "$DIST/pkg" && ditto -c -k --keepParent Kashiko.app "../$APP_ZIP" \
  && zip -q "../$APP_ZIP" LICENSE-MIT LICENSE-APACHE THIRD_PARTY_LICENSES)
rm -rf "$DIST/pkg"
cp dict.tgz "$DIST/"

sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
DICT_BIN_SHA256="$(sed -n 's/^DICT_BIN_SHA256 = //p' Makefile)"
DICT_TGZ_SHA256="$(sed -n 's/^DICT_TGZ_SHA256 = //p' Makefile)"
[ "$(sha "$DIST/dict.tgz")" = "$DICT_TGZ_SHA256" ] || { echo "dict.tgz does not match the pinned digest" >&2; exit 1; }

DICT_TAG="$(sed -n 's/^DICT_RELEASE ?= //p' Makefile)"
sed -e "s|@RELEASE_REPO@|$RELEASE_REPO|" -e "s|@TAG@|$TAG|" -e "s|@DICT_TAG@|$DICT_TAG|" \
    -e "s|@APP_ZIP@|$APP_ZIP|" -e "s|@APP_ZIP_SHA256@|$(sha "$DIST/$APP_ZIP")|" \
    -e "s|@DICT_TGZ_SHA256@|$DICT_TGZ_SHA256|" -e "s|@DICT_BIN_SHA256@|$DICT_BIN_SHA256|" \
    scripts/install.sh.in > "$DIST/install.sh"
chmod +x "$DIST/install.sh"
(cd "$DIST" && shasum -a 256 "$APP_ZIP" dict.tgz install.sh > SHA256SUMS)
ls -la "$DIST"
