OUTDIR = build
APP = $(OUTDIR)/Kashiko.app
SWIFT_PACKAGE = macos
SWIFT_BUILD = $(SWIFT_PACKAGE)/.build
# Names inside the bundle, which are what Activity Monitor shows: the input
# method itself, and the engine process that runs the model.
APP_EXECUTABLE = Kashiko IME
ENGINE_EXECUTABLE = Kashiko IME Engine
ENGINE_BIN = kashiko-imserver
INSTALL_DIR = $(HOME)/Library/Input Methods
DATA_DIR = $(HOME)/Library/Application Support/Kashiko
LEGACY_DATA_DIR = $(HOME)/Library/Application Support/com.kashiko.kashiko-im
# Pinned release asset (never `latest`), fetched over HTTPS; a local DICT_TGZ
# is used instead when present.
RELEASE_REPO ?= Ryuto42/Kashiko-IME
DICT_RELEASE ?= dict-20260226
DICT_URL ?= https://github.com/$(RELEASE_REPO)/releases/download/$(DICT_RELEASE)/dict.tgz
DICT_TGZ ?= dict.tgz
DICT_TGZ_SHA256 = f194d2526bf826622bc5baf7c07e7525ad9bde2dc63896cd929045283a4aeccd
DICT_BIN_SHA256 = d85fed3c7e408e67f4a5dbe2314338366ed6183e0498d2726bc5a4dcaf5bb83b

.PHONY: all build build-universal bundle install install-dict install-models test fmt lint clean

all: build

build:
	swift build -c release --package-path $(SWIFT_PACKAGE)
	cargo build --release -p kashiko-im --bin $(ENGINE_BIN)

# Intel + Apple Silicon. Prebuilt releases are Apple Silicon only.
build-universal:
	swift build -c release --package-path $(SWIFT_PACKAGE) --arch arm64
	swift build -c release --package-path $(SWIFT_PACKAGE) --arch x86_64
	mkdir -p $(SWIFT_BUILD)/release
	lipo -create \
		$(SWIFT_BUILD)/arm64-apple-macosx/release/KashikoIME \
		$(SWIFT_BUILD)/x86_64-apple-macosx/release/KashikoIME \
		-output $(SWIFT_BUILD)/release/KashikoIME
	rustup target add aarch64-apple-darwin x86_64-apple-darwin
	cargo build --release -p kashiko-im --bin $(ENGINE_BIN) --target aarch64-apple-darwin
	cargo build --release -p kashiko-im --bin $(ENGINE_BIN) --target x86_64-apple-darwin
	lipo -create \
		target/aarch64-apple-darwin/release/$(ENGINE_BIN) \
		target/x86_64-apple-darwin/release/$(ENGINE_BIN) \
		-output target/release/$(ENGINE_BIN)

test:
	swift test --package-path $(SWIFT_PACKAGE)
	cargo test --workspace

# Assemble the .app bundle in build/ from the built binaries
bundle: build
	rm -rf $(APP)
	mkdir -p $(APP)/Contents/MacOS
	mkdir -p $(APP)/Contents/Resources
	cp $(SWIFT_PACKAGE)/Info.plist $(APP)/Contents/
	cp $(SWIFT_BUILD)/release/KashikoIME "$(APP)/Contents/MacOS/$(APP_EXECUTABLE)"
	cp target/release/$(ENGINE_BIN) "$(APP)/Contents/MacOS/$(ENGINE_EXECUTABLE)"
	cp -r $(SWIFT_PACKAGE)/Resources/* $(APP)/Contents/Resources/
	xattr -cr $(APP)
	codesign --force --options runtime --sign - "$(APP)/Contents/MacOS/$(ENGINE_EXECUTABLE)"
	codesign --force --options runtime --sign - $(APP)

# Install the prebuilt system dictionary unless one is already installed.
# Both the archive and the extracted dict.bin are verified against pinned
# SHA-256 digests. Non-fatal on failure — the IME works model-only without it.
install-dict:
	@if [ ! -e "$(DATA_DIR)" ] && [ -d "$(LEGACY_DATA_DIR)" ]; then \
		mv "$(LEGACY_DATA_DIR)" "$(DATA_DIR)" && echo "Moved $(LEGACY_DATA_DIR) to $(DATA_DIR)"; \
	fi
	@if [ -f "$(DATA_DIR)/dict.bin" ]; then \
		echo "System dictionary already installed: $(DATA_DIR)/dict.bin (skipping download)"; \
	else \
		tmp=$$(mktemp -d) && \
		( if [ -f "$(DICT_TGZ)" ]; then \
		    cp "$(DICT_TGZ)" "$$tmp/dict.tgz"; \
		  else \
		    echo "Downloading system dictionary from $(DICT_URL) ..." && \
		    curl -fL --proto '=https' --progress-bar -o "$$tmp/dict.tgz" "$(DICT_URL)"; \
		  fi && \
		  echo "$(DICT_TGZ_SHA256)  $$tmp/dict.tgz" | shasum -a 256 -c - && \
		  tar xzf "$$tmp/dict.tgz" -C "$$tmp" dict.bin && \
		  echo "$(DICT_BIN_SHA256)  $$tmp/dict.bin" | shasum -a 256 -c - && \
		  mkdir -p "$(DATA_DIR)" && \
		  cp "$$tmp/dict.bin" "$(DATA_DIR)/" && \
		  echo "Installed system dictionary to $(DATA_DIR)/dict.bin" \
		) || echo "WARNING: system dictionary install failed; install manually (see README)"; \
		rm -rf "$$tmp"; \
	fi

# Prefetch the conversion models into the HuggingFace cache and build the
# mapped dictionary cache, so the first launch doesn't block on either.
# Non-fatal on failure (e.g. offline) — the engine retries on first launch.
install-models: build
	@echo "Prefetching conversion models (HuggingFace cache) ..."
	@target/release/$(ENGINE_BIN) --prefetch-models || \
		echo "WARNING: model prefetch failed; models will be downloaded on first launch"

install: bundle install-dict install-models
	rm -rf "$(INSTALL_DIR)/Kashiko.app"
	cp -a $(APP) "$(INSTALL_DIR)/"
	-killall "$(APP_EXECUTABLE)" KashikoIME 2>/dev/null || true
	@echo "Installed to $(INSTALL_DIR)/Kashiko.app"
	@echo "First install: log out and back in, then add Kashiko in System Settings > Keyboard > Input Sources."

fmt:
	swift-format format -ri $(SWIFT_PACKAGE)/Sources/ $(SWIFT_PACKAGE)/Tests/
	cargo fmt --all

lint:
	swift-format lint -r $(SWIFT_PACKAGE)/Sources/ $(SWIFT_PACKAGE)/Tests/

clean:
	rm -rf $(SWIFT_BUILD) $(OUTDIR)
