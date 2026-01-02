# crimson-mute

Use AMD's noise cancellation on Linux (and other platforms).

AMD's Adrenalin drivers include a neural network-based noise suppression feature for microphones. This plugin extracts those models and runs them as a standard audio plugin (VST3/CLAP) or standalone application.

**This project is not affiliated with, endorsed by, or supported by AMD. Use at your own risk.**

Currently only CPU inference is supported. The official Windows driver also supports GPU offloading.

## Requirements

- **Linux:** Works out of the box
- **Models:** ONNX models extracted from AMD drivers (see below)

## Getting the Models

Run the fetch script to download and extract the models from AMD's drivers:

```bash
./fetch-models.sh
```

This will:
1. Download the official AMD Adrenalin driver package (~1GB)
2. Extract the ONNX models
3. Install them to `~/.local/share/anr-plugin/`

**Note:** You must accept AMD's EULA to use the models.

## Building

```bash
cargo xtask bundle crimson-mute --release
```

This builds the VST3, CLAP, and standalone executable.

## Installation

Copy the built plugin to your plugin directory:

```bash
# VST3
cp -r target/bundled/crimson-mute.vst3 ~/.vst3/

# CLAP
cp -r target/bundled/crimson-mute.clap ~/.clap/
```

## Usage

### As a Plugin

Load `crimson-mute` in your DAW or audio host on a microphone track.

### Standalone

```bash
./target/bundled/crimson-mute --help
```

### Parameters

- **Mode:** Processing mode
  - `PMLN Only` - Primary model only
  - `STLN Only` - Secondary model only
  - `STLN + PMLN` - Both models cascaded (strongest noise reduction)
- **STLN Strength:** Adjust suppression intensity (0.5 - 4.0)

## Model Locations

The plugin searches for models in this order:

1. `ANR_MODEL_DIR` environment variable
2. Next to the plugin binary
3. `~/.local/share/anr-plugin/`
4. `/usr/share/anr-plugin/`
5. `/usr/local/share/anr-plugin/`

## License

MIT. The AMD ONNX models are proprietary and subject to AMD's EULA.
