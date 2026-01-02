use nih_plug::prelude::*;

use crimson_mute::AnrPlugin;

fn main() {
    nih_export_standalone::<AnrPlugin>();
}
