use nih_plug::prelude::*;

use anr_plugin::AnrPlugin;

fn main() {
    nih_export_standalone::<AnrPlugin>();
}
