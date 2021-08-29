use bevy::{DefaultPlugins, prelude::App, wgpu::{WgpuFeature, WgpuFeatures, WgpuOptions}, diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin}};

#[cfg(debug_assertions)]
#[allow(unused_imports)] use bevy_dylib;
// use ndarray::array;

use crate::render::RenderPlugin;

mod filter;
mod render;

fn main() {
    color_eyre::install().unwrap();
    // simple_logger::SimpleLogger::new().init().unwrap();

    println!("Hello, world!");

    let particles = filter::Particles::new(10000, 100.0);

    App::new()
        .insert_resource(particles)
        .insert_resource(WgpuOptions {
            features: WgpuFeatures {
                features: vec![WgpuFeature::NonFillPolygonMode],
            },
            ..Default::default()
        })
        .add_plugins(DefaultPlugins)
        .add_plugin(RenderPlugin)
        // .add_plugin(FrameTimeDiagnosticsPlugin::default())
        // .add_plugin(LogDiagnosticsPlugin::default())
        // .add_plugin(bevy::wgpu::diagnostic::WgpuResourceDiagnosticsPlugin::default())
        .run();

    // let positions = vec![array![30.0, -30.0], array![30.0, 30.0], array![-30.0, -30.0]];

    // for i in 0..50 {
    //     particles.predict(0.1, 1.0);
    //     // lol
    //     particles.update(1.0, &positions);

    //     if particles.neff() < (particles.n as f64 / 2.0) {
    //         particles.resample();
    //     }

    //     let estimated = particles.estimate(3);

    //     println!("{}: {:?}", i, estimated);
    //     // println!("{:#?}", particles);
    // }
}
