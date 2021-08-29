use ndarray::array;

mod filter;

fn main() {
    color_eyre::install().unwrap();

    println!("Hello, world!");

    let mut particles = filter::Particles::new(5000, 100.0);
    let positions = vec![
        array![30.0, -30.0],
        array![30.0, 30.0],
        array![-30.0, -30.0],
    ];

    loop {
        particles.predict(1.0, 1.0 / 10.0);

        // lol
        particles.update(20.0, &positions);

        if particles.neff() < (particles.n as f32 / 6.0) {
            println!("resampling");
            dbg!(particles.neff());
            particles.resample();
        }

        // let estimated = particles.estimate(3);
    }
}
