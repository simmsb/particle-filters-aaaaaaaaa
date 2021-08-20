use ndarray::array;

mod filter;

fn main() {
    color_eyre::install().unwrap();

    println!("Hello, world!");

    let mut particles = filter::Particles::new(100);

    let positions = vec![array![30.0, -30.0], array![30.0, 30.0], array![-30.0, -30.0]];

    for i in 0..5 {
        particles.predict();
        // lol
        particles.update(0.4, &positions, &[42.426, 42.426, 42.426]);

        let estimated = particles.estimate(3);

        println!("{}: {:?}", i, estimated);
        // println!("{:#?}", particles);
    }
}
