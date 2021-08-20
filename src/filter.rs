use itertools::Itertools;
use linfa::traits::{Fit, Predict};
use linfa_clustering::KMeans;
use ndarray as nd;
use ndarray_linalg::Norm;
use ndarray_rand::RandomExt;
use ndarray_stats::SummaryStatisticsExt;

#[derive(Debug)]
pub struct Particles {
    positions: nd::Array2<f64>,
    velocities: nd::Array2<f64>,
    weights: nd::Array1<f64>,
    n: usize,
}

impl Particles {
    pub fn new(n: usize) -> Self {
        let position_distr = ndarray_rand::rand_distr::Normal::new(0.0, 50.0).unwrap();
        let positions = nd::Array::random((n, 2), position_distr);

        let velocity_distr = ndarray_rand::rand_distr::Normal::new(0.0, 4.0).unwrap();
        let velocities = nd::Array::random((n, 2), velocity_distr);

        let weights = nd::Array::from_elem(n, 1.0 / n as f64);

        Self {
            positions,
            velocities,
            weights,
            n,
        }
    }

    pub fn predict(&mut self) {
        // TODO: add randomness here
        self.positions += &self.velocities;
    }

    pub fn update(&mut self, std_dev: f64, positions: &[nd::Array1<f64>], measured: &[f64]) {
        for (pos, measured) in itertools::izip!(positions, measured) {
            let dist = &self.positions - pos;

            let dist_norm = dist
                .axis_iter(nd::Axis(0))
                .map(|x| x.norm())
                .collect::<Vec<_>>();
            self.weights *= &gen_weight_updates(&dist_norm, std_dev, *measured);
        }

        self.weights += 1e-300;
        self.weights /= self.weights.sum();
    }

    pub fn estimate(&self, n: usize) -> Vec<((f64, f64), (f64, f64))> {
        // TODO: don't do this clone
        let observations = linfa::DatasetBase::from(self.positions.clone());

        let model = KMeans::params(n).fit(&observations).unwrap();

        dbg!(model.centroids());

        let dataset = model.predict(observations);

        //.filter_map(|(g, p, w)| g.map(|g| (g, (p, w))))
        let groups = itertools::izip!(dataset.targets(), self.positions.rows(), &self.weights)
            .map(|(g, p, w)| (g, (p, w)))
            .into_grouping_map()
            .fold(
                (nd::Array2::default((0, 2)), nd::Array1::default(0)),
                |(mut pa, mut wa), _, (p, w)| {
                    pa.push_row(p).unwrap();
                    wa.append(nd::Axis(0), nd::ArrayView1::from_shape(1, &[*w]).unwrap())
                        .unwrap();
                    (pa, wa)
                },
            );

        groups
            .values()
            .map(|(positions, weights)| {
                let mean = positions.weighted_mean_axis(nd::Axis(0), weights).unwrap();
                let var = positions
                    .weighted_var_axis(nd::Axis(0), weights, 0.0)
                    .unwrap();

                dbg!(&mean);
                dbg!(&var);

                match (mean.as_slice().unwrap(), var.as_slice().unwrap()) {
                    (&[x, y], &[xv, yv]) => ((x, y), (xv, yv)),
                    _ => panic!("fuck"),
                }
            })
            .collect_vec()
    }
}

fn gen_weight_updates(distances: &[f64], std_dev: f64, measured: f64) -> nd::Array1<f64> {
    fn pdf(x: f64, mean: f64, std_dev: f64) -> f64 {
        let d = (x - mean) / std_dev;
        (-0.5 * d * d).exp() / (2.5066282746310002 * std_dev)
    }

    nd::Array1::from_iter(distances.iter().map(|d| pdf(measured, *d, std_dev)))
}
