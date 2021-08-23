use itertools::Itertools;
use linfa::traits::{Fit, Predict};
use linfa_clustering::KMeans;
use ndarray as nd;
use ndarray_linalg::Norm;
use ndarray_rand::{rand::Rng, RandomExt};
use ndarray_stats::{QuantileExt, SummaryStatisticsExt};

#[derive(Debug)]
pub struct Particles {
    positions: nd::Array2<f64>,
    velocities: nd::Array2<f64>,
    weights: nd::Array1<f64>,
    latest_groups: nd::Array1<usize>,
    pub n: usize,
    search_space: f64,
}

fn norm_rand_array2(mean: f64, std_dev: f64, n: usize) -> nd::Array2<f64> {
    let distr = ndarray_rand::rand_distr::Normal::new(mean, std_dev).unwrap();

    nd::Array::random((n, 2), distr)
}

fn uniform_rand_array2(min: f64, max: f64, n: usize) -> nd::Array2<f64> {
    let distr = ndarray_rand::rand_distr::Uniform::new(min, max);

    nd::Array::random((n, 2), distr)
}

impl Particles {
    pub fn new(n: usize, search_space: f64) -> Self {
        let positions = uniform_rand_array2(-search_space, search_space, n);
        let velocities = norm_rand_array2(0.0, 1.0, n);

        let weights = nd::Array::from_elem(n, 1.0 / n as f64);
        let latest_groups = nd::Array::zeros(n);

        Self {
            positions,
            velocities,
            weights,
            latest_groups,
            n,
            search_space,
        }
    }

    pub fn positions(&self) -> nd::ArrayView2<f64> {
        self.positions.view()
    }

    pub fn latest_groups(&self) -> nd::ArrayView1<usize> {
        self.latest_groups.view()
    }

    pub fn weights(&self) -> nd::ArrayView1<f64> {
        self.weights.view()
    }

    pub fn upper_bounds(&self) -> (f64, f64) {
        // minmax pls?
        let max_x = *self.positions.column(0).max().unwrap();
        let min_x = *self.positions.column(0).min().unwrap();

        // minmax pls?
        let max_y = *self.positions.column(1).max().unwrap();
        let min_y = *self.positions.column(1).min().unwrap();

        let x = max_x.abs().max(min_x.abs());
        let y = max_y.abs().max(min_y.abs());

        (x, y)
    }

    pub fn predict(&mut self, std_dev: f64, dt: f64) {
        let offsets = norm_rand_array2(0.0, std_dev, self.n);
        self.positions += &((&self.velocities + offsets) * dt);
    }

    pub fn update(&mut self, std_dev: f64, positions: &[nd::Array1<f64>]) {
        for pos in positions {
            let dist = &self.positions - pos;

            let dist_norm = dist
                .axis_iter(nd::Axis(0))
                .map(|x| x.norm())
                .collect::<Vec<_>>();
            self.weights *= &gen_weight_updates(&dist_norm, std_dev);
        }

        self.weights += 1e-300;
        self.weights /= self.weights.sum();
    }

    pub fn estimate(&mut self, n: usize) -> Vec<((f64, f64), (f64, f64))> {
        // TODO: don't do this clone
        let observations = linfa::DatasetBase::from(self.positions.clone());

        let model = KMeans::params(n).fit(&observations).unwrap();

        // dbg!(model.centroids());

        let dataset = model.predict(observations);

        self.latest_groups = dataset.targets().clone();

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

                match (mean.as_slice().unwrap(), var.as_slice().unwrap()) {
                    (&[x, y], &[xv, yv]) => ((x, y), (xv, yv)),
                    _ => panic!("fuck"),
                }
            })
            .collect_vec()
    }

    pub fn resample(&mut self) {
        let distr = ndarray_rand::rand_distr::Uniform::new(0.0, 1.0);

        let weight_positions = (nd::Array::random(self.n, distr)
            + nd::Array::range(0.0, self.n as f64, 1.0))
            / self.n as f64;

        let mut indexes = nd::Array1::<usize>::zeros(self.n);

        let mut cumsum = self.weights.clone();
        cumsum.accumulate_axis_inplace(nd::Axis(0), |&prev, curr| *curr += prev);
        cumsum[self.n - 1] = 1.0;

        let (mut i, mut j) = (0, 0);

        while i < self.n {
            debug_assert!(i < weight_positions.len());
            debug_assert!(j < cumsum.len());
            if weight_positions[i] < cumsum[j] {
                indexes[i] = j;
                i += 1;
            } else {
                j += 1;
            }
        }

        self.positions =
            nd::Array::from_shape_fn((self.n, 2), |(i, j)| self.positions[(indexes[i], j)]);
        self.velocities =
            nd::Array::from_shape_fn((self.n, 2), |(i, j)| self.velocities[(indexes[i], j)]);

        // introduce a bit of randomness
        self.positions += &norm_rand_array2(0.0, 1.0, self.n);
        self.velocities += &norm_rand_array2(0.0, 0.1, self.n);

        let mut rng = ndarray_rand::rand::thread_rng();

        for _ in 0..rng.gen_range(0..(self.n / 4)) {
            self.positions[(rng.gen_range(0..self.n), 0)] =
                rng.gen_range(-self.search_space..self.search_space);
            self.positions[(rng.gen_range(0..self.n), 1)] =
                rng.gen_range(-self.search_space..self.search_space);
        }

        self.weights.fill(1.0 / self.n as f64);
    }

    pub fn neff(&self) -> f64 {
        1.0 / self.weights.fold(0.0, |acc, val| acc + val * val)
    }
}

fn gen_weight_updates(distances: &[f64], std_dev: f64) -> nd::Array1<f64> {
    fn pdf(x: f64, mean: f64, std_dev: f64) -> f64 {
        let d = (x - mean) / std_dev;
        (-0.5 * d * d).exp() / (2.5066282746310002 * std_dev)
    }

    nd::Array1::from_iter(distances.iter().map(|d| pdf(*d, 0.0, std_dev)))
}
