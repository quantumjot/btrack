//! Spatial and 4D spatial-temporal hash grid indexing.

use hashbrown::HashMap;
use nalgebra::Vector3;

/// Discretized integer coordinate in 4D (x, y, z, t).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HashIndex {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub n: i32,
}

impl HashIndex {
    #[inline]
    pub fn from_coords(pos: &Vector3<f64>, t: f64, bin_size_xyz: f64, bin_size_t: f64) -> Self {
        Self {
            x: (pos[0] / bin_size_xyz).floor() as i32,
            y: (pos[1] / bin_size_xyz).floor() as i32,
            z: (pos[2] / bin_size_xyz).floor() as i32,
            n: (t / bin_size_t).floor() as i32,
        }
    }
}

/// 3D/4D spatial grid to quickly index and query observations within `max_search_radius`.
#[derive(Debug, Clone)]
pub struct ObjectBin {
    bin_size_xyz: f64,
    cube: HashMap<HashIndex, Vec<usize>>,
}

impl ObjectBin {
    pub fn new(bin_size_xyz: f64, _bin_size_t: f64) -> Self {
        Self {
            bin_size_xyz: bin_size_xyz.max(1.0),
            cube: HashMap::new(),
        }
    }

    pub fn clear(&mut self) {
        self.cube.clear();
    }

    #[inline]
    pub fn hash_index(&self, pos: &Vector3<f64>) -> HashIndex {
        HashIndex {
            x: (pos[0] / self.bin_size_xyz).floor() as i32,
            y: (pos[1] / self.bin_size_xyz).floor() as i32,
            z: (pos[2] / self.bin_size_xyz).floor() as i32,
            n: 1,
        }
    }

    pub fn add_object(&mut self, pos: &Vector3<f64>, idx: usize) {
        let h = self.hash_index(pos);
        self.cube.entry(h).or_default().push(idx);
    }

    /// Queries objects in the 27 neighboring spatial bins surrounding `pos`.
    pub fn get_near(&self, pos: &Vector3<f64>) -> Vec<usize> {
        let center = self.hash_index(pos);
        let mut results = Vec::new();

        for z in (center.z - 1)..=(center.z + 1) {
            for y in (center.y - 1)..=(center.y + 1) {
                for x in (center.x - 1)..=(center.x + 1) {
                    let key = HashIndex { x, y, z, n: 1 };
                    if let Some(indices) = self.cube.get(&key) {
                        results.extend_from_slice(indices);
                    }
                }
            }
        }
        results
    }
}

/// 4D spatial-temporal grid for candidate tracklet lookup during hypothesis generation.
#[derive(Debug, Clone)]
pub struct HypercubeBin<T: Clone> {
    bin_size_xyz: f64,
    bin_size_t: f64,
    cube: HashMap<HashIndex, Vec<T>>,
}

impl<T: Clone> HypercubeBin<T> {
    pub fn new(bin_size_xyz: f64, bin_size_t: f64) -> Self {
        Self {
            bin_size_xyz: bin_size_xyz.max(1.0),
            bin_size_t: bin_size_t.max(1.0),
            cube: HashMap::new(),
        }
    }

    pub fn clear(&mut self) {
        self.cube.clear();
    }

    #[inline]
    pub fn hash_index(&self, pos: &Vector3<f64>, t: usize) -> HashIndex {
        HashIndex::from_coords(pos, t as f64, self.bin_size_xyz, self.bin_size_t)
    }

    pub fn add(&mut self, pos: &Vector3<f64>, t: usize, item: T) {
        let idx = self.hash_index(pos, t);
        self.cube.entry(idx).or_default().push(item);
    }

    /// Finds items in spatial neighbor bins and forward in time `[n, n + 1]`.
    pub fn get_candidates(&self, pos: &Vector3<f64>, t: usize) -> Vec<T> {
        let center = self.hash_index(pos, t);
        let mut results = Vec::new();

        for n in center.n..=(center.n + 1) {
            for z in (center.z - 1)..=(center.z + 1) {
                for y in (center.y - 1)..=(center.y + 1) {
                    for x in (center.x - 1)..=(center.x + 1) {
                        let key = HashIndex { x, y, z, n };
                        if let Some(items) = self.cube.get(&key) {
                            results.extend_from_slice(items);
                        }
                    }
                }
            }
        }
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TrackObject;

    #[test]
    fn test_object_bin_query() {
        let mut bin = ObjectBin::new(10.0, 1.0);
        let obj1 = TrackObject::new(1, 5.0, 5.0, 5.0, 0);
        let obj2 = TrackObject::new(2, 50.0, 50.0, 50.0, 0);

        bin.add_object(&obj1.position(), 0);
        bin.add_object(&obj2.position(), 1);

        let query_pos = Vector3::new(6.0, 6.0, 6.0);
        let near = bin.get_near(&query_pos);
        assert_eq!(near.len(), 1);
        assert_eq!(near[0], 0);
    }
}
