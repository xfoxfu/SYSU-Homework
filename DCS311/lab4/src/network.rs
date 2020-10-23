use crate::{Case, Context, Layer};
use nalgebra::VectorN;

pub struct Network {
    cases: Vec<(VectorN<f64, nalgebra::U13>, VectorN<f64, nalgebra::U1>)>,
    pub layer1: Layer<nalgebra::U13, nalgebra::U10>,
    pub layer2: Layer<nalgebra::U10, nalgebra::U10>,
    pub layer3: Layer<nalgebra::U10, nalgebra::U1>,
    ctx: Context,
}

impl Network {
    pub fn new(eta: f64, cases: impl Iterator<Item = Case>) -> Self {
        Self {
            cases: cases.map(Case::into_io).collect(),
            layer1: Layer::new(),
            layer2: Layer::new(),
            layer3: Layer::new(),
            ctx: Context::new(eta),
        }
    }

    pub fn learn(&mut self) {
        // 获得输入
        let a0s: Vec<_> = self.cases.iter().map(|(a0, _)| a0).collect();
        // 正向传播获得 $\matv{a}$
        let a1s: Vec<_> = a0s
            .iter()
            .map(|a0| self.layer1.forward_pass_activate(a0))
            .collect();
        let a2s: Vec<_> = a1s
            .iter()
            .map(|a1| self.layer2.forward_pass_activate(a1))
            .collect();
        let z3s: Vec<_> = a2s.iter().map(|a2| self.layer3.forward_pass(a2)).collect();
        let a3s = &z3s;

        // 获得最后一层 $\delta$
        let g3s: Vec<_> = a3s
            .iter()
            .zip(self.cases.iter())
            .map(|(a3, (_, a3h))| a3 - a3h)
            .collect();
        // 特殊地计算最后一层的 $\Delta\matv{W}, \Delta\matv{b}$
        let dw3 = g3s
            .iter()
            .zip(a2s.iter())
            .map(|(g3, a2)| g3 * a2.transpose())
            .sum();
        let db3 = g3s.iter().sum();
        // 第 2 层反向传播
        let (g2s, dw2, db2) =
            self.layer2
                .back_propogation(a1s.iter(), a2s.iter(), g3s.iter(), &self.layer3);
        // 第 1 层反向传播
        let (_, dw1, db1) =
            self.layer1
                .back_propogation(a0s.iter().copied(), a1s.iter(), g2s.iter(), &self.layer2);

        // 应用 $\Delta\matv{W}, \Delta\matv{b}$
        self.layer3
            .apply_dw_db(&dw3, &db3, self.ctx.eta, self.cases.len());
        self.layer2
            .apply_dw_db(&dw2, &db2, self.ctx.eta, self.cases.len());
        self.layer1
            .apply_dw_db(&dw1, &db1, self.ctx.eta, self.cases.len());
    }

    pub fn guess(&self, a0: &VectorN<f64, nalgebra::U13>) -> f64 {
        let a1 = self.layer1.forward_pass_activate(a0);
        let a2 = self.layer2.forward_pass_activate(&a1);
        // 最后一层不需要激活
        let z3 = self.layer3.forward_pass(&a2);

        *z3.iter().next().unwrap()
    }
}
