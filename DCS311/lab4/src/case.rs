use nalgebra::VectorN;
use std::str::FromStr;

#[derive(Debug)]
pub struct Case {
    season: f64,
    yr: f64,
    mnth: f64,
    day: f64,
    hr: f64,
    holiday: f64,
    weekday: f64,
    workingday: f64,
    weathersit: f64,
    temp: f64,
    atemp: f64,
    hum: f64,
    windspeed: f64,
    cnt: f64,
}

impl FromStr for Case {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut iter = s.split(',');
        let _instant = iter.next().unwrap();
        let mut date_iter = iter.next().unwrap().split('/');
        let date_year = date_iter
            .next()
            .and_then(|v| f64::from_str(v).ok())
            .unwrap();
        let date_month = date_iter
            .next()
            .and_then(|v| f64::from_str(v).ok())
            .unwrap();
        let date_day = date_iter
            .next()
            .and_then(|v| f64::from_str(v).ok())
            .unwrap();
        let day = (date_year + date_month * 30.0 + date_day) / 2400.0;
        let season = iter.next().and_then(|v| f64::from_str(v).ok()).unwrap() / 4.0;
        let yr = iter.next().and_then(|v| f64::from_str(v).ok()).unwrap();
        let mnth = iter.next().and_then(|v| f64::from_str(v).ok()).unwrap() / 12.0;
        let hr = iter.next().and_then(|v| f64::from_str(v).ok()).unwrap() / 23.0;
        let holiday = iter.next().and_then(|v| f64::from_str(v).ok()).unwrap();
        let weekday = iter.next().and_then(|v| f64::from_str(v).ok()).unwrap();
        let workingday = iter.next().and_then(|v| f64::from_str(v).ok()).unwrap();
        let weathersit = iter.next().and_then(|v| f64::from_str(v).ok()).unwrap() / 4.0;
        let temp = iter.next().and_then(|v| f64::from_str(v).ok()).unwrap();
        let atemp = iter.next().and_then(|v| f64::from_str(v).ok()).unwrap();
        let hum = iter.next().and_then(|v| f64::from_str(v).ok()).unwrap();
        let windspeed = iter.next().and_then(|v| f64::from_str(v).ok()).unwrap();
        let cnt = iter.next().and_then(|v| f64::from_str(v).ok()).unwrap();

        Ok(Case {
            season,
            yr,
            mnth,
            day,
            hr,
            holiday,
            weekday,
            workingday,
            weathersit,
            temp,
            atemp,
            hum,
            windspeed,
            cnt,
        })
    }
}

impl Case {
    pub fn into_io(self) -> (VectorN<f64, nalgebra::U13>, VectorN<f64, nalgebra::U1>) {
        (
            VectorN::<_, nalgebra::U13>::from_row_slice(&[
                self.season,
                self.yr,
                self.mnth,
                self.day,
                self.hr,
                self.holiday,
                self.weekday,
                self.workingday,
                self.weathersit,
                self.temp,
                self.atemp,
                self.hum,
                self.windspeed,
            ]),
            VectorN::<_, nalgebra::U1>::new(self.cnt),
        )
    }
}
