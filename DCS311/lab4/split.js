const { readFileSync, writeFileSync } = require("fs");

const shuffle = (a) => {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
};

const file = readFileSync("dataset/train.csv").toString("utf-8");

const arr = shuffle(file.split("\n"));

writeFileSync(
  "dataset/train_tr.csv",
  arr.slice(0, Math.round(arr.length * 0.75)).join("\n"),
);

writeFileSync(
  "dataset/train_va.csv",
  arr.slice(Math.round(arr.length * 0.75)).join("\n"),
);
