export interface LayerHeader {
  rows: number;
  cols: number;
  bias_size: number;
}

export interface LayerSnapshot {
  header: LayerHeader;
  weights: number[];
  biases: number[];
}

export interface Snapshot {
  version: number;
  epoch: number;
  step: number;
  loss: number;
  layers: LayerSnapshot[];
}
