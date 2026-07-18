# DNN 3D Visualizer

A real-time 3D visualization of a Deep Neural Network training process using Three.js and React.

## Features

- **Real-time WebSocket Updates**: Connects to the Go WebSocket server to receive live training data
- **3D Network Visualization**: 
  - Green nodes represent input layer neurons
  - Blue nodes represent output layer neurons
  - Connections show weights (green = positive, red = negative)
  - Line opacity indicates weight magnitude
- **Stats Panel**: Displays epoch, step, loss, and layer architecture details
- **Interactive Camera**: Use mouse to orbit, zoom, and pan around the neural network

## How It Works

1. The WebSocket hook (`useWebSocket.ts`) connects to `ws://localhost:8000/ws`
2. Receives `Snapshot` data with layer weights, biases, and training stats
3. The `DNNVisualizer` component renders each layer in 3D space:
   - Input nodes on one side, output nodes on the other
   - Weights visualized as colored lines between nodes
4. The `StatsPanel` shows training metrics overlaid on the visualization

## Controls

- **Left Mouse**: Rotate camera
- **Right Mouse / Two Fingers**: Pan camera
- **Scroll**: Zoom in/out

## Running

```bash
npm run dev
```

Make sure the Go WebSocket server is running on port 8000.
