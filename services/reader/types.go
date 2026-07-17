package reader

// ---------------------------------------------------------------------------
// types.go
//
// Go representations of the C++ shared memory layout.
// All structs match the C++ side which uses #pragma pack(1) — no padding.
//
// ShmHeader  (24 bytes)
//   version    uint64   (8)
//   num_layers uint32   (4)
//   epoch      uint32   (4)
//   step       uint32   (4)
//   loss       float32  (4)
//
// LayerHeader (12 bytes)
//   rows       uint32   (4)
//   cols       uint32   (4)
//   bias_size  uint32   (4)
//
// Followed immediately by:
//   weights    [rows * cols]float32
//   biases     [bias_size]float32
// ---------------------------------------------------------------------------

const (
	shmHeaderSize  = 24 // sizeof(ShmHeader)  — packed, no padding
	layerHeaderSize = 12 // sizeof(LayerHeader) — packed, no padding
)

// ShmHeader mirrors the C++ ShmHeader struct.
type ShmHeader struct {
	Version   uint64
	NumLayers uint32
	Epoch     uint32
	Step      uint32
	Loss      float32
}

// LayerHeader mirrors the C++ LayerHeader struct.
type LayerHeader struct {
	Rows     uint32
	Cols     uint32
	BiasSize uint32
}

// LayerSnapshot holds a fully deserialized layer from shared memory.
type LayerSnapshot struct {
	Header  LayerHeader
	Weights []float32
	Biases  []float32
}

// Snapshot is a fully deserialized frame from shared memory.
type Snapshot struct {
	Version uint64
	Epoch   uint32
	Step    uint32
	Loss    float32
	Layers  []LayerSnapshot
}
