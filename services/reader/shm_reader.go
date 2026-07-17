package reader

// ---------------------------------------------------------------------------
// shm_reader.go
//
// SharedMemoryReader opens a Windows named shared memory region (read-only),
// polls it at a configurable interval, and emits new snapshots whenever the
// C++ version counter advances.
//
// Version-check protocol (matches C++ writer):
//   1. Read version  (acquire)
//   2. If version == lastVersion → skip
//   3. Read the snapshot
//   4. Read version again
//   5. If versions differ → discard (writer was mid-write)
//   6. Otherwise → accept snapshot, update lastVersion, print
// ---------------------------------------------------------------------------

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"strings"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"

	"golang.org/x/sys/windows"
)

// openFileMapping wraps the Win32 OpenFileMappingW syscall, which is not
// exposed by golang.org/x/sys/windows.
var (
	modKernel32         = syscall.NewLazyDLL("kernel32.dll")
	procOpenFileMapping = modKernel32.NewProc("OpenFileMappingW")
)

func openFileMapping(desiredAccess uint32, inheritHandle bool, name *uint16) (windows.Handle, error) {
	inherit := uintptr(0)
	if inheritHandle {
		inherit = 1
	}
	r, _, err := procOpenFileMapping.Call(
		uintptr(desiredAccess),
		inherit,
		uintptr(unsafe.Pointer(name)),
	)
	if r == 0 {
		return 0, err
	}
	return windows.Handle(r), nil
}

// ---------------------------------------------------------------------------
// SharedMemoryReader
// ---------------------------------------------------------------------------

// SharedMemoryReader opens and continuously reads a named shared memory region.
type SharedMemoryReader struct {
	name        string
	handle      windows.Handle
	data        []byte // mapped view slice
	lastVersion uint64
}

// NewSharedMemoryReader creates a reader for the given shared memory name.
func NewSharedMemoryReader(name string) *SharedMemoryReader {
	return &SharedMemoryReader{name: name}
}

// Open opens the named shared memory region as read-only.
// The region must already exist (created by the C++ writer).
func (r *SharedMemoryReader) Open() error {
	namePtr, err := windows.UTF16PtrFromString(r.name)
	if err != nil {
		return fmt.Errorf("invalid shm name %q: %w", r.name, err)
	}

	// Open existing mapping — read-only.
	handle, err := openFileMapping(windows.FILE_MAP_READ, false, namePtr)
	if err != nil {
		return fmt.Errorf("OpenFileMapping(%q) failed: %w", r.name, err)
	}
	r.handle = handle

	// Map the entire region. Pass 0 for size to map the full object.
	addr, err := windows.MapViewOfFile(handle, windows.FILE_MAP_READ, 0, 0, 0)
	if err != nil {
		windows.CloseHandle(handle)
		return fmt.Errorf("MapViewOfFile failed: %w", err)
	}

	// Determine the mapped size via VirtualQuery so we know how many bytes
	// are accessible without needing the size from the caller.
	var memInfo windows.MemoryBasicInformation
	err = windows.VirtualQuery(addr, &memInfo, unsafe.Sizeof(memInfo))
	if err != nil {
		windows.UnmapViewOfFile(addr)
		windows.CloseHandle(handle)
		return fmt.Errorf("VirtualQuery failed: %w", err)
	}

	size := memInfo.RegionSize
	// Build a byte slice backed by the mapped memory.
	// This is safe as long as we hold the mapping open.
	r.data = unsafe.Slice((*byte)(unsafe.Pointer(addr)), size)

	return nil
}

// Close unmaps the view and closes the mapping handle.
func (r *SharedMemoryReader) Close() {
	if r.data != nil {
		addr := uintptr(unsafe.Pointer(&r.data[0]))
		windows.UnmapViewOfFile(addr)
		r.data = nil
	}
	if r.handle != 0 {
		windows.CloseHandle(r.handle)
		r.handle = 0
	}
}

// Poll reads the shared memory once.
// Returns a non-nil *Snapshot if a new version was detected and validated.
func (r *SharedMemoryReader) Poll() *Snapshot {
	if len(r.data) < shmHeaderSize {
		return nil
	}

	// Step 1: Read version with acquire semantics.
	v1 := r.readVersion()

	// Step 2: Skip if nothing changed.
	if v1 == r.lastVersion {
		return nil
	}

	// Step 3: Read the full snapshot.
	snap, ok := r.ReadSnapshot()
	if !ok {
		return nil
	}

	// Step 4: Re-read version to detect torn writes.
	v2 := r.readVersion()

	// Step 5: Discard if writer modified data during our read.
	if v1 != v2 {
		return nil
	}

	// Step 6: Accept.
	r.lastVersion = v1
	snap.Version = v1
	return snap
}

// ReadSnapshot deserializes the current shared memory contents into a Snapshot.
// Returns (snapshot, true) on success, (nil, false) if the buffer is malformed.
func (r *SharedMemoryReader) ReadSnapshot() (*Snapshot, bool) {
	buf := r.data

	if len(buf) < shmHeaderSize {
		return nil, false
	}

	// --- ShmHeader ---
	offset := 0
	// version is at offset 0 (8 bytes) — already read by Poll; skip here.
	offset += 8
	numLayers := readUint32(buf, offset)
	offset += 4
	epoch := readUint32(buf, offset)
	offset += 4
	step := readUint32(buf, offset)
	offset += 4
	loss := readFloat32(buf, offset)
	offset += 4

	snap := &Snapshot{
		Epoch:  epoch,
		Step:   step,
		Loss:   loss,
		Layers: make([]LayerSnapshot, 0, numLayers),
	}

	// --- Per-layer data ---
	for i := uint32(0); i < numLayers; i++ {
		if offset+layerHeaderSize > len(buf) {
			return nil, false
		}

		rows := readUint32(buf, offset)
		offset += 4
		cols := readUint32(buf, offset)
		offset += 4
		biasSize := readUint32(buf, offset)
		offset += 4

		wCount := int(rows) * int(cols)
		bCount := int(biasSize)

		wBytes := wCount * 4
		bBytes := bCount * 4

		if offset+wBytes+bBytes > len(buf) {
			return nil, false
		}

		weights := readFloats(buf, offset, wCount)
		offset += wBytes
		biases := readFloats(buf, offset, bCount)
		offset += bBytes

		snap.Layers = append(snap.Layers, LayerSnapshot{
			Header:  LayerHeader{Rows: rows, Cols: cols, BiasSize: biasSize},
			Weights: weights,
			Biases:  biases,
		})
	}

	return snap, true
}

// PrintSnapshot writes a formatted snapshot to stdout.
func PrintSnapshot(s *Snapshot) {
	divider := strings.Repeat("-", 50)
	thin := strings.Repeat("-", 34)

	fmt.Println(divider)
	fmt.Printf("Version : %d\n", s.Version)
	fmt.Printf("Epoch   : %d\n", s.Epoch)
	fmt.Printf("Step    : %d\n", s.Step)
	fmt.Printf("Loss    : %f\n", s.Loss)

	for i, layer := range s.Layers {
		fmt.Println(thin)
		fmt.Printf("Layer %d\n", i)
		fmt.Printf("  Rows : %d\n", layer.Header.Rows)
		fmt.Printf("  Cols : %d\n", layer.Header.Cols)
		fmt.Printf("  Bias : %d\n", layer.Header.BiasSize)

		fmt.Println("  Weights")
		for _, w := range layer.Weights {
			fmt.Printf("    %s\n", formatFloat(w))
		}

		fmt.Println("  Bias")
		for _, b := range layer.Biases {
			fmt.Printf("    %s\n", formatFloat(b))
		}
	}

	fmt.Println(divider)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// readVersion reads the version field with acquire semantics via atomic load.
// The version is the first 8 bytes of the shared memory buffer.
func (r *SharedMemoryReader) readVersion() uint64 {
	ptr := (*uint64)(unsafe.Pointer(&r.data[0]))
	return atomic.LoadUint64(ptr)
}

func readUint32(buf []byte, offset int) uint32 {
	return binary.LittleEndian.Uint32(buf[offset : offset+4])
}

func readFloat32(buf []byte, offset int) float32 {
	bits := binary.LittleEndian.Uint32(buf[offset : offset+4])
	return math.Float32frombits(bits)
}

func readFloats(buf []byte, offset, count int) []float32 {
	out := make([]float32, count)
	for i := range out {
		out[i] = readFloat32(buf, offset+i*4)
	}
	return out
}

func formatFloat(f float32) string {
	return fmt.Sprintf("%.6f", f)
}

// Run starts the polling loop. It blocks until the provided stop channel is closed.
func (r *SharedMemoryReader) Run(interval time.Duration, stop context.Context, out chan *Snapshot) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	fmt.Printf("Polling shared memory %q every %v...\n", r.name, interval)

	for {
		select {
		case <-stop.Done():
			return
		case <-ticker.C:
			if snap := r.Poll(); snap != nil {
				fmt.Println("Wrote in channel")
				out <- snap
			}
		}
	}
}
