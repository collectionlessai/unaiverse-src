// The byte-exact vectors of the chunk frames, shared with the browser SDK (plan J5): the JS suite reads a copy of
// testdata/chunk_frame_vectors.json and must build and read the same bytes.
package main

import (
	"encoding/hex"
	"encoding/json"
	"os"
	"testing"
)

// chunkVector is one chunk frame and what built it.
type chunkVector struct {
	Name    string `json:"name"`
	Seq     uint64 `json:"seq"`
	Index   uint32 `json:"index"`
	Attempt uint8  `json:"attempt"`
	File    bool   `json:"file"`
	First   bool   `json:"first"`
	Last    bool   `json:"last"`
	Total   uint64 `json:"total"`
	Channel string `json:"channel"`
	Data    string `json:"data"`  // UTF-8
	Frame   string `json:"frame"` // Hex
}

// chunkVectorCases are the frames of the vectors (Frame filled by buildChunkFrameOf).
var chunkVectorCases = []chunkVector{
	{Name: "single chunk (first and last)", Seq: 1, Index: 0, Attempt: 0, First: true, Last: true, Total: 5, Channel: "ch", Data: "hello"},
	{Name: "first of three, attempt 3", Seq: 0x0010203040506070, Index: 0, Attempt: 3, First: true, Total: 10, Channel: "12D3KooWPeer::dm:12D3KooWOther-interaction", Data: "abcde"},
	{Name: "middle", Seq: 0x0010203040506070, Index: 1, Attempt: 3, Total: 10, Data: "fgh"},
	{Name: "last, attempt 31", Seq: 0x0010203040506070, Index: 2, Attempt: 31, Last: true, Total: 10, Data: "ij"},
	{Name: "first of a file message", Seq: 7, Index: 0, Attempt: 1, File: true, First: true, Total: 3, Channel: "f", Data: "xyz"},
}

// TestChunkFrameVectors checks the frames against the shared vectors; with UNAI_WRITE_CHUNK_VECTORS=1 it writes them.
func TestChunkFrameVectors(t *testing.T) {
	built := make([]chunkVector, len(chunkVectorCases))
	for i, c := range chunkVectorCases {
		frame := buildChunkFrameOf(c.Seq, c.Index, c.Attempt, c.File, c.First, c.Last, c.Total, c.Channel, []byte(c.Data))
		c.Frame = hex.EncodeToString(frame)
		built[i] = c
	}
	path := "testdata/chunk_frame_vectors.json"
	if os.Getenv("UNAI_WRITE_CHUNK_VECTORS") == "1" {
		out, _ := json.MarshalIndent(built, "", "  ")
		if err := os.MkdirAll("testdata", 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, append(out, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v (write it with UNAI_WRITE_CHUNK_VECTORS=1)", path, err)
	}
	var want []chunkVector
	if err := json.Unmarshal(raw, &want); err != nil {
		t.Fatal(err)
	}
	if len(want) != len(built) {
		t.Fatalf("%d vectors in %s, %d built", len(want), path, len(built))
	}
	for i := range built {
		if built[i] != want[i] {
			t.Errorf("vector %q: built %s, file says %s", built[i].Name, built[i].Frame, want[i].Frame)
		}
	}
}
