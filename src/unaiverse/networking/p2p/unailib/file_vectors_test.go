// The byte-exact vectors of a file message, shared with the browser SDK (plan J6): the JS suite reads a copy of
// testdata/file_message_vectors.json and must build the same frames and read the files back from them.
package main

import (
	"encoding/hex"
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

// fileVector is a file message, the chunk size it was cut with, and its chunk frames.
type fileVector struct {
	Name      string           `json:"name"`
	ChunkSize int              `json:"chunk_size"`
	Seq       uint64           `json:"seq"`
	Attempt   uint8            `json:"attempt"`
	Channel   string           `json:"channel"`
	Msg       string           `json:"msg"` // Hex
	Files     []fileVectorFile `json:"files"`
	Frames    []string         `json:"frames"` // Hex, each with its 4-byte length
}

// fileVectorFile is a file of a vector.
type fileVectorFile struct {
	Key     string `json:"key"`
	Name    string `json:"name"`
	Content string `json:"content"` // Hex
}

// fileVectorCases are the vectors (Frames filled by the test).
var fileVectorCases = []fileVector{
	{
		Name: "two files, one empty, cut in several chunks", ChunkSize: 64, Seq: 0x0102030405060708, Attempt: 2,
		Channel: "12D3KooWPeer::dm:12D3KooWOther-stream_sample", Msg: hex.EncodeToString([]byte("protobuf bytes of the message")),
		Files: []fileVectorFile{
			{Key: "processor", Name: "report.txt", Content: hex.EncodeToString([]byte("The quick brown fox jumps over the lazy dog, then over the lazy cat, again and again."))},
			{Key: "empty", Name: "nothing.bin", Content: ""},
		},
	},
	{
		Name: "one small file, a single chunk", ChunkSize: 1024, Seq: 9, Attempt: 0,
		Channel: "c", Msg: hex.EncodeToString([]byte("m")),
		Files: []fileVectorFile{{Key: "f", Name: "a.png", Content: hex.EncodeToString([]byte{0, 1, 2, 3, 255})}},
	},
}

// TestFileMessageVectors checks the frames against the shared vectors; with UNAI_WRITE_CHUNK_VECTORS=1 it writes them.
func TestFileMessageVectors(t *testing.T) {
	built := make([]fileVector, len(fileVectorCases))
	for i, c := range fileVectorCases {
		withFiles(t, c.ChunkSize, 0)
		msg, _ := hex.DecodeString(c.Msg)
		refs := make([]fileRef, len(c.Files))
		for j, f := range c.Files {
			content, _ := hex.DecodeString(f.Content)
			refs[j] = fileRef{Key: f.Key, Name: f.Name, Path: writeTemp(t, f.Name, content)}
		}
		fp, err := newFilePayload(c.Channel, msg, refs)
		if err != nil {
			t.Fatal(err)
		}
		fp.attempt = c.Attempt
		c.Frames = nil
		for _, frame := range fileFrames(t, c.Seq, fp) {
			c.Frames = append(c.Frames, hex.EncodeToString(frame))
		}
		built[i] = c
	}
	path := "testdata/file_message_vectors.json"
	if os.Getenv("UNAI_WRITE_CHUNK_VECTORS") == "1" {
		out, _ := json.MarshalIndent(built, "", "  ")
		if err := os.WriteFile(path, append(out, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v (write it with UNAI_WRITE_CHUNK_VECTORS=1)", path, err)
	}
	var want []fileVector
	if err := json.Unmarshal(raw, &want); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(want, built) {
		t.Errorf("the file message frames differ from %s", path)
	}
}
