// The values unailib must hold equal to the browser transport, compared with the table both sides read.
package main

import (
	"encoding/json"
	"os"
	"testing"
	"time"
)

// sharedTransportValuesFile is the table of those values. It lives in the Python package, outside this folder,
// so that the pure wheel carries it to the SDK, whose test reads it there.
const sharedTransportValuesFile = "../shared_transport_values.json"

// sharedTransportValue is one entry of the table.
type sharedTransportValue struct {
	Name  string `json:"name"`
	Value int64  `json:"value"`
	Unit  string `json:"unit"`
	Step  string `json:"step"`
}

// sharedTransportConstants gives, for each entry of the table, the constant of unailib that must hold its value.
func sharedTransportConstants() map[string]time.Duration {
	return map[string]time.Duration{
		"disconnection_grace":      DisconnectionGracePeriod,
		"session_frame_send":       SessionSendTimeout,
		"relay_reservation":        RelayReservationTimeout,
		"webrtc_signaling":         WebRTCSignalingTimeout,
		"webrtc_direct_stage":      WebRTCDirectStageTimeout,
		"webrtc_stage_pause":       WebRTCStagePause,
		"webrtc_gather_quiet":      WebRTCGatherQuiet,
		"webrtc_direct_gather_max": WebRTCDirectGatherMax,
		"webrtc_turn_gather_max":   WebRTCTURNGatherMax,
		"webrtc_freshness":         WebRTCFreshnessWindow,
		"webrtc_reassembly":        WebRTCReassemblyTimeout,
	}
}

// A constant of unailib that differs from its entry, an entry with no constant and a constant with no entry all
// fail: the browser transport is compared with the same table, so a value changed on one side only is a drift.
func TestSharedTransportValuesMatchUnailib(t *testing.T) {
	data, err := os.ReadFile(sharedTransportValuesFile)
	if err != nil {
		t.Fatalf("table of the shared values: %v", err)
	}
	var table struct {
		Values []sharedTransportValue `json:"values"`
	}
	if err := json.Unmarshal(data, &table); err != nil {
		t.Fatalf("table of the shared values: %v", err)
	}
	units := map[string]time.Duration{"ms": time.Millisecond, "s": time.Second}
	constants := sharedTransportConstants()
	listed := make(map[string]bool)
	for _, entry := range table.Values {
		if listed[entry.Name] {
			t.Errorf("%q is listed twice", entry.Name)
		}
		listed[entry.Name] = true
		if entry.Step == "" {
			t.Errorf("%q does not say what step it bounds", entry.Name)
		}
		unit, ok := units[entry.Unit]
		if !ok {
			t.Errorf("%q has the unit %q, not one of ms and s", entry.Name, entry.Unit)
			continue
		}
		constant, ok := constants[entry.Name]
		if !ok {
			t.Errorf("%q has no constant in unailib", entry.Name)
			continue
		}
		if want := time.Duration(entry.Value) * unit; constant != want {
			t.Errorf("%q is %v in unailib and %v in the table", entry.Name, constant, want)
		}
	}
	for name := range constants {
		if !listed[name] {
			t.Errorf("%q has a constant in unailib and no entry in the table", name)
		}
	}
}
