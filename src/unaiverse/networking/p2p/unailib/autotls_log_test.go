// Tests of the log configuration toward the AutoTLS certificate manager.
package main

import (
	"slices"
	"testing"

	"github.com/caddyserver/certmagic"
	golog "github.com/ipfs/go-log/v2"
	p2pforge "github.com/ipshipyard/p2p-forge/client"
)

// InitializeLibrary sets the levels by name before any certificate manager exists, and a
// logger created later would start at the default level instead of the configured one.
func TestLogConfigurationReachesTheCertificateManager(t *testing.T) {
	if err := golog.SetLogLevel(P2PForgeLoggerName, "info"); err != nil {
		t.Fatalf("level of %q before a manager exists: %v", P2PForgeLoggerName, err)
	}
	t.Cleanup(func() { golog.SetLogLevel(P2PForgeLoggerName, "fatal") })

	known := golog.GetSubsystems()
	_, err := p2pforge.NewP2PForgeCertMgr(
		p2pforge.WithCertificateStorage(&certmagic.FileStorage{Path: t.TempDir()}),
	)
	if err != nil {
		t.Fatalf("certificate manager: %v", err)
	}
	// A manager that named its logger otherwise would have added one the configuration
	// does not know.
	for _, name := range golog.GetSubsystems() {
		if !slices.Contains(known, name) {
			t.Errorf("the certificate manager logs as %q, which no level was set for", name)
		}
	}
}
