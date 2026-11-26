// Tests of the resource manager of a node: the new connections it admits from one address,
// and the WARN lines of the ones it refuses.
package main

import (
	"bufio"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	golog "github.com/ipfs/go-log/v2"
	"github.com/libp2p/go-libp2p/core/connmgr"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/test"
	ma "github.com/multiformats/go-multiaddr"
)

// warnings records the lines a refusal log writes.
type warnings struct {
	mu    sync.Mutex
	lines []string
}

// nodeResourceManager builds the resource manager of a node and closes it at the end of the test.
func nodeResourceManager(t *testing.T, instanceIndex int) network.ResourceManager {
	t.Helper()
	rm, err := newNodeResourceManager(instanceIndex)
	if err != nil {
		t.Fatalf("resource manager: %v", err)
	}
	t.Cleanup(func() { rm.Close() })
	return rm
}

// admitUntilRefused opens new inbound connections from the endpoints given, closing each
// scope at once so that only the rate counts, until one is refused or max are admitted.
func admitUntilRefused(t *testing.T, rm network.ResourceManager, endpoint func(i int) string, max int) (int, time.Duration, error) {
	t.Helper()
	start := time.Now()
	for i := 0; i < max; i++ {
		scope, err := rm.OpenConnection(network.DirInbound, false, ma.StringCast(endpoint(i)))
		if err != nil {
			return i, time.Since(start), err
		}
		scope.Done()
	}
	return max, time.Since(start), nil
}

// checkBurst fails unless the first refusal came after the burst, within what the refill
// of the bucket can add while the connections were opened.
func checkBurst(t *testing.T, admitted int, elapsed time.Duration, err error) {
	t.Helper()
	if err == nil || !strings.Contains(err.Error(), "rate limit exceeded") {
		t.Fatalf("after %d admitted in %v, refusal %v; want one for the rate", admitted, elapsed, err)
	}
	refilled := int(elapsed.Seconds()*NewConnsPerSecondPerAddress) + 1
	if admitted < MaxConnsPerAddress || admitted > MaxConnsPerAddress+refilled {
		t.Fatalf("%d admitted in %v before the first refusal; want %d, plus at most %d refilled",
			admitted, elapsed, MaxConnsPerAddress, refilled)
	}
}

// captureWarnings sends the WARN lines of the loggers to the test until the function it
// returns is called, which gives back those of the resource manager.
func captureWarnings(t *testing.T) func() []string {
	t.Helper()
	previous, err := golog.SubsystemLevelName("unailib")
	if err != nil {
		t.Fatalf("level of unailib: %v", err)
	}
	if err := golog.SetLogLevel("unailib", "warn"); err != nil {
		t.Fatalf("level of unailib: %v", err)
	}
	t.Cleanup(func() { golog.SetLogLevel("unailib", previous) })

	// A write to the pipe returns once it is read: the lines written before Close are all
	// read before the scanner ends.
	pipe := golog.NewPipeReader(golog.PipeFormat(golog.PlaintextOutput), golog.PipeLevel(golog.LevelWarn))
	var lines []string
	done := make(chan struct{})
	go func() {
		defer close(done)
		scanner := bufio.NewScanner(pipe)
		for scanner.Scan() {
			if strings.Contains(scanner.Text(), "resource manager refused") {
				lines = append(lines, scanner.Text())
			}
		}
	}()
	return func() []string {
		pipe.Close()
		<-done
		return lines
	}
}

// A class behind one NAT reaches a public host from one address: forty nodes dialling it
// at once are admitted, where the default rate of go-libp2p refused the seventeenth.
func TestNodeResourceManagerAdmitsAClassBehindOneAddress(t *testing.T) {
	rm := nodeResourceManager(t, 0)
	admitted, _, err := admitUntilRefused(t, rm, func(i int) string {
		return fmt.Sprintf("/ip4/31.158.55.148/udp/%d/quic-v1", 58742+i)
	}, 40)
	if err != nil {
		t.Fatalf("connection %d from one address refused: %v", admitted+1, err)
	}
}

// The rate still bounds a flood from one address: past the burst a new connection from it
// is refused, one from another address is not, and the refusals make one WARN line.
func TestNodeConnRateLimitStillBitesPastTheBurst(t *testing.T) {
	stop := captureWarnings(t)
	rm := nodeResourceManager(t, 7)
	flooder := func(i int) string { return fmt.Sprintf("/ip4/31.158.55.148/tcp/%d", 10000+i) }
	admitted, elapsed, err := admitUntilRefused(t, rm, flooder, 2*MaxConnsPerAddress)
	checkBurst(t, admitted, elapsed, err)

	scope, err := rm.OpenConnection(network.DirInbound, true, ma.StringCast("/ip4/31.158.55.149/tcp/4001"))
	if err != nil {
		t.Fatalf("another address refused while the first was: %v", err)
	}
	scope.Done()
	if _, err := rm.OpenConnection(network.DirInbound, true, ma.StringCast(flooder(0))); err == nil {
		t.Fatal("the address that spent its burst was admitted again at once")
	}

	lines := stop()
	if len(lines) != 1 {
		t.Fatalf("WARN lines for two refusals of one address: %q; want one", lines)
	}
	for _, part := range []string{"WARN", "Instance 7", "31.158.55.148", "Inbound", flooder(admitted), "rate limit exceeded"} {
		if !strings.Contains(lines[0], part) {
			t.Errorf("the WARN line %q does not say %q", lines[0], part)
		}
	}
}

// A connection through a relay is counted at the address of the relay, whoever dials: the
// peers that reach a private node through one relay are admitted up to the burst too.
func TestNodeResourceManagerAdmitsRelayedConnectionsUpToTheBurst(t *testing.T) {
	rm := nodeResourceManager(t, 0)
	relay := test.RandPeerIDFatal(t)
	admitted, elapsed, err := admitUntilRefused(t, rm, func(int) string {
		return fmt.Sprintf("/ip4/185.31.40.10/udp/30060/quic-v1/p2p/%s/p2p-circuit", relay)
	}, 2*MaxConnsPerAddress)
	checkBurst(t, admitted, elapsed, err)
}

// Loopback is never limited, as in the default rate of go-libp2p.
func TestNodeResourceManagerNeverRefusesLoopback(t *testing.T) {
	rm := nodeResourceManager(t, 0)
	for _, ip := range []string{"/ip4/127.0.0.1", "/ip6/::1"} {
		admitted, _, err := admitUntilRefused(t, rm, func(i int) string {
			return fmt.Sprintf("%s/tcp/%d", ip, 10000+i)
		}, 3*MaxConnsPerAddress)
		if err != nil {
			t.Fatalf("connection %d from %s refused: %v", admitted+1, ip, err)
		}
	}
}

// libp2p checks its connection manager against the limit of the resource manager when this
// tells it, which the wrapper of the manager must keep doing.
func TestNodeResourceManagerTellsItsConnectionLimit(t *testing.T) {
	rm := nodeResourceManager(t, 0)
	limiter, ok := rm.(connmgr.GetConnLimiter)
	if !ok {
		t.Fatal("the resource manager of a node does not tell its connection limit")
	}
	if limit := limiter.GetConnLimit(); limit < 256 {
		t.Fatalf("connection limit %d, below the base of the system scope", limit)
	}
}

// Many refusals of one address within a window make one line at once and one with their
// count when the window ends; a window with none forgets the address.
func TestRefusalLogWritesOneLinePerAddressAndWindow(t *testing.T) {
	var w warnings
	refusals := newRefusalLog(3, 100*time.Millisecond, 4, w.warnf)
	defer refusals.close()
	endpoint := ma.StringCast("/ip4/31.158.55.148/udp/58742/quic-v1")
	for range 40 {
		refusals.record(network.DirInbound, endpoint, fmt.Errorf("rate limit exceeded"))
	}
	if lines := w.all(); len(lines) != 1 || !strings.Contains(lines[0], "refused a connection of 31.158.55.148") {
		t.Fatalf("lines right after 40 refusals: %q; want the first one only", lines)
	}

	waitUntil(t, "the line of the window", func() bool { return len(w.all()) == 2 })
	if line := w.all()[1]; !strings.Contains(line, "Instance 3") || !strings.Contains(line, "39 more connections of 31.158.55.148") ||
		!strings.Contains(line, "Inbound") || !strings.Contains(line, "rate limit exceeded") {
		t.Fatalf("line of the window %q; want the count of the 39 after the first", line)
	}
	waitUntil(t, "the address forgotten", func() bool {
		refusals.mu.Lock()
		defer refusals.mu.Unlock()
		return len(refusals.windows) == 0
	})
	if lines := w.all(); len(lines) != 2 {
		t.Fatalf("lines after a window with no refusal: %q", lines)
	}
}

// Past the addresses counted apart, the refusals of any other address are counted together,
// and closing the log writes what it still counted.
func TestRefusalLogCountsTheAddressesBeyondTheLimitTogether(t *testing.T) {
	var w warnings
	refusals := newRefusalLog(0, time.Hour, 2, w.warnf)
	for i := range 10 {
		endpoint := ma.StringCast(fmt.Sprintf("/ip4/31.158.55.%d/tcp/4001", 100+i))
		for range 3 {
			refusals.record(network.DirInbound, endpoint, fmt.Errorf("rate limit exceeded"))
		}
	}
	refusals.mu.Lock()
	windows := len(refusals.windows)
	refusals.mu.Unlock()
	if lines := w.all(); len(lines) != 3 || windows != 3 {
		t.Fatalf("%d windows and lines %q; want two addresses apart and the others together", windows, lines)
	}

	refusals.close()
	text := strings.Join(w.all(), "\n")
	for _, part := range []string{"2 more connections of 31.158.55.100", "2 more connections of 31.158.55.101",
		"23 more connections of an address beyond the 2 counted apart"} {
		if !strings.Contains(text, part) {
			t.Errorf("lines %q do not say %q", text, part)
		}
	}
	refusals.record(network.DirInbound, ma.StringCast("/ip4/31.158.55.100/tcp/4001"), fmt.Errorf("rate limit exceeded"))
	if lines := w.all(); len(lines) != 6 {
		t.Fatalf("lines %q; want none after the close", lines)
	}
}

// warnf records a line.
func (w *warnings) warnf(template string, args ...any) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.lines = append(w.lines, fmt.Sprintf(template, args...))
}

// all returns the lines recorded so far.
func (w *warnings) all() []string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return append([]string(nil), w.lines...)
}
