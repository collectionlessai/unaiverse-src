// The resource manager of a node: the limits of go-libp2p as this library raises them, a
// rate of new connections per address that fits many nodes behind one address, and a WARN
// line for the connections it refuses.
package main

import (
	"errors"
	"fmt"
	"net/netip"
	"sync"
	"time"

	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/connmgr"
	"github.com/libp2p/go-libp2p/core/network"
	rcmgr "github.com/libp2p/go-libp2p/p2p/host/resource-manager"
	"github.com/libp2p/go-libp2p/x/rate"
	ma "github.com/multiformats/go-multiaddr"
	manet "github.com/multiformats/go-multiaddr/net"
)

// After the first refusal of an address, its next ones are counted for refusalLogWindow and
// written in one line; refusalLogMaxAddrs addresses are counted apart at a time, and the
// refusals of any other address are counted together.
const (
	refusalLogWindow   = time.Minute
	refusalLogMaxAddrs = 32
)

// refusalOthers is the key of the refusals counted together, which no address can have.
const refusalOthers = "others"

var errNoConnLimit = errors.New("the resource manager of go-libp2p does not tell its connection limit")

// resourceManagerWithLimit is what libp2p asks of the resource manager of a host: the
// interface, and the connection limit that config.go checks the connection manager against
// when the resource manager tells it.
type resourceManagerWithLimit interface {
	network.ResourceManager
	connmgr.GetConnLimiter
}

// loggingResourceManager is the resource manager of go-libp2p with a WARN line for the
// connections it refuses, which go-libp2p logs at debug level only.
type loggingResourceManager struct {
	resourceManagerWithLimit
	refusals *refusalLog
}

// refusalLog writes the connections a resource manager refuses at WARN, at most one line
// per remote address and window: the first refusal at once, and the next ones with their
// count when the window ends. While refusals keep coming a new window follows, and with
// none the address is forgotten. A flood thus turns neither into a flood of lines nor into
// a map that grows.
type refusalLog struct {
	instance int
	window   time.Duration
	maxAddrs int
	warnf    func(template string, args ...any)

	mu      sync.Mutex
	closed  bool
	windows map[string]*refusalWindow
}

// refusalWindow counts the refusals of one remote address after the one that opened it.
type refusalWindow struct {
	count    int
	dir      network.Direction
	endpoint ma.Multiaddr
	err      error
	timer    *time.Timer
}

// newNodeResourceManager builds the resource manager that CreateNode gives to libp2p.
func newNodeResourceManager(instanceIndex int) (network.ResourceManager, error) {
	scalingLimits := rcmgr.DefaultLimits
	libp2p.SetDefaultServiceLimits(&scalingLimits)

	// These apply per unique Peer ID.
	scalingLimits.PeerBaseLimit.Conns = 64
	scalingLimits.PeerBaseLimit.ConnsInbound = 64
	scalingLimits.PeerBaseLimit.ConnsOutbound = 64

	// Tweak System Limits
	scalingLimits.SystemBaseLimit.Conns = 256
	scalingLimits.SystemBaseLimit.ConnsInbound = 128
	scalingLimits.SystemBaseLimit.ConnsOutbound = 128

	// Compute the concrete limits
	scaledLimits := scalingLimits.AutoScale()

	// Raise the per-IP limits
	customIP4Limits := []rcmgr.ConnLimitPerSubnet{
		{
			PrefixLength: 32,                 // /32 means "one specific IP address"
			ConnCount:    MaxConnsPerAddress, // Allow 1024 conns from the same IP
		},
	}
	customIP6Limits := []rcmgr.ConnLimitPerSubnet{
		{
			PrefixLength: 56,
			ConnCount:    MaxConnsPerAddress,
		},
	}

	// Create the limiter and manager
	limiter := rcmgr.NewFixedLimiter(scaledLimits)
	rm, err := rcmgr.NewResourceManager(
		limiter,
		rcmgr.WithLimitPerSubnet(customIP4Limits, customIP6Limits),
		rcmgr.WithConnRateLimiters(nodeConnRateLimiter()),
	)
	if err != nil {
		return nil, err
	}
	withLimit, ok := rm.(resourceManagerWithLimit)
	if !ok {
		rm.Close()
		return nil, errNoConnLimit
	}
	return &loggingResourceManager{
		resourceManagerWithLimit: withLimit,
		refusals:                 newRefusalLog(instanceIndex, refusalLogWindow, refusalLogMaxAddrs, logger.Warnf),
	}, nil
}

// nodeConnRateLimiter is the rate limiter of new connections that go-libp2p builds by
// default, with the numbers of MaxConnsPerAddress and NewConnsPerSecondPerAddress: loopback
// is never limited, there is no global limit, and an address is forgotten a minute after
// its bucket is full again.
func nodeConnRateLimiter() *rate.Limiter {
	perAddress := rate.Limit{RPS: NewConnsPerSecondPerAddress, Burst: MaxConnsPerAddress}
	return &rate.Limiter{
		NetworkPrefixLimits: []rate.PrefixLimit{
			{Prefix: netip.MustParsePrefix("127.0.0.0/8"), Limit: rate.Limit{}},
			{Prefix: netip.MustParsePrefix("::1/128"), Limit: rate.Limit{}},
		},
		GlobalLimit: rate.Limit{},
		SubnetRateLimiter: rate.SubnetLimiter{
			IPv4SubnetLimits: []rate.SubnetLimit{{PrefixLength: 32, Limit: perAddress}},
			IPv6SubnetLimits: []rate.SubnetLimit{{PrefixLength: 56, Limit: perAddress}},
			GracePeriod:      time.Minute,
		},
	}
}

// OpenConnection asks the manager wrapped, and records a refusal.
func (m *loggingResourceManager) OpenConnection(dir network.Direction, usefd bool, endpoint ma.Multiaddr) (network.ConnManagementScope, error) {
	scope, err := m.resourceManagerWithLimit.OpenConnection(dir, usefd, endpoint)
	if err != nil {
		m.refusals.record(dir, endpoint, err)
	}
	return scope, err
}

// Close writes the refusals still counted and closes the manager wrapped.
func (m *loggingResourceManager) Close() error {
	m.refusals.close()
	return m.resourceManagerWithLimit.Close()
}

// newRefusalLog returns a log of refusals of the given instance that writes through warnf.
func newRefusalLog(instance int, window time.Duration, maxAddrs int, warnf func(string, ...any)) *refusalLog {
	return &refusalLog{
		instance: instance,
		window:   window,
		maxAddrs: maxAddrs,
		warnf:    warnf,
		windows:  make(map[string]*refusalWindow),
	}
}

// record writes a refusal at once if it opens a window, and counts it otherwise.
func (l *refusalLog) record(dir network.Direction, endpoint ma.Multiaddr, err error) {
	key := refusalKey(endpoint)
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return
	}
	w, open := l.windows[key]
	if !open && l.countedApart() >= l.maxAddrs {
		key = refusalOthers
		w, open = l.windows[key]
	}
	if open {
		w.count++
		w.dir, w.endpoint, w.err = dir, endpoint, err
		return
	}

	// The timer takes the lock this call holds, so it finds the window in the map
	l.windows[key] = &refusalWindow{timer: time.AfterFunc(l.window, func() { l.endWindow(key) })}
	l.warnf("[GO] ⚠️ Instance %d: resource manager refused a connection of %s (%s, remote %s): %v; more within %v are counted in one line",
		l.instance, l.subject(key), dir, endpoint, err, l.window)
}

// close writes the refusals still counted and stops the windows.
func (l *refusalLog) close() {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return
	}
	l.closed = true
	for key, w := range l.windows {
		w.timer.Stop()
		if w.count > 0 {
			l.writeCount(key, w)
		}
	}
	clear(l.windows)
}

// countedApart is the number of addresses whose refusals have a window of their own.
func (l *refusalLog) countedApart() int {
	if _, ok := l.windows[refusalOthers]; ok {
		return len(l.windows) - 1
	}
	return len(l.windows)
}

// endWindow writes the refusals counted in a window, and opens the next one if there were any.
func (l *refusalLog) endWindow(key string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	w, open := l.windows[key]
	if l.closed || !open {
		return
	}
	if w.count == 0 {
		delete(l.windows, key)
		return
	}
	l.writeCount(key, w)
	w.count = 0
	w.timer.Reset(l.window)
}

// subject names whose refusals a window counts.
func (l *refusalLog) subject(key string) string {
	switch key {
	case refusalOthers:
		return fmt.Sprintf("an address beyond the %d counted apart", l.maxAddrs)
	case "":
		return "an endpoint with no IP"
	}
	return key
}

// writeCount writes the line of the refusals a window counted after its first.
func (l *refusalLog) writeCount(key string, w *refusalWindow) {
	l.warnf("[GO] ⚠️ Instance %d: resource manager refused %d more connections of %s within %v (the last %s, remote %s): %v",
		l.instance, w.count, l.subject(key), l.window, w.dir, w.endpoint, w.err)
}

// refusalKey is the IP of a remote endpoint, by which the limits of go-libp2p count an IPv4
// address, or "" for an endpoint with no IP.
func refusalKey(endpoint ma.Multiaddr) string {
	ip, err := manet.ToIP(endpoint)
	if err != nil {
		return ""
	}
	return ip.String()
}
