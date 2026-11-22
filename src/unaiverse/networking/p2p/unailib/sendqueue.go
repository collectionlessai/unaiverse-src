// Here we define the per-peer send queues of the direct messages and their writers.
package main

import (
	"container/list"
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/libp2p/go-libp2p/core/peer"
	pwebrtc "github.com/pion/webrtc/v4"
)

// A send of a direct message only queues its frame: SendMessageToPeer returns at once,
// whatever the state of the peer. One writer goroutine per peer writes the queue in order,
// and owns everything that can wait: the choice between the WebRTC DataChannel and the
// chat stream, the dial of the stream, and the write under its deadline.
//
// The queues are bounded, per peer in bytes and in messages, and in bytes over all the
// peers. When a frame does not fit it is refused, and what is queued is never touched:
// losing a message in the middle of a sequence can break the pipeline a world is built
// on, while a refusal is seen by the caller. Once a queue refused a frame it keeps
// refusing until it has drained to the low-water mark, so that a burst cannot see a frame
// refused and the next one accepted (a gap in the sequence).
//
// A frame whose write failed stays at the head of the queue: the stream is closed, and the
// frame is written again from its first byte on a new one, after a backoff. The receiver
// dropped the partial copy with the old stream, so the framing is clean. Nothing is ever
// skipped: the queue is dropped whole, and only after maxDialAttempts failed writes in a
// row, when its head waited maxWaitAge with no progress, or when DropSendQueue says so.
//
// Every accepted frame gets a sequence number of its peer, and the events of the queues
// (written up to a number, refused, failed, recovered, dropped) are recorded for Python,
// which pulls them with PopSendEvents. Go never calls Python.

// The types of the send events.
const (
	SendEventWritten      = "written"
	SendEventWriteFailed  = "write_failed"
	SendEventRecovered    = "recovered"
	SendEventRefused      = "refused"
	SendEventQueueDropped = "queue_dropped"
	SendEventEventsLost   = "events_lost"

	// Progress of the messages larger than a chunk (chunked.go, file.go), throttled
	SendEventProgress        = "progress"         // Sender: bytes of message seq written, of total
	SendEventReceiveProgress = "receive_progress" // Receiver: bytes of message seq of the peer received, of total
	SendEventReceiveDropped  = "receive_dropped"  // Receiver: message seq of the peer dropped, with the reason
)

// Why a frame was refused (the reason of a refused event).
const (
	SendRefusedPeerMsgs      = "peer_msgs"       // the message cap of the peer
	SendRefusedPeerBytes     = "peer_bytes"      // the byte cap of the peer
	SendRefusedTotalBytes    = "total_bytes"     // the byte cap over all the peers
	SendRefusedLowWater      = "low_water"       // the peer has not drained to its low-water mark yet
	SendRefusedLowWaterTotal = "low_water_total" // all the peers have not drained to the low-water mark yet
	SendRefusedTooLarge      = "frame_too_large" // the frame alone exceeds a byte cap
)

// Why a queue was dropped (the reason of a queue_dropped event).
const (
	SendDroppedMaxAttempts = "max_attempts"
	SendDroppedMaxAge      = "max_age"
	SendDroppedExplicit    = "explicit"
)

// The error codes of the replies of SendMessageToPeer, for Python to tell them apart.
const (
	SendCodeQueueFull     = "queue_full"
	SendCodeNotConfigured = "send_queue_not_configured"
	SendCodeInvalidPeer   = "invalid_peer"
	SendCodeSelfSend      = "self_send"
	SendCodeTooLarge      = "too_large"
	SendCodeNoFileStream  = "no_file_stream"
)

// SendQueueConfig holds the limits of the send queues. All of them come from Python
// (custom.py, through ConfigureSendQueues): Go has no defaults of its own.
type SendQueueConfig struct {
	MaxBytesPerPeer int64           // byte cap of the queue of one peer
	MaxMsgsPerPeer  int             // message cap of the queue of one peer
	MaxBytesTotal   int64           // byte cap over the queues of all the peers
	LowWater        float64         // a queue that refused accepts again at this fraction of its caps
	MaxWaitAge      time.Duration   // the queue is dropped when its head waited this long with no progress
	MaxDialAttempts int             // the queue is dropped after this many failed writes in a row
	RetryBackoff    []time.Duration // wait before the retry after the n-th failed write (the last one repeats)
	MaxEvents       int             // events kept for PopSendEvents; the oldest go first
}

// validate checks the limits as ConfigureSendQueues receives them.
func (cfg *SendQueueConfig) validate() error {
	switch {
	case cfg.MaxBytesPerPeer <= 0:
		return fmt.Errorf("max bytes per peer must be positive, got %d", cfg.MaxBytesPerPeer)
	case cfg.MaxMsgsPerPeer <= 0:
		return fmt.Errorf("max messages per peer must be positive, got %d", cfg.MaxMsgsPerPeer)
	case cfg.MaxBytesTotal <= 0:
		return fmt.Errorf("max total bytes must be positive, got %d", cfg.MaxBytesTotal)
	case cfg.LowWater < 0 || cfg.LowWater >= 1:
		return fmt.Errorf("low water must be in [0, 1), got %v", cfg.LowWater)
	case cfg.MaxWaitAge <= 0:
		return fmt.Errorf("max wait age must be positive, got %v", cfg.MaxWaitAge)
	case cfg.MaxDialAttempts <= 0:
		return fmt.Errorf("max dial attempts must be positive, got %d", cfg.MaxDialAttempts)
	case len(cfg.RetryBackoff) == 0:
		return fmt.Errorf("retry backoff must have at least one value")
	case cfg.MaxEvents <= 0:
		return fmt.Errorf("max events must be positive, got %d", cfg.MaxEvents)
	}
	for _, b := range cfg.RetryBackoff {
		if b < 0 {
			return fmt.Errorf("retry backoff values must not be negative, got %v", b)
		}
	}
	return nil
}

// SendEvent is one event of a send queue, as PopSendEvents hands it to Python. Which fields
// are set depends on the type:
//   - written: Seq, the last frame written on the transport (and, in order, all before it).
//     Consecutive ones of a peer are merged into the last.
//   - write_failed: Seq of the frame that failed (kept for a retry), Attempt, Error.
//   - recovered: Seq of the first frame written after failures, Count of the failures.
//   - refused: Count and Bytes of the refused frames, Reason of the last one. Consecutive
//     ones of a peer are merged.
//   - queue_dropped: Reason, FromSeq..ToSeq the frames lost, Count, Bytes, the last Error.
//   - events_lost: Count of the oldest events dropped because the buffer was full.
//
// Time is a unix time in seconds, of the last event merged into this one.
type SendEvent struct {
	Type    string   `json:"type"`
	PeerID  string   `json:"peer_id,omitempty"`
	Seq     uint64   `json:"seq,omitempty"`
	FromSeq uint64   `json:"from_seq,omitempty"`
	ToSeq   uint64   `json:"to_seq,omitempty"`
	Count   int      `json:"count,omitempty"`
	Bytes   int64    `json:"bytes,omitempty"`
	Total   int64    `json:"total,omitempty"` // Of a progress event: the size of the whole message
	Kind    string   `json:"kind,omitempty"`  // Of a progress event: "chunked" or "file"
	Files   []string `json:"files,omitempty"` // Of a progress event of a file message: the names of its files
	Attempt int      `json:"attempt,omitempty"`
	Reason  string   `json:"reason,omitempty"`
	Error   string   `json:"error,omitempty"`
	Time    float64  `json:"time"`
}

// sendRefusal is the error of a frame that a queue refused.
type sendRefusal struct {
	reason string
}

func (r *sendRefusal) Error() string {
	return fmt.Sprintf("send queue full (%s)", r.reason)
}

// queuedFrame is one message waiting in the queue of a peer: a built frame, or a message
// that the writer cuts into chunk frames (see chunked.go) when it is larger than a chunk.
type queuedFrame struct {
	seq        uint64
	frame      []byte
	chunked    *chunkedPayload
	file       *filePayload // A file message (file.go)
	enqueuedAt time.Time
}

// chunkedPayload is a message that travels as chunk frames: its channel and its payload.
type chunkedPayload struct {
	channel string
	payload []byte
	attempt uint8 // Of its next start (chunked.go); only the writer of the queue touches it
}

// size is what the message counts in the byte caps of the queues.
func (f *queuedFrame) size() int64 {
	if f.file != nil {
		return int64(len(f.file.msg) + len(f.file.channel)) // Its files are on disk, not in memory
	}
	if f.chunked != nil {
		return int64(len(f.chunked.payload) + len(f.chunked.channel))
	}
	return int64(len(f.frame))
}

// peerQueue is the queue of one peer. All its fields are guarded by SendStore.mu.
type peerQueue struct {
	frames  *list.List // of *queuedFrame, oldest first; the head stays here while it is written
	bytes   int64
	nextSeq uint64 // the sequence number of the last accepted frame; never reset

	refusing bool // refused a frame and has not drained to the low-water mark since
	running  bool // a writer goroutine is in charge of this queue

	// gen changes when the queue is dropped: a writer that comes back from a write
	// with another gen than the one it started with drops its outcome.
	gen uint64
	// wake cuts short the backoff of the writer when the queue is dropped.
	wake chan struct{}

	failures int    // failed writes in a row
	lastErr  string // of the last failed write
	// headSince is when the head started to wait with no progress: when it was queued,
	// or when the frame before it was written if that came later. A write that fails
	// does not move it, so an outage is bounded by MaxWaitAge as a whole.
	headSince time.Time
}

// SendStore holds the send queues of one instance.
type SendStore struct {
	mu  sync.Mutex
	ctx context.Context
	cfg SendQueueConfig

	// write writes one frame to the peer: the transport. A test can replace it.
	write func(pid peer.ID, frame []byte) error
	// writeChunked writes one message as chunk frames, from its first chunk (chunked.go).
	writeChunked func(pid peer.ID, seq uint64, cp *chunkedPayload) error
	// writeFile writes one file message, from its first chunk (file.go).
	writeFile func(pid peer.ID, seq uint64, fp *filePayload) error
	// now is the clock of the queues: time.Now, unless a test says otherwise.
	now func() time.Time

	queues        map[peer.ID]*peerQueue
	totalBytes    int64
	refusingTotal bool

	events     []*SendEvent
	lastEvent  map[string]*SendEvent // the last event of each peer still in events
	eventsLost int
}

// newSendStore returns the send queues of an instance, whose writers stop with ctx.
func newSendStore(ctx context.Context, cfg SendQueueConfig, write func(pid peer.ID, frame []byte) error) *SendStore {
	return &SendStore{
		ctx:       ctx,
		cfg:       cfg,
		write:     write,
		now:       time.Now,
		queues:    make(map[peer.ID]*peerQueue),
		lastEvent: make(map[string]*SendEvent),
	}
}

// newInstanceSendStore returns the send queues of an instance, writing on its transports,
// or nil when ConfigureSendQueues was not called.
func newInstanceSendStore(ni *NodeInstance) *SendStore {
	if sendQueueSettings == nil {
		return nil
	}
	store := newSendStore(ni.ctx, *sendQueueSettings, func(pid peer.ID, frame []byte) error {
		return writeQueuedFrame(ni, pid, frame)
	})
	store.writeChunked = func(pid peer.ID, seq uint64, cp *chunkedPayload) error {
		progress := store.progressOf(pid, seq, transferChunked, nil)
		return writeLargeMessage(ni, pid, seq, transferChunked, int64(len(cp.payload)), func(write func([]byte) error) error {
			return writeChunks(seq, cp, write, progress)
		})
	}
	store.writeFile = func(pid peer.ID, seq uint64, fp *filePayload) error {
		progress := store.progressOf(pid, seq, transferFile, fp.names())
		return writeLargeMessage(ni, pid, seq, transferFile, fp.fileBytes(), func(write func([]byte) error) error {
			return writeFileMessage(seq, fp, write, progress)
		})
	}
	return store
}

// Enqueue queues a built frame for the peer and returns its sequence number, or a
// *sendRefusal when the frame does not fit: then nothing changes in the queue. It never
// waits on the network.
func (store *SendStore) Enqueue(pid peer.ID, frame []byte) (uint64, error) {
	return store.enqueue(pid, &queuedFrame{frame: frame})
}

// EnqueueChunked queues a message that the writer sends as chunk frames (it is larger
// than a chunk, and the peer takes them), as Enqueue does for a frame.
func (store *SendStore) EnqueueChunked(pid peer.ID, channel string, payload []byte) (uint64, error) {
	return store.enqueue(pid, &queuedFrame{chunked: &chunkedPayload{channel: channel, payload: payload}})
}

// enqueue is the body of Enqueue and EnqueueChunked: it numbers the message and queues
// it, or refuses it when it does not fit.
func (store *SendStore) enqueue(pid peer.ID, item *queuedFrame) (uint64, error) {
	store.mu.Lock()
	defer store.mu.Unlock()

	q := store.queues[pid]
	if q == nil {
		q = &peerQueue{frames: list.New(), wake: make(chan struct{}, 1)}
		store.queues[pid] = q
	}
	size := item.size()
	if reason := store.refusalLocked(q, size); reason != "" {
		store.recordLocked(&SendEvent{Type: SendEventRefused, PeerID: pid.String(), Count: 1, Bytes: size, Reason: reason})
		return 0, &sendRefusal{reason: reason}
	}

	now := store.now()
	if q.frames.Len() == 0 {
		q.headSince = now
	}
	q.nextSeq++
	item.seq, item.enqueuedAt = q.nextSeq, now
	q.frames.PushBack(item)
	q.bytes += size
	store.totalBytes += size
	if !q.running {
		q.running = true
		go store.runWriter(pid, q)
	}
	return q.nextSeq, nil
}

// Drop empties the queue of the peer at once, as when the peer is purged, and records it
// as dropped. Its writer stops after the write it may be in, which is bounded by its
// deadline. It returns how many frames were dropped.
func (store *SendStore) Drop(pid peer.ID) int {
	store.mu.Lock()
	defer store.mu.Unlock()
	q := store.queues[pid]
	if q == nil {
		return 0
	}
	return store.dropLocked(pid, q, SendDroppedExplicit)
}

// PopEvents returns the events recorded since the last call, oldest first, and forgets them.
func (store *SendStore) PopEvents() []*SendEvent {
	store.mu.Lock()
	defer store.mu.Unlock()
	events := store.events
	if store.eventsLost > 0 {
		lost := &SendEvent{Type: SendEventEventsLost, Count: store.eventsLost, Time: unixSeconds(store.now())}
		events = append([]*SendEvent{lost}, events...)
	}
	store.events = nil
	store.lastEvent = make(map[string]*SendEvent)
	store.eventsLost = 0
	return events
}

// refusalLocked says why a frame of the given size cannot be queued for q, or "" when it
// can. A refusal at a cap arms the low-water mark of the cap.
func (store *SendStore) refusalLocked(q *peerQueue, size int64) string {
	cfg := &store.cfg
	if size > cfg.MaxBytesPerPeer || size > cfg.MaxBytesTotal {
		return SendRefusedTooLarge
	}
	if q.refusing {
		if q.bytes > int64(cfg.LowWater*float64(cfg.MaxBytesPerPeer)) || float64(q.frames.Len()) > cfg.LowWater*float64(cfg.MaxMsgsPerPeer) {
			return SendRefusedLowWater
		}
		q.refusing = false
	}
	if store.refusingTotal {
		if store.totalBytes > int64(cfg.LowWater*float64(cfg.MaxBytesTotal)) {
			return SendRefusedLowWaterTotal
		}
		store.refusingTotal = false
	}
	switch {
	case q.frames.Len()+1 > cfg.MaxMsgsPerPeer:
		q.refusing = true
		return SendRefusedPeerMsgs
	case q.bytes+size > cfg.MaxBytesPerPeer:
		q.refusing = true
		return SendRefusedPeerBytes
	case store.totalBytes+size > cfg.MaxBytesTotal:
		store.refusingTotal = true
		return SendRefusedTotalBytes
	}
	return ""
}

// runWriter is the writer goroutine of a queue: it writes the head until the queue is
// empty, dropped, or the instance closes, and then ends. Enqueue starts a new one when a
// frame arrives at a queue that has none.
func (store *SendStore) runWriter(pid peer.ID, q *peerQueue) {
	defer recoverGoroutine("sendQueueWriter")
	defer func() {
		// Also after a panic: a queue left marked as running would never be written again.
		store.mu.Lock()
		q.running = false
		store.mu.Unlock()
	}()

	for {
		store.mu.Lock()
		if store.ctx.Err() != nil {
			store.mu.Unlock()
			return
		}
		element := q.frames.Front()
		if element == nil {
			store.mu.Unlock()
			return
		}
		if store.now().Sub(q.headSince) > store.cfg.MaxWaitAge {
			store.dropLocked(pid, q, SendDroppedMaxAge)
			store.mu.Unlock()
			return
		}
		head := element.Value.(*queuedFrame)
		gen := q.gen
		store.mu.Unlock()

		var err error
		if head.file != nil {
			err = store.writeFile(pid, head.seq, head.file)
		} else if head.chunked != nil {
			err = store.writeChunked(pid, head.seq, head.chunked)
		} else {
			err = store.write(pid, head.frame)
		}

		store.mu.Lock()
		if q.gen != gen {
			// The queue was dropped during the write: its outcome belongs to nobody.
			store.mu.Unlock()
			continue
		}
		if err == nil {
			q.frames.Remove(element)
			size := head.size()
			q.bytes -= size
			store.totalBytes -= size
			q.headSince = store.now()
			if q.failures > 0 {
				logger.Infof("[GO] ✅ Send queue to %s recovered after %d failed writes (seq %d).\n", pid, q.failures, head.seq)
				store.recordLocked(&SendEvent{Type: SendEventRecovered, PeerID: pid.String(), Seq: head.seq, Count: q.failures})
				q.failures, q.lastErr = 0, ""
			}
			store.recordLocked(&SendEvent{Type: SendEventWritten, PeerID: pid.String(), Seq: head.seq})
			store.mu.Unlock()
			continue
		}

		var permanent *permanentSendError
		if errors.As(err, &permanent) {
			// Written again it would fail the same way (a file changed): dropped alone.
			logger.Warnf("[GO] 🗑️ Send queue to %s: seq %d dropped (%s): %v\n", pid, head.seq, permanent.reason, err)
			q.frames.Remove(element)
			size := head.size()
			q.bytes -= size
			store.totalBytes -= size
			q.headSince = store.now()
			store.recordLocked(&SendEvent{Type: SendEventQueueDropped, PeerID: pid.String(), Reason: permanent.reason, FromSeq: head.seq, ToSeq: head.seq, Count: 1, Bytes: size, Error: err.Error()})
			store.mu.Unlock()
			continue
		}
		q.failures++
		q.lastErr = err.Error()
		logger.Warnf("[GO] ⚠️ Send queue to %s: write of seq %d failed (attempt %d of %d): %v\n", pid, head.seq, q.failures, store.cfg.MaxDialAttempts, err)
		store.recordLocked(&SendEvent{Type: SendEventWriteFailed, PeerID: pid.String(), Seq: head.seq, Attempt: q.failures, Error: q.lastErr})
		if q.failures >= store.cfg.MaxDialAttempts {
			store.dropLocked(pid, q, SendDroppedMaxAttempts)
			store.mu.Unlock()
			return
		}

		// Wait for the retry, but no longer than the head may still wait: the age check
		// at the top of the loop drops the queue at once when the backoff outlives it.
		backoff := store.cfg.RetryBackoff[len(store.cfg.RetryBackoff)-1]
		if q.failures-1 < len(store.cfg.RetryBackoff) {
			backoff = store.cfg.RetryBackoff[q.failures-1]
		}
		if left := q.headSince.Add(store.cfg.MaxWaitAge).Sub(store.now()); left < backoff {
			backoff = left + time.Millisecond
		}
		wake := q.wake
		store.mu.Unlock()

		if backoff > 0 {
			timer := time.NewTimer(backoff)
			select {
			case <-timer.C:
			case <-wake:
			case <-store.ctx.Done():
			}
			timer.Stop()
		}
	}
}

// dropLocked empties the queue, records the drop and returns how many frames it held.
// The failure count starts again from zero, and so does the low-water mark.
func (store *SendStore) dropLocked(pid peer.ID, q *peerQueue, reason string) int {
	count := q.frames.Len()
	if count > 0 {
		first := q.frames.Front().Value.(*queuedFrame)
		last := q.frames.Back().Value.(*queuedFrame)
		logger.Warnf("[GO] 🗑️ Send queue to %s dropped (%s): %d messages, seq %d..%d.\n", pid, reason, count, first.seq, last.seq)
		store.recordLocked(&SendEvent{
			Type:    SendEventQueueDropped,
			PeerID:  pid.String(),
			Reason:  reason,
			FromSeq: first.seq,
			ToSeq:   last.seq,
			Count:   count,
			Bytes:   q.bytes,
			Error:   q.lastErr,
		})
	}
	store.totalBytes -= q.bytes
	q.frames.Init()
	q.bytes = 0
	q.refusing = false
	q.failures, q.lastErr = 0, ""
	q.gen++
	select {
	case q.wake <- struct{}{}:
	default:
	}
	return count
}

// recordLocked adds an event, merged into the last one of its peer when both are written
// or both are refused events. When the buffer is full the oldest event goes.
func (store *SendStore) recordLocked(ev *SendEvent) {
	ev.Time = unixSeconds(store.now())
	if last, ok := store.lastEvent[ev.PeerID]; ok && last.Type == ev.Type {
		switch ev.Type {
		case SendEventWritten:
			last.Seq, last.Time = ev.Seq, ev.Time
			return
		case SendEventRefused:
			last.Count += ev.Count
			last.Bytes += ev.Bytes
			last.Reason, last.Time = ev.Reason, ev.Time
			return
		case SendEventProgress, SendEventReceiveProgress:
			if last.Seq == ev.Seq {
				// Only the latest progress of a message matters until Python pops it
				last.Bytes, last.Total, last.Attempt, last.Time = ev.Bytes, ev.Total, ev.Attempt, ev.Time
				if ev.Files != nil {
					last.Files = ev.Files
				}
				return
			}
		}
	}
	if len(store.events) >= store.cfg.MaxEvents {
		oldest := store.events[0]
		store.events = store.events[1:]
		store.eventsLost++
		if store.lastEvent[oldest.PeerID] == oldest {
			delete(store.lastEvent, oldest.PeerID)
		}
	}
	store.events = append(store.events, ev)
	store.lastEvent[ev.PeerID] = ev
}

// unixSeconds is a time as the events give it to Python.
func unixSeconds(t time.Time) float64 {
	return float64(t.UnixNano()) / 1e9
}

// progressOf returns what records the progress of the writing of one message: a progress
// event when the throttle says so, the throttle starting again with each attempt.
func (store *SendStore) progressOf(pid peer.ID, seq uint64, kind string, names []string) func(sent, total uint64, attempt uint8) {
	var throttle progressThrottle
	current := -1
	return func(sent, total uint64, attempt uint8) {
		if int(attempt) != current {
			throttle, current = progressThrottle{}, int(attempt)
		}
		if !throttle.due(store.now(), sent, total) {
			return
		}
		store.mu.Lock()
		defer store.mu.Unlock()
		store.recordLocked(&SendEvent{Type: SendEventProgress, PeerID: pid.String(), Seq: seq, Bytes: int64(sent),
			Total: int64(total), Attempt: int(attempt), Kind: kind, Files: names})
	}
}

// queueDirectMessage is the direct-message branch of SendMessageToPeer: it checks the
// message, builds its frame and queues it. It returns the sequence number of the frame,
// or the error code (SendCode*) and the text of the reply.
//
// data may be borrowed (the memory of the caller, valid only during the call): what is queued
// is always a copy of it, the frame or the payload of a chunked message.
func queueDirectMessage(ni *NodeInstance, pid peer.ID, channel string, data []byte) (uint64, string, string, error) {
	if pid == ni.host.ID() {
		return 0, SendCodeSelfSend, "Attempt to send direct message to self is invalid", nil
	}
	if ni.sendStore == nil {
		return 0, SendCodeNotConfigured, "Send queue not configured (ConfigureSendQueues was not called)", nil
	}
	// Larger than a chunk, to a peer that takes chunk frames: queued whole, cut by the writer
	if limits := ni.limitsOf(pid); usesChunks(limits, channel, len(data)) {
		if len(channel) > 255 {
			return 0, SendCodeTooLarge, fmt.Sprintf("Cannot send direct message to %s", pid), fmt.Errorf("channel name exceeds 255 bytes: %s", channel)
		}
		if limit := chunkedLimitFor(limits); limit > 0 && int64(len(data)) > limit {
			return 0, SendCodeTooLarge, fmt.Sprintf("Cannot send direct message to %s", pid), fmt.Errorf("message of %d bytes above the limit (%d) of the peer or of this node", len(data), limit)
		}
		seq, err := ni.sendStore.EnqueueChunked(pid, channel, append([]byte(nil), data...))
		if err != nil {
			return 0, SendCodeQueueFull, fmt.Sprintf("Direct message to %s refused", pid), err
		}
		logger.Debugf("[GO] 📥 Instance %d: Queued chunked message to %s (seq %d, %d bytes).\n", ni.instanceIndex, pid, seq, len(data))
		return seq, "", fmt.Sprintf("Direct message to %s queued.", pid), nil
	}

	frame, err := buildDirectMessageFrame(channel, data)
	if err != nil {
		return 0, SendCodeTooLarge, fmt.Sprintf("Cannot send direct message to %s", pid), err
	}
	seq, err := ni.sendStore.Enqueue(pid, frame)
	if err != nil {
		return 0, SendCodeQueueFull, fmt.Sprintf("Direct message to %s refused", pid), err
	}
	logger.Debugf("[GO] 📥 Instance %d: Queued direct message to %s (seq %d, %d bytes).\n", ni.instanceIndex, pid, seq, len(frame))
	return seq, "", fmt.Sprintf("Direct message to %s queued.", pid), nil
}

// sendDirectMessageNow is the direct send of a library whose send queues were never
// configured (an older caller): it writes at once, the WebRTC DataChannel first when one is
// open, and replies when the write is done, as before the queues.
func sendDirectMessageNow(ni *NodeInstance, pid peer.ID, channel string, data []byte) string {
	if pid == ni.host.ID() {
		return jsonErrorResponseWithCode("Attempt to send direct message to self is invalid", SendCodeSelfSend, nil)
	}
	ni.webrtcMutex.RLock()
	wconn, hasWebRTC := ni.webrtcConnections[pid]
	ni.webrtcMutex.RUnlock()
	if hasWebRTC && wconn.dc != nil && wconn.dc.ReadyState() == pwebrtc.DataChannelStateOpen {
		err := writeToWebRTCDataChannel(wconn, channel, data)
		if err == nil {
			logger.Infof("[GO] 📤 Instance %d: Sent to %s via WebRTC DataChannel\n", ni.instanceIndex, pid)
			return jsonSuccessResponse(fmt.Sprintf("Direct message sent to %s (WebRTC).", pid))
		}
		logger.Warnf("[GO] ⚠️ Instance %d: WebRTC send to %s failed (%v), falling back to stream\n", ni.instanceIndex, pid, err)
	}
	message, err := sendOnChatStream(ni, pid, channel, data)
	if err != nil {
		return jsonErrorResponse(message, err)
	}
	return jsonSuccessResponse(message)
}

// openDataChannel is the WebRTC DataChannel to the peer when one is open, else nil.
func openDataChannel(ni *NodeInstance, pid peer.ID) *WebRTCConn {
	ni.webrtcMutex.RLock()
	wconn, ok := ni.webrtcConnections[pid]
	ni.webrtcMutex.RUnlock()
	if ok && wconn.dc != nil && wconn.dc.ReadyState() == pwebrtc.DataChannelStateOpen {
		return wconn
	}
	return nil
}

// writeQueuedFrame is the transport of the writer of a send queue: the WebRTC DataChannel
// when one is open, the chat stream otherwise or when the DataChannel send fails.
func writeQueuedFrame(ni *NodeInstance, pid peer.ID, frame []byte) error {
	if wconn := openDataChannel(ni, pid); wconn != nil {
		err := writeFrameToWebRTCDataChannel(wconn, frame)
		if err == nil {
			logger.Infof("[GO] 📤 Instance %d: Sent to %s via WebRTC DataChannel\n", ni.instanceIndex, pid)
			return nil
		}
		logger.Warnf("[GO] ⚠️ Instance %d: WebRTC send to %s failed (%v), falling back to stream\n", ni.instanceIndex, pid, err)
	}
	message, err := sendFrameOnChatStream(ni, pid, frame)
	if err != nil {
		return fmt.Errorf("%s %w", message, err)
	}
	return nil
}
