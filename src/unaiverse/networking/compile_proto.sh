export PATH="$PATH:$(go env GOPATH)/bin"

# message.proto: chat/transport Msg — compiled to both Python and Go (messaging.go
# unpacks the rendezvous discovery Msg via pg.Message).
( cd p2p && protoc --proto_path=. --go_out=./unailib/proto-go --go_opt=paths=source_relative ./message.proto )
( cd p2p && protoc --proto_path=. --python_out=. ./message.proto )

# session.proto: session-FSM handshake messages — Python only. Go never parses these
# (it is a pure pipe), so no --go_out here; the session codec lives in session/sess_msg.py.
( cd session && protoc --proto_path=. --python_out=. ./session.proto )
