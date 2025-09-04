export PATH="$PATH:$(go env GOPATH)/bin"
protoc --proto_path=. --go_out=./proto-go --go_opt=paths=source_relative ./message.proto
protoc --proto_path=. --python_out=. ./message.proto
