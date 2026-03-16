// Here we define helpers and small wrappers for the P2P library.
package main

import (
	"fmt"
	"bytes"
	"encoding/json"
	"container/list"
)

// jsonErrorResponse creates a JSON string representing an error state.
// It takes a base message and an optional error, formats them, escapes the message
// for JSON embedding, and returns an error string.
// The caller (usually C/Python) is responsible for freeing this C string using FreeString.
func jsonErrorResponse(
	message string,
	err error,
) string {

	errMsg := message
	if err != nil {
		errMsg = fmt.Sprintf("%s: %s", message, err.Error())
	}
	logger.Errorf("[GO] ❌ Error: %s", errMsg)
	escapedErrMsg := escapeStringForJSON(errMsg)
	jsonError := fmt.Sprintf(`{"state":"Error","message":"%s"}`, escapedErrMsg)
	return jsonError
}

// jsonSuccessResponse creates a JSON string representing a success state.
// It takes an arbitrary Go object (`message`), marshals it into JSON, wraps it
// in a standard {"state": "Success", "message": {...}} structure, and returns
// an error string.
// The caller (usually C/Python) is responsible for freeing this C string using FreeString.
func jsonSuccessResponse(
	message interface{},
) string {

	jsonData, err := json.Marshal(message)
	if err != nil {
		return jsonErrorResponse("Failed to marshal success response", err)
	}
	jsonSuccess := fmt.Sprintf(`{"state":"Success","message":%s}`, string(jsonData))
	return jsonSuccess
}

// escapeStringForJSON performs basic escaping of characters (like double quotes and backslashes)
// within a string to ensure it's safe to embed within a JSON string value.
// It uses Go's standard JSON encoder for robust escaping.
func escapeStringForJSON(
	s string,
) string {

	var buf bytes.Buffer
	// Encode the string using Go's JSON encoder, which handles escaping.
	json.NewEncoder(&buf).Encode(s)
	// The encoder adds surrounding quotes and a trailing newline, which we remove.
	res := buf.String()
	// Check bounds before slicing to avoid panic.
	if len(res) > 2 && res[0] == '"' && res[len(res)-2] == '"' {
		return res[1 : len(res)-2] // Trim surrounding quotes and newline
	}
	// Fallback if encoding behaves unexpectedly (e.g., empty string).
	return s
}

// getInstance is a new helper to safely retrieve a node instance.
// It handles bounds checking, nil checks, and locking.
func getInstance(instanceIndex int) (*NodeInstance, error) {
	if instanceIndex < 0 || instanceIndex >= maxInstances {
		return nil, fmt.Errorf("invalid instance index: %d. Must be between 0 and %d", instanceIndex, maxInstances-1)
	}

	// Use a Read Lock, which is fast and allows concurrent reads.
	globalInstanceMutex.RLock()
	instance := allInstances[instanceIndex]
	globalInstanceMutex.RUnlock()

	if instance == nil {
		return nil, fmt.Errorf("instance %d is not initialized or has been closed", instanceIndex)
	}

	// Check host as a proxy for *full* initialization,
	// as it's set late in CreateNode
	if instance.host == nil {
		return nil, fmt.Errorf("instance %d is not fully initialized (host is nil)", instanceIndex)
	}

	return instance, nil
}

// newMessageStore initializes a new MessageStore.
func newMessageStore() *MessageStore {
	return &MessageStore{
		messagesByChannel: make(map[string]*list.List),
	}
}
