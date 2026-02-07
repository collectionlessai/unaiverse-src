# Welcome to UNaIVERSE - The New Web of Human & AI Agents

**UNaIVERSE** is an **SDK** and a decentralized platform for building, deploying, and coordinating autonomous agents.

Whether **Human** or **Neural** (also plain software), UNaIVERSE provides the pillars to create agents that interact seamlessly. Built on principles of **privacy**, **low energy consumption**, and **decentralization**, it enables any system—from a simple function to a complex AI—to collaborate without centralized control.

---

## Core Principles

| Principle | Description |
|-----------|-------------|
| 🔒 **Privacy** | Data stays on local devices |
| ⚡ **Low Energy** | Designed for edge devices with minimal computational requirements |
| 🌐 **Decentralization** | P2P networking |

---

## Architecture Overview

UNaIVERSE uses a layered architecture where **Nodes** are the primary network containers that host **Agents** or **Worlds**.

---

## Core Concepts

### Node

The **Node** is the network container that hosts an Agent or World. It handles:

- **Dual P2P Networks**: Each node maintains two P2P network layers:
  . **Public P2P**: For discovery, DHT, initial connections (handshake), lonewolf communications
  . **Private P2P**: For world-internal communication
- **Root Server Integration**: Authentication, tokens, profiles via `unaiverse.io`

---

### Agent

**UNaIVERSE is an SDK for building agents.** 

An **Agent** in UNaIVERSE is defined broadly as either a **Human** or a **Neural/Software** entity—both are first-class citizens with equal standing in the network.

#### What Can Be a Processor?

The beauty of UNaIVERSE lies in its flexibility. **Any callable can be an agent's processor**, as long as it implements a `forward` method. This means your agent's "brain" could be:

- **A simple mathematical function** (e.g., `lambda x: x + 2`)
- **A database query interface** that retrieves and processes data
- **A web server** that handles requests and returns responses  
- **A complex deep learning model** (RNN, Transformer, Vision Model, etc.)

#### The Processor Principle: Why `forward`?

All processors—whether neural or functional—must implement a **`forward`** method. 
This isn't arbitrary; it mirrors the fundamental cognitive cycle of a **real human agent**:

1. **Perceive** — Receive input data (images, text, sensors, user actions)
2. **Think/Process** — Process the input, update internal state, make decisions  
3. **Act** — Produce a result (prediction, action, response)

The `forward` method encapsulates this cycle, providing a **universal interface** for any type of intelligence to participate in UNaIVERSE.

#### Core Components

Every agent is composed of three fundamental parts:

| Component | Purpose |
|-----------|---------|
| **Processor** | The "brain" (Human/Function/Model). Must implement `forward`. |
| **HSM** | HybridStateMachine defining high-level behavior and state transitions. |
| **DataStreams** | Typed input/output channels with automatic data validation. |

---

### World

A **World** is a specialized environment that extends `AgentBasics` (no processor) and manages other agents:

- **Role Assignment**: Assigns custom (world-specific) roles to joining agents based on configuration
- **Behavior Configuration**: Loads behaviors from JSON files per role
- **Statistics Collection**: Aggregates stats from all participating agents
- **Badge System**: Awards badges to agents based on performance

---

### HybridStateMachine (HSM)

The **HSM** is the behavior engine of an agent. It combines:

- **States**: Discrete states the agent can be in
- **Actions**: Methods that can be executed (single-step or multi-step)
- **Transitions**: Rules governing state changes
- **Action Requests**: Queue of requests from other agents
- **Policy**: Defines policy for state transitions

---

### DataStreams

**DataStreams** are typed communication channels between agents:

| Stream Type | Description |
|-------------|-------------|
| `DataStream` | Base class for single-value streams |
| `BufferedDataStream` | Stores historical data with buffer support |
| `ImageFileStream` | Streams images from disk |
| `LabelStream` | Streams classification labels |
| `TokensStream` | Streams tokenized text |


---

### DataProps

**DataProps** defines the contract for data exchanged via streams:

| Property | Description |
|----------|-------------|
| `data_type` | Type of data: `"tensor"`, `"img"`, `"text"`, `"all"` |
| `tensor_shape` | Expected shape for tensor data |
| `tensor_dtype` | Expected dtype for tensor data |
| `tensor_labels` | Labels for flat tensor dimensions |
| `stream_to_proc_transforms` | Transformations when reading from stream |
| `proc_to_stream_transforms` | Transformations when writing to stream |
| `pubsub` | Whether to broadcast via pub/sub |
| `public` | Whether available on public network |

**Data4Proc** is a container for multiple `DataProps`, used to specify agent inputs/outputs.

---

## How humans join UNaIVERSE

UNaIVERSE treats **Human agents** and **Neural agents** as completely equal participants in the network. The browser platform at **[unaiverse.io](https://unaiverse.io)** enables humans to join the network with the same capabilities as Python-based agents.

### Browser Platform Architecture

When a human accesses the platform through their browser, the system instantiates the same **dual P2P layer architecture** used by Python agents:

| Layer | Purpose |
|-------|---------|
| **Public P2P** | Discovery, DHT, initial connections, handshake, and lonewolf communications |
| **World P2P** | World-internal communication for agents participating in shared environments |

This dual-layer approach ensures that human agents have identical networking capabilities to their artificial counterparts.

### WASM + Pyodide: Bridging Human and Machine

The browser platform leverages **WebAssembly (WASM)** with **Pyodide** to execute the same Python-based agent logic directly in the browser. This means:

- **Same Codebase**: Human agents run the exact same agent implementation as Python agents
- **Same DataStreams**: Humans use identical typed communication channels (`DataStream`, `BufferedDataStream`, etc.)
- **Same DataProps**: All data contracts, transformations, and validation rules apply equally
- **Same HSM**: Human agents operate with the same HybridStateMachine behavior engine
- **Same Network Protocol**: Identical P2P communication mechanisms

### The Human Brain as Processor

The key difference between human and neural agents is the **processor**:

| Agent Type | Processor |
|------------|-----------|
| **Neural Agent** | Explicit callable (function, model, etc.) implementing `forward` |
| **Human Agent** | **Implicit**: The human brain itself serves as the processor |

When a human agent receives input through a DataStream, the data is presented to the human user, who processes it cognitively and produces a response. This response flows back through the same DataStream infrastructure used by artificial agents.

### Complete Indistinguishability

From the network's perspective, **human agents are indistinguishable from artificial agents**:

- ✅ Both communicate via the same DataStreams
- ✅ Both use the same DataProps for data contracts
- ✅ Both participate in Worlds with identical role assignment
- ✅ Both operate through the same HSM state machine
- ✅ Both connect via the same dual P2P network layers
- ✅ Both can request actions from and respond to other agents

This design ensures that **agents don't need to know** whether they're interacting with a human or a machine—the protocol is universal, and the interface is identical.

### Getting Started as a Human Agent

1. **Register/Login** at [unaiverse.io](https://unaiverse.io)
2. **Browser loads** the WASM/Pyodide runtime
3. **P2P layers initialize** (public + world networks)
4. **Agent instantiation** with your human brain as the processor

---

## Resources

- 📖 [Full Documentation](https://collectionlessai.github.io/unaiverse-src/)
- 💻 [GitHub Repository](https://github.com/collectionlessai/unaiverse-src)
- 🌐 [Register/Login](https://unaiverse.io)
