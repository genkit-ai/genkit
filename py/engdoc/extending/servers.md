# Servers

Starting a Genkit application or developer CLI spawns several server daemons as
either independent processes, threads, goroutines, or coroutines, depending upon
the runtime. The initialization process deals with:

* **Environment-based startup**: In development mode (`GENKIT_ENV=dev`), the
  Reflection server starts automatically.
* **Port selection**: All servers attempt to find available ports, with
  configurable defaults.
* **Registration**: Each server registers itself for cleanup on application
  exit.
* **Runtime files**: Reflection servers write metadata files to enable tool
  discovery (`<working-directory>/.genkit/runtimes/<timestamp>.json`).
* **Traces**: Traces metadata (`<working-directory>/.genkit/traces`).

## Types

| Server Type               | Purpose                                                               | Implementation                            | Notes                                                   |
|---------------------------|-----------------------------------------------------------------------|-------------------------------------------|---------------------------------------------------------|
| Reflection                | Development-time API for inspecting and interacting with Genkit       | Both Go and JavaScript                    | Only starts in development mode (`GENKIT_ENV=dev`)      |
| Flow                      | Exposes registered flows as HTTP endpoints                            | Go (HTTP server) and JavaScript (Express) | Main server for production environment                  |
| Dev UI                    | Web interface for monitoring and interacting with Genkit applications | JavaScript only                           | Provides dashboard, monitoring, and debugging tools     |
| Telemetry                 | Collects and stores traces of Genkit operations                       | JavaScript only                           | Can use local file system or Firestore as backing store |
| Engineering documentation | `mkdocs` instances showing this information                           |                                           | Engineering documentation                               |

## Networking

| Server                    | Host        | Port                                     | Deployment Environment |
|---------------------------|-------------|------------------------------------------|------------------------|
| Flows                     | `localhost` | 3400 (override `PORT`)                   | `'dev'`, `'prod'`      |
| Dev UI/Tools API          | `localhost` | 4000-4099                                | `'dev'`, `'prod'`      |
| Reflection API            | `localhost` | 3100 (override `GENKIT_REFLECTION_PORT`) | `'dev'`                |
| Telemetry                 | `localhost` | 4033 (specified programmatically)        | `'dev'`, `'prod'`      |
| Engineering documentation | `localhost` | 8000                                     | `'dev'`                |

## Implementations

| Server           | Sources                                                                                                                                                                  |
|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Flows            | [JS](https://github.com/firebase/genkit/blob/main/js/plugins/express/src/index.ts), [Go](TODO), [Python](https://github.com/firebase/genkit/blob/main/py/packages/genkit/src/genkit/core/flows.py) |
| Telemetry        | [JS](https://github.com/firebase/genkit/blob/main/genkit-tools/telemetry-server/src/index.ts)                                                                            |
| Dev UI/Tools API | [JS](https://github.com/firebase/genkit/blob/main/genkit-tools/common/src/server/server.ts)                                                                              |
| Reflection       | [JS](https://github.com/firebase/genkit/blob/main/js/core/src/reflection.ts), [Go](https://github.com/firebase/genkit/blob/main/go/genkit/reflection.go), [Python](https://github.com/firebase/genkit/blob/main/py/packages/genkit/src/genkit/core/reflection.py) |

## Environment Variables

| Environment Variable           | Server/Component                | Default Value | Description                                                                                                                                                           |
|--------------------------------|---------------------------------|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `DEBUG`                        | Logger                          | `false`       | Sets the logging level to `'debug'` instead of `'info'`.                                                                                                              |
| `FIREBASE_CONFIG`              | Firebase plugins                | -             | JSON configuration for Firebase services.                                                                                                                             |
| `FIREBASE_DEBUG_FEATURES`      | Firebase plugins                | -             | List of Firebase features to enable debugging for.                                                                                                                    |
| `FIREBASE_DEBUG_MODE`          | Firebase plugins                | `false`       | Enables debug mode for Firebase plugins.                                                                                                                              |
| `FIREBASE_PROJECT_ID`          | Firebase plugins                | -             | Project ID for Firebase services.                                                                                                                                     |
| `FIRESTORE_COLLECTION`         | Firebase Firestore              | -             | Name of Firestore collection to use.                                                                                                                                  |
| `GCLOUD_LOCATION`              | Vertex AI                       | -             | Location for Vertex AI services.                                                                                                                                      |
| `GCLOUD_PROJECT`               | Google Cloud plugins            | -             | Project ID for Google Cloud services.                                                                                                                                 |
| `GCLOUD_SERVICE_ACCOUNT_CREDS` | Google Cloud plugins            | -             | JSON credentials for authenticating with Google Cloud.                                                                                                                |
| `GENKIT_ENV`                   | All servers                     | `'prod'`      | Controls the environment mode. Values: `'dev'` (development) or `'prod'` (production). In `'dev'` mode, additional servers are started such as the Reflection server. |
| `GENKIT_GA_DEBUG`              | Analytics                       | `false`       | Enables debug mode for Google Analytics in the dev tools.                                                                                                             |
| `GENKIT_GA_VALIDATE`           | Analytics                       | `false`       | Enables validation mode for Google Analytics in the dev tools.                                                                                                        |
| `GENKIT_REFLECTION_PORT`       | Reflection Server               | 3100          | The port on which the Reflection API server listens in development mode.                                                                                              |
| `GENKIT_RUNTIME_ID`            | Reflection Server               | Process ID    | Custom identifier for the runtime (see `.genkit/runtimes/<timestamp>.json`).                                                                                          |
| `GENKIT_TELEMETRY_SERVER`      | Telemetry Client (Go/JS/Python) | -             | URL of the telemetry server to send trace data to.                                                                                                                    |
| `GOOGLE_API_KEY`               | Google APIs                     | -             | General API key for Google services, used as fallback.                                                                                                                |
| `GOOGLE_CLOUD_PROJECT`         | Google Cloud plugins            | -             | Alternative name for Project ID for Google Cloud services.                                                                                                            |
| `GOOGLE_GENAI_API_KEY`         | Google Generative AI            | -             | API key for Google's generative AI services.                                                                                                                          |
| `PINECONE_API_KEY`             | Pinecone plugin                 | -             | API key for authenticating with Pinecone vector database.                                                                                                             |
| `PORT`                         | Flow Server                     | 3400          | The port on which the HTTP Flow server listens.                                                                                                                       |
| `WEAVIATE_API_KEY`             | Weaviate plugin                 | -             | API key for authenticating with Weaviate.                                                                                                                             |
| `WEAVIATE_URL`                 | Weaviate plugin                 | -             | URL for the Weaviate vector database.                                                                                                                                 |

## Signal Handling

Many of these servers handle signals to handle graceful termination and clean up.

| Signal    | Handling                          | Handlers             |
|-----------|-----------------------------------|----------------------|
| `SIGTERM` | Graceful termination              | All servers          |
| `SIGINT`  | Graceful termination and clean up | main Genkit instance |

!!! note annotate "Common Signals"

    | Signal               | Number   | Description                                                                                             | Default Action        | Notes                                                           |
    |----------------------|----------|---------------------------------------------------------------------------------------------------------|-----------------------|-----------------------------------------------------------------|
    | `SIGHUP`             | 1        | Hangup signal. Sent when the controlling terminal closes or a process is terminated.                    | Terminate             | Often used to tell daemons to reload their configuration files. |
    | `SIGINT`             | 2        | Interrupt signal. Sent when the user presses `Ctrl+C`.                                                  | Terminate             | Typically used to interrupt a running program.                  |
    | `SIGQUIT`            | 3        | Quit signal. Sent when the user presses `Ctrl+\\`.                                                      | Terminate (core dump) | Similar to SIGINT, but also generates a core dump.              |
    | `SIGILL`             | 4        | Illegal instruction. Sent when a process attempts to execute an invalid instruction.                    | Terminate (core dump) | Indicates a programming error.                                  |
    | `SIGTRAP`            | 5        | Trace/breakpoint trap. Sent when a breakpoint is hit during debugging.                                  | Terminate (core dump) | Used by debuggers.                                              |
    | `SIGABRT` (`SIGIOT`) | 6        | Abort signal. Sent when a process calls the `abort()` function.                                         | Terminate (core dump) | Indicates an abnormal termination.                              |
    | `SIGBUS`             | 7 or 10  | Bus error. Sent when a process attempts to access memory that is not properly aligned.                  | Terminate (core dump) | Indicates a hardware or memory error.                           |
    | `SIGFPE`             | 8        | Floating-point exception. Sent when a process performs an invalid arithmetic operation.                 | Terminate (core dump) | Indicates an arithmetic error.                                  |
    | `SIGKILL`            | 9        | Kill signal. Forces a process to terminate immediately.                                                 | Terminate             | Cannot be caught or ignored.                                    |
    | `SIGUSR1`            | 10 or 30 | User-defined signal 1.                                                                                  | Terminate             | Can be used for custom signal handling.                         |
    | `SIGSEGV`            | 11       | Segmentation violation. Sent when a process attempts to access memory that it is not allowed to access. | Terminate (core dump) | Indicates a memory access error.                                |
    | `SIGUSR2`            | 12 or 31 | User-defined signal 2.                                                                                  | Terminate             | Can be used for custom signal handling.                         |
    | `SIGPIPE`            | 13       | Pipe broken. Sent when a process attempts to write to a pipe that has no readers.                       | Terminate             | Indicates a communication error.                                |
    | `SIGALRM`            | 14       | Alarm clock. Sent when a timer expires.                                                                 | Terminate             | Used for timeouts.                                              |
    | `SIGTERM`            | 15       | Termination signal. Sent by the `kill` command by default.                                              | Terminate             | Allows a process to perform cleanup before exiting.             |
    | `SIGCHLD`            | 17 or 20 | Child process status changed. Sent to a parent process when a child process terminates or stops.        | Ignore                | Used for process management.                                    |
    | `SIGCONT`            | 18 or 19 | Continue signal. Sent to a stopped process to resume execution.                                         | Continue              | Used for job control.                                           |
    | `SIGSTOP`            | 19 or 17 | Stop signal. Forces a process to stop execution.                                                        | Stop                  | Cannot be caught or ignored.                                    |
    | `SIGTSTP`            | 20 or 18 | Terminal stop signal. Sent when the user presses `Ctrl+Z`.                                              | Stop                  | Used for job control.                                           |
    | `SIGTTIN`            | 21       | Terminal input. Sent to a background process that attempts to read from the terminal.                   | Stop                  | Used for job control.                                           |
    | `SIGTTOU`            | 22       | Terminal output. Sent to a background process that attempts to write to the terminal.                   | Stop                  | Used for job control.                                           |
