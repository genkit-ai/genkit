# Firebase Telemetry Demo for Genkit Go

This sample demonstrates Firebase telemetry integration with Genkit Go.

## Structure

- `GUIDE.md` - Complete setup and verification guide
- `simple/` - Basic joke flow with Firebase telemetry
- `kitchen_sink/` - Advanced demo with tool calls and various AI operations

## Quick Start

1. Set environment variables:
```bash
export FIREBASE_PROJECT_ID="your-project-id"
export GEMINI_API_KEY="your-api-key"
```

2. Run simple demo:
```bash
cd simple
go run simple_joke.go
```

```bash
# Simple joke flow
curl -X POST http://localhost:3400/jokeFlow -H 'Content-Type: application/json' -d '{"data": "cats"}'
```

3. Run kitchen sink demo:
```bash
cd kitchen_sink  
go run kitchen_sink.go
```

```bash
# Image analysis flow  
curl -X POST http://localhost:3400/imageFlow -H 'Content-Type: application/json' -d '{"data": {"imageUrl": "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png", "prompt": "What logo is this?"}}'

# Text + Tools flow
curl -X POST http://localhost:3400/textFlow -H 'Content-Type: application/json' -d '{"data": "machine learning"}'

# Batch processing flow
curl -X POST http://localhost:3400/batchFlow -H 'Content-Type: application/json' -d '{"data": ["AI", "robotics", "quantum"]}'
```

5. Check telemetry in Google Cloud Console:
   - [Cloud Trace](https://console.cloud.google.com/traces)
   - [Cloud Monitoring](https://console.cloud.google.com/monitoring)
   - [Cloud Logging](https://console.cloud.google.com/logs)

## What You'll See

- All 5 telemetry modules working (Generate, Feature, Action, Engagement, Path)
- Real-time traces and metrics in Google Cloud
- Multimodal AI image analysis working perfectly
- Tool-based search and batch processing flows
- Complete observability with one line: `firebase.FirebaseTelemetry()`

## Verified Working Features

✅ **Simple Joke Flow** - Basic AI text generation  
✅ **Image Analysis** - Multimodal AI that can analyze any image URL  
✅ **Tool Integration** - AI using custom tools for web search  
✅ **Batch Processing** - Multiple AI operations in sequence  
✅ **Firebase Telemetry** - All telemetry data flowing to Google Cloud
