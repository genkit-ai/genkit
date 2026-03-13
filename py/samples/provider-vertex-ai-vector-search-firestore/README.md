# Vertex AI - Vector Search Firestore

An example demonstrating the use Vector Search API with Firestore retriever for Vertex AI

### Monitoring and Running

For an enhanced development experience, use the provided `run.sh` script to start the sample with automatic reloading:

```bash
./run.sh
```

This script uses `watchmedo` to monitor changes in:
- `src/` (Python logic)
- `../../packages` (Genkit core)
- `../../plugins` (Genkit plugins)
- File patterns: `*.py`, `*.prompt`, `*.json`

Changes will automatically trigger a restart of the sample. You can also pass command-line arguments directly to the script, e.g., `./run.sh --some-flag`.

## Setup environment

1. Install [GCP CLI](https://cloud.google.com/sdk/docs/install).
2. Run the following code to connect to VertexAI.
```bash
gcloud auth application-default login
```
3. Set the following env vars to run the sample
```
export LOCATION=''
export PROJECT_ID=''
export FIRESTORE_COLLECTION=''
export VECTOR_SEARCH_DEPLOYED_INDEX_ID=''
export VECTOR_SEARCH_INDEX_ENDPOINT_PATH=''
export VECTOR_SEARCH_API_ENDPOINT=''
```
4. Run the sample.

## Env vars definition
| Variable                            | Definition                                                                                                |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `LOCATION`                          | The Google Cloud region or multi-region where your resources (e.g., Vertex AI Index, Firestore database) are located. Example: `us-central1`. |
| `PROJECT_ID`                        | The name or unique identifier for your Google Cloud Project.                                                      |
| `FIRESTORE_COLLECTION`              | The name of the Firestore collection used, for example, to store metadata associated with your vectors or the source documents. |
| `VECTOR_SEARCH_DEPLOYED_INDEX_ID`   | The ID of your deployed Vertex AI Vector Search index that you want to query.                               |
| `VECTOR_SEARCH_INDEX_ENDPOINT_PATH` | The full storage path of the Vertex AI Vector Search Index Endpoint. Example: `projects/YOUR_PROJECT_ID/locations/YOUR_LOCATION/indexEndpoints/YOUR_INDEX_ENDPOINT_ID`. |
| `VECTOR_SEARCH_API_ENDPOINT`        | The regional API endpoint for making calls to the Vertex AI Vector Search service. Example: `YOUR_LOCATION-aiplatform.googleapis.com`. |

## Run the sample

```bash
genkit start -- uv run src/main.py
```

## Testing This Demo

1. **Prerequisites** - Set up GCP resources:
   ```bash
   # Required environment variables
   export LOCATION=us-central1
   export PROJECT_ID=your_project_id
   export FIRESTORE_COLLECTION=your_collection_name
   export VECTOR_SEARCH_DEPLOYED_INDEX_ID=your_deployed_index_id
   export VECTOR_SEARCH_INDEX_ENDPOINT_PATH=your_endpoint_path
   export VECTOR_SEARCH_API_ENDPOINT=your_api_endpoint

   # Authenticate with GCP
   gcloud auth application-default login
   ```

2. **GCP Setup Required**:
   - Create Vertex AI Vector Search index
   - Deploy index to an endpoint
   - Create Firestore collection with documents
   - Ensure documents have matching IDs in both services

3. **Run the demo**:
   ```bash
   cd py/samples/provider-vertex-ai-vector-search-firestore
   ./run.sh
   ```

4. **Open DevUI** at http://localhost:4000

5. **Test the flows**:
   - [ ] `retrieve_documents` - Vector similarity search
   - [ ] Check results are ranked by distance
   - [ ] Verify Firestore document metadata is returned

6. **Expected behavior**:
   - Query is embedded and sent to Vector Search
   - Similar vectors are found and IDs returned
   - Firestore is queried for full document content
   - Results sorted by similarity distance
