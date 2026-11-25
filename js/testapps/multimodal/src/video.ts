/**
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { vertexAI } from '@genkit-ai/google-genai';
import fileTypeChecker from 'file-type-checker';
import fs from 'fs';
import { Document, z } from 'genkit';
import path from 'path';
import { vectraStore } from './vector-store.js';

import { ai } from './genkit.js';
import { augmentedVideoPrompt } from './prompt.js';

// The multimodal embedder returns one embedding per video segment; the store
// keeps each segment as its own document, with the segment offsets in
// `metadata.embedMetadata`.
export const videoStore = vectraStore({
  ai,
  indexName: 'localMultiModalIndex',
  embedder: vertexAI.embedder('multimodalembedding@001'),
});

const VideoSegmentSchema = z.object({
  startOffsetSec: z.number(),
  endOffsetSec: z.number(),
});

// Define a local video indexer flow
export const localIndexVideo = ai.defineFlow(
  {
    name: 'localIndexVideo',
    inputSchema: z
      .string()
      .describe('A Video URL')
      .default('gs://cloud-samples-data/generative-ai/video/pixel8.mp4'),
  },
  async (videoUrl: string) => {
    const documents = await ai.run('extract-video', () =>
      extractVideo(videoUrl)
    );

    await videoStore.index(documents);
  }
);

// Suffix based type
function getVideoType(url: string) {
  const lastDotIndex = url.lastIndexOf('.');
  if (lastDotIndex === -1) {
    throw new Error('Error: Unable to determine video mime type');
  }
  const suffix = url.substring(lastDotIndex + 1);
  return `video/${suffix}`;
}

async function extractVideo(filePath: string): Promise<Document[]> {
  const videoDocs: Document[] = [];

  if (filePath.startsWith('http')) {
    throw new Error(
      'Vertex AI does not support http(s) video urls. Please use Google Cloud Storage (gs://) urls'
    );
  } else if (filePath.startsWith('gs://')) {
    // The default configuration is to look at the first 120 seconds of the
    // video and produce embeddings in 16 second increments.
    // This is not really necessary, since we are very close to the defaults
    // (i.e. 15 v.s. 16 seconds) it is just here to show what it looks like.
    // See also:
    // https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-multimodal-embeddings#video-modes
    // for pricing differences for the different intervals.
    const metadataFirst120Seconds = {
      videoSegmentConfig: {
        startOffsetSec: 0,
        endOffsetSec: 120,
        intervalSec: 15,
      },
    };

    // If your video is longer than 120 seconds, you can add additional video
    // document requests with different start/stop values e.g.
    // const metadataNext120Seconds = {
    //   "videoSegmentConfig": {
    //     "startOffsetSec": 120,
    //     "endOffsetSec": 240,
    //     "intervalSec": 15
    //   }
    // }
    // and then:
    // videoDocs.push(Document.fromMedia(filePath, getVideoType(filePath), metadataNext120Seconds));
    //
    // sample ~4 minute video: gs://cloud-samples-data/generative-ai/video/google_sustainability.mp4

    videoDocs.push(
      Document.fromMedia(
        filePath,
        getVideoType(filePath),
        metadataFirst120Seconds
      )
    );

    return videoDocs;
  }

  // Note, this is valid, but it only works for very very tiny videos.
  // Otherwise the API request message size is too big.
  // The recommended way to handle video is using a 'gs://' URL
  const file = path.resolve(filePath);
  const dataBuffer = fs.readFileSync(file);
  const detectedFileInfo = fileTypeChecker.detectFile(dataBuffer);
  if (
    detectedFileInfo?.mimeType &&
    detectedFileInfo?.mimeType.startsWith('video/')
  ) {
    videoDocs.push(
      Document.fromMedia(
        dataBuffer.toString('base64'),
        detectedFileInfo?.mimeType
      )
    );
  } else {
    throw new Error('Error: Unable to determine mime type of the file.');
  }
  return videoDocs;
}

// Define a video QA flow
export const localVideoQAFlow = ai.defineFlow(
  {
    name: 'localVideoQuestions',
    inputSchema: z
      .string()
      .describe('A question about the video')
      .default('describe the video'),
    outputSchema: z.string(),
  },
  async (query: string, { sendChunk }) => {
    // A single segment of video is enough context.
    const [doc] = await ai.run('retrieve', () =>
      videoStore.query(query, { k: 1 })
    );
    const video = doc?.media[0];
    if (!video?.url || !video.contentType) {
      throw new Error('No indexed video found. Run localIndexVideo first.');
    }
    const segment = VideoSegmentSchema.parse(doc.metadata?.embedMetadata);
    console.log(
      `Retrieved video: ${video.url} from ${segment.startOffsetSec}s to ${segment.endOffsetSec}s`
    );

    const { text } = await augmentedVideoPrompt(
      {
        question: query,
        media: {
          gcsUrl: video.url,
          contentType: video.contentType,
          ...segment,
        },
      },
      {
        onChunk: (c) => sendChunk(c.text),
      }
    );
    return text;
  }
);
