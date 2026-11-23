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

import AdmZip from 'adm-zip';
import axios from 'axios';
import { existsSync, mkdirSync, writeFileSync } from 'fs';
import path from 'path';
import { logger } from './logger.js';

interface DownloadAndExtractOptions {
  fileUrl: string;
  extractPath: string;
  zipFileName: string;
}

// Helper function for downloading UI zip file and extracting it to local fs.
export async function downloadAndExtractUiAssets({
  fileUrl,
  extractPath,
  zipFileName,
}: DownloadAndExtractOptions) {
  try {
    const downloadedFilePath = path.join(extractPath, zipFileName);
    if (!existsSync(downloadedFilePath)) {
      const response = await axios({
        url: fileUrl,
        method: 'GET',
        responseType: 'arraybuffer',
      });

      // Save the downloaded zip file
      mkdirSync(extractPath, { recursive: true });
      writeFileSync(downloadedFilePath, response.data);
    }

    // Extract the entire content of the zip file
    const zip = new AdmZip(downloadedFilePath);
    zip.extractAllTo(extractPath, true /* overwrite any existing files*/);
  } catch (error) {
    logger.error('Error downloading or extracting UI assets zip: ', error);
  }
}
