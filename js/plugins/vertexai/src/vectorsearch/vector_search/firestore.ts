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

import type { Firestore } from 'firebase-admin/firestore';
import { Document, DocumentDataSchema } from 'genkit';
import type { DocumentIndexer, DocumentRetriever, Neighbor } from './types.js';
/**
 * Creates a Firestore Document Retriever.
 *
 * This function returns a DocumentRetriever function that retrieves documents
 * from a Firestore collection based on the provided Vertex AI Vector Search neighbors.
 *
 * @param {Firestore} db - The Firestore instance.
 * @param {string} collectionName - The name of the Firestore collection.
 * @returns {DocumentRetriever} - The DocumentRetriever function.
 */
export const getFirestoreDocumentRetriever = (
  db: Firestore,
  collectionName: string
): DocumentRetriever => {
  const firestoreRetriever: DocumentRetriever = async (
    neighbors: Neighbor[]
  ): Promise<Document[]> => {
    const docs: Document[] = [];
    for (const neighbor of neighbors) {
      const docRef = db
        .collection(collectionName)
        .doc(neighbor.datapoint?.datapointId!);
      const docSnapshot = await docRef.get();
      if (docSnapshot.exists) {
        const docData = { ...docSnapshot.data() }; // includes content & metadata
        docData.metadata = { ...docData.metadata, ...neighbor }; // add neighbor
        const parsedDocData = DocumentDataSchema.safeParse(docData);
        if (parsedDocData.success) {
          docs.push(new Document(parsedDocData.data));
        }
      }
    }
    return docs;
  };
  return firestoreRetriever;
};

/**
 * Creates a Firestore Document Indexer.
 *
 * This function returns a DocumentIndexer function that indexes documents
 * into a Firestore collection.
 *
 * @param {Firestore} db - The Firestore instance.
 * @param {string} collectionName - The name of the Firestore collection.
 * @returns {DocumentIndexer} - The DocumentIndexer function.
 */
export const getFirestoreDocumentIndexer = (
  db: Firestore,
  collectionName: string
) => {
  const firestoreIndexer: DocumentIndexer = async (
    docs: Document[]
  ): Promise<string[]> => {
    const batch = db.batch();
    const ids: string[] = [];
    docs.forEach((doc) => {
      const docRef = db.collection(collectionName).doc();
      batch.set(docRef, {
        content: doc.content,
        metadata: doc.metadata || null,
      });
      ids.push(docRef.id);
    });
    await batch.commit();
    return ids;
  };
  return firestoreIndexer;
};
