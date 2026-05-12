import { GoogleAuth } from 'google-auth-library';

/**
 * Scopes required for Cloud Monitoring and Cloud Trace APIs.
 */
const SCOPES = [
  'https://www.googleapis.com/auth/cloud-platform',
  'https://www.googleapis.com/auth/monitoring.read',
  'https://www.googleapis.com/auth/trace.readonly',
];

let authInstance: GoogleAuth | null = null;

/**
 * Gets (or creates) a GoogleAuth instance configured for ADC.
 * This will use Application Default Credentials, which can be set up via:
 *   gcloud auth application-default login
 */
function getAuth(): GoogleAuth {
  if (!authInstance) {
    authInstance = new GoogleAuth({
      scopes: SCOPES,
    });
  }
  return authInstance;
}

/**
 * Gets an authenticated client that can make API calls.
 * The client handles token refresh automatically.
 */
export async function getAuthClient() {
  return getAuth().getClient();
}

/**
 * Gets the default project ID from ADC.
 * Returns undefined if no project is configured.
 */
export async function getProjectId(): Promise<string | undefined> {
  try {
    const projectId = await getAuth().getProjectId();
    return projectId || undefined;
  } catch {
    return undefined;
  }
}

/**
 * Gets an access token for making direct API calls.
 * Handles token refresh automatically.
 */
export async function getAccessToken(): Promise<string> {
  const client = await getAuthClient();
  const token = await client.getAccessToken();
  if (!token.token) {
    throw new Error(
      'Failed to get access token. Run: gcloud auth application-default login'
    );
  }
  return token.token;
}
