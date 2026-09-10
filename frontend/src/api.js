// Single place that knows how to talk to the backend.
//
// Keeping fetch calls out of components means the UI never has to reason about
// status codes -- it gets either data or an Error with a message already worth
// showing to a human.

const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8080'

/**
 * The backend uses status codes to mean specific things, and the difference
 * matters to the user:
 *   422 -> your question was malformed (you can fix it)
 *   502 -> the language model failed (retrying may work)
 *   503 -> the vector database is unreachable (retrying will not help yet)
 * Collapsing them all into "something went wrong" throws that away.
 */
async function request(path, options = {}) {
  let response
  try {
    response = await fetch(`${BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    })
  } catch (networkError) {
    // fetch only rejects on network failure / CORS, never on 4xx or 5xx.
    throw new Error(
      `Cannot reach the API at ${BASE}. Is the backend running?`,
    )
  }

  if (!response.ok) {
    let detail = ''
    try {
      const body = await response.json()
      detail = typeof body.detail === 'string' ? body.detail : ''
    } catch {
      // A non-JSON error body is not worth failing over.
    }

    const messages = {
      422: detail || 'That question could not be processed. Try rephrasing it.',
      502: 'The language model is unavailable right now. Try again in a moment.',
      503: 'The document database is unreachable. The service is starting up or down.',
    }
    throw new Error(messages[response.status] || detail || `Request failed (${response.status})`)
  }

  return response.json()
}

export function ask(question, topK) {
  return request('/query', {
    method: 'POST',
    body: JSON.stringify(topK ? { question, top_k: topK } : { question }),
  })
}

export function getReady() {
  return request('/ready')
}

export function getCacheStats() {
  return request('/cache')
}
