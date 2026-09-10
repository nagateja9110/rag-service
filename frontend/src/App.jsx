import { useCallback, useEffect, useState } from 'react'
import { ask, getReady } from './api.js'
import AskBox from './components/AskBox.jsx'
import AnswerCard from './components/AnswerCard.jsx'
import StatusBar from './components/StatusBar.jsx'

export default function App() {
  const [status, setStatus] = useState({ state: 'checking' })
  const [result, setResult] = useState(null)
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Poll readiness so the UI can say "the backend is down" instead of only
  // failing once the user has typed a question and pressed Ask.
  const refreshStatus = useCallback(async () => {
    try {
      const ready = await getReady()
      setStatus({ state: 'ready', ...ready })
    } catch (err) {
      setStatus({ state: 'down', message: err.message })
    }
  }, [])

  useEffect(() => {
    refreshStatus()
    const id = setInterval(refreshStatus, 15000)
    return () => clearInterval(id)
  }, [refreshStatus])

  async function handleAsk(text) {
    setLoading(true)
    setError(null)
    setQuestion(text)
    try {
      setResult(await ask(text))
    } catch (err) {
      setError(err.message)
      setResult(null)
    } finally {
      setLoading(false)
      refreshStatus()
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <span className="brand-mark">◆</span>
          <div>
            <h1>Ask your documents</h1>
            <p className="tagline">
              Answers grounded in your files — with citations, or an honest
              &ldquo;I don&rsquo;t know&rdquo;.
            </p>
          </div>
        </div>
        <StatusBar status={status} />
      </header>

      <main className="main">
        <AskBox onAsk={handleAsk} loading={loading} disabled={status.state === 'down'} />

        {error && (
          <div className="card card-error" role="alert">
            <strong>Couldn&rsquo;t answer that</strong>
            <p>{error}</p>
          </div>
        )}

        {loading && (
          <div className="card card-loading" aria-live="polite">
            <div className="pipeline-steps">
              {['Searching', 'Ranking', 'Writing'].map((step, i) => (
                <span key={step} className="step" style={{ animationDelay: `${i * 0.2}s` }}>
                  {step}
                </span>
              ))}
            </div>
            <div className="skeleton" />
            <div className="skeleton short" />
          </div>
        )}

        {!loading && result && <AnswerCard result={result} question={question} />}

        {!loading && !result && !error && (
          <div className="empty">
            <p>Ask a question about the documents that have been indexed.</p>
          </div>
        )}
      </main>

      <footer className="footer">
        Hybrid retrieval (vector + keyword) → cross-encoder re-ranking → grounded
        generation
      </footer>
    </div>
  )
}
