import { useState } from 'react'

// Questions that exercise the corpus, plus one the corpus deliberately cannot
// answer -- the refusal is a feature worth demonstrating, not an edge case to
// hide.
const EXAMPLES = [
  'What makes the ingestion pipeline idempotent?',
  'Why does chunk overlap exist?',
  'What index does Chroma use for approximate search?',
  'What were the Q3 revenue figures?',
]

export default function AskBox({ onAsk, loading, disabled }) {
  const [value, setValue] = useState('')

  function submit(text) {
    const question = (text ?? value).trim()
    // The backend rejects anything under 3 characters with a 422; catching it
    // here saves a round trip and gives instant feedback.
    if (question.length < 3 || loading || disabled) return
    setValue(question)
    onAsk(question)
  }

  function handleKeyDown(event) {
    // Enter submits, Shift+Enter makes a newline -- the convention people
    // already expect from chat inputs.
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      submit()
    }
  }

  return (
    <section className="askbox">
      <div className="askbox-input">
        <textarea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask something about your documents…"
          rows={3}
          disabled={disabled}
          aria-label="Your question"
        />
        <button
          className="primary"
          onClick={() => submit()}
          disabled={loading || disabled || value.trim().length < 3}
        >
          {loading ? 'Thinking…' : 'Ask'}
        </button>
      </div>

      <div className="examples">
        <span className="examples-label">Try:</span>
        {EXAMPLES.map((example) => (
          <button
            key={example}
            className="chip"
            onClick={() => submit(example)}
            disabled={loading || disabled}
          >
            {example}
          </button>
        ))}
      </div>
    </section>
  )
}
