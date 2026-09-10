import { useState } from 'react'

export default function SourceList({ sources }) {
  const [openId, setOpenId] = useState(null)

  // Cross-encoder scores are unbounded logits, so there is no absolute scale to
  // draw a bar against. Normalising against the top hit shows RELATIVE
  // confidence, which is the only honest reading of these numbers.
  const top = Math.max(...sources.map((s) => s.score))
  const bottom = Math.min(...sources.map((s) => s.score))
  const span = top - bottom || 1

  return (
    <section className="sources">
      <h2>
        Sources <span className="count">{sources.length}</span>
      </h2>
      <ul>
        {sources.map((source) => {
          const open = openId === source.chunk_id
          const relative = (source.score - bottom) / span
          return (
            <li key={source.chunk_id} className="source">
              <button
                className="source-head"
                onClick={() => setOpenId(open ? null : source.chunk_id)}
                aria-expanded={open}
              >
                <span className="source-file">{source.file_name}</span>
                <span className="source-page">p.{source.page_number}</span>
                <span className="score-bar" aria-hidden="true">
                  <span
                    className="score-fill"
                    style={{ width: `${20 + relative * 80}%` }}
                  />
                </span>
                <span className="chev">{open ? '−' : '+'}</span>
              </button>
              {open && <p className="source-preview">{source.preview}</p>}
            </li>
          )
        })}
      </ul>
    </section>
  )
}
