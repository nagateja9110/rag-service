import { useState } from 'react'

/**
 * Exposes what the pipeline actually did. Mostly this is a debugging aid, but
 * it is also the clearest way to show that hybrid retrieval is real: `overlap`
 * counts the chunks BOTH retrievers found independently, which is exactly the
 * agreement signal RRF is built to reward.
 */
export default function RetrievalDetails({ retrieval, timings }) {
  const [open, setOpen] = useState(false)
  if (!retrieval && !timings) return null

  const stages = [
    ['Vector search', retrieval?.vector_hits],
    ['Keyword search (BM25)', retrieval?.bm25_hits],
    ['Found by both', retrieval?.overlap],
    ['After fusion', retrieval?.fused_candidates],
  ].filter(([, v]) => v != null)

  const timing = [
    ['Retrieve', timings?.retrieval],
    ['Re-rank', timings?.rerank],
    ['Generate', timings?.synthesis],
  ].filter(([, v]) => v != null)

  return (
    <div className="details">
      <button className="details-toggle" onClick={() => setOpen(!open)} aria-expanded={open}>
        {open ? 'Hide' : 'Show'} pipeline details
      </button>
      {open && (
        <div className="details-body">
          <div className="details-col">
            {stages.map(([label, value]) => (
              <div key={label} className="kv">
                <span>{label}</span>
                <strong>{value}</strong>
              </div>
            ))}
            {retrieval?.bm25_available === false && (
              <div className="kv warn">
                <span>Keyword search</span>
                <strong>unavailable</strong>
              </div>
            )}
          </div>
          <div className="details-col">
            {timing.map(([label, value]) => (
              <div key={label} className="kv">
                <span>{label}</span>
                <strong>{Math.round(value)} ms</strong>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
