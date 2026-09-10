import SourceList from './SourceList.jsx'
import RetrievalDetails from './RetrievalDetails.jsx'

export default function AnswerCard({ result, question }) {
  const {
    answer,
    sources,
    grounded,
    cached,
    reranker_degraded: degraded,
    timings_ms: timings,
    retrieval,
  } = result

  return (
    <article className={`card answer ${grounded ? '' : 'answer-refused'}`}>
      <p className="asked">{question}</p>

      {/* A refusal is a CORRECT outcome, so it gets its own treatment rather
          than looking like a failed request. */}
      {!grounded && (
        <div className="refusal-flag">
          Not found in your documents — the model declined to guess
        </div>
      )}

      <p className="answer-text">{answer}</p>

      <div className="badges">
        {cached && <span className="badge badge-info">cached</span>}
        {degraded && (
          <span className="badge badge-warn" title="The re-ranker failed; results fell back to fusion order">
            reranker degraded
          </span>
        )}
        {timings?.total != null && (
          <span className="badge">{Math.round(timings.total)} ms</span>
        )}
      </div>

      {grounded && sources?.length > 0 && <SourceList sources={sources} />}

      <RetrievalDetails retrieval={retrieval} timings={timings} />
    </article>
  )
}
