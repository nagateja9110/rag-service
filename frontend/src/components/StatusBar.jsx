export default function StatusBar({ status }) {
  const map = {
    checking: { label: 'Connecting…', tone: 'muted' },
    ready: { label: 'Connected', tone: 'ok' },
    down: { label: 'Backend offline', tone: 'bad' },
  }
  const { label, tone } = map[status.state] ?? map.checking

  return (
    <div className={`status status-${tone}`} title={status.message || ''}>
      <span className="dot" aria-hidden="true" />
      <span className="status-label">{label}</span>
      {status.state === 'ready' && (
        <span className="status-meta">
          {status.vectors.toLocaleString()} chunks indexed · reranker:{' '}
          {status.reranker}
        </span>
      )}
      {status.state === 'down' && (
        <span className="status-meta">{status.message}</span>
      )}
    </div>
  )
}
