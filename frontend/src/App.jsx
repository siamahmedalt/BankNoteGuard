import { useEffect, useRef, useState } from 'react'
import { Camera, Check, Clock3, History, ImagePlus, RefreshCw, ScanLine, Upload, X } from 'lucide-react'

const ACCEPTED = new Set(['image/jpeg', 'image/png', 'image/webp'])
const percent = value => value == null ? '—' : `${(Number(value) * 100).toFixed(1)}%`
const dateLabel = value => value ? new Date(value).toLocaleString() : '—'
const needsReview = item => item.status === 'REJECTED' && String(item.inferencePath || '').startsWith('review_')
const typeLabel = item => needsReview(item) ? 'Unverified' : item.status === 'REJECTED' ? 'Invalid' : 'Banknote'
const statusLabel = item => needsReview(item) ? 'UNCERTAIN' : item.status === 'REJECTED' ? 'INVALID' : item.status
const shortId = value => value ? String(value).slice(0, 8).toUpperCase() : '—'

export default function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  const [source, setSource] = useState('upload')
  const [cameraOn, setCameraOn] = useState(false)
  const [cameraError, setCameraError] = useState('')
  const [dragging, setDragging] = useState(false)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [system, setSystem] = useState(null)
  const [history, setHistory] = useState([])
  const [clearing, setClearing] = useState(false)
  const [historyError, setHistoryError] = useState('')
  const historyVersion = useRef(0)
  const inputRef = useRef(null)
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const previewRef = useRef('')

  const releasePreview = () => {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current)
    previewRef.current = ''
  }

  const clearSelection = () => {
    releasePreview()
    setPreview('')
    setFile(null)
    setResult(null)
    setError('')
    if (inputRef.current) inputRef.current.value = ''
  }

  const stopCamera = () => {
    if (streamRef.current) streamRef.current.getTracks().forEach(track => track.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setCameraOn(false)
  }

  const acceptFile = (nextFile, nextSource = 'upload') => {
    if (!nextFile) return
    if (!ACCEPTED.has(nextFile.type)) {
      setError('Choose a JPG, PNG or WebP image.')
      return
    }
    if (nextFile.size > 10 * 1024 * 1024) {
      setError('Maximum image size is 10 MB.')
      return
    }
    releasePreview()
    const url = URL.createObjectURL(nextFile)
    previewRef.current = url
    setFile(nextFile)
    setPreview(url)
    setSource(nextSource)
    setResult(null)
    setError('')
    setCameraError('')
  }

  const startCamera = async () => {
    setCameraError('')
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError('Camera access requires localhost or HTTPS.')
      return
    }
    try {
      clearSelection()
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 1280 } },
        audio: false
      })
      streamRef.current = stream
      setSource('camera')
      setCameraOn(true)
    } catch (cameraException) {
      setCameraError(cameraException?.message || 'Camera permission was denied.')
    }
  }

  useEffect(() => {
    if (cameraOn && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current
      videoRef.current.play().catch(() => {})
    }
  }, [cameraOn])

  const capture = () => {
    const video = videoRef.current
    if (!video?.videoWidth) {
      setCameraError('Camera is not ready.')
      return
    }
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height)
    canvas.toBlob(blob => {
      if (blob) {
        const captured = new File([blob], `banknote-${Date.now()}.jpg`, { type: 'image/jpeg' })
        acceptFile(captured, 'camera')
      }
    }, 'image/jpeg', 0.94)
    stopCamera()
  }

  const refreshSystem = async () => {
    try {
      const response = await fetch('/api/system', { cache: 'no-store' })
      if (!response.ok) throw new Error(`Backend returned ${response.status}`)
      const data = await response.json()
      setSystem(data)
      return data
    } catch (e) {
      const data = { api: 'offline', ai: { ready: false, message: e.message || 'Backend unavailable' } }
      setSystem(data)
      return data
    }
  }

  const refreshHistory = async () => {
    const version = historyVersion.current
    try {
      const response = await fetch('/api/scans', { cache: 'no-store' })
      if (response.ok) {
        const data = await response.json()
        if (Array.isArray(data) && version === historyVersion.current) setHistory(data)
      }
    } catch { /* Preserve the last known history on brief outages. */ }
  }

  const clearHistory = async () => {
    if (busy || clearing || !window.confirm('Clear all scan history?')) return
    setClearing(true)
    setHistoryError('')
    historyVersion.current += 1
    try {
      const response = await fetch('/api/scans', { method: 'DELETE' })
      if (!response.ok) throw new Error(`Could not clear history (${response.status})`)
      historyVersion.current += 1
      setHistory([])
    } catch (err) { setHistoryError(err.message) }
    finally { setClearing(false) }
  }

  const refresh = () => Promise.all([refreshSystem(), refreshHistory()])

  useEffect(() => {
    refresh()
    const timer = window.setInterval(refreshSystem, 4000)
    return () => {
      window.clearInterval(timer)
      if (streamRef.current) streamRef.current.getTracks().forEach(track => track.stop())
      releasePreview()
    }
  }, [])

  const analyze = async () => {
    if (!file || busy || clearing) return
    setBusy(true)
    setError('')
    setResult(null)
    const body = new FormData()
    body.append('image', file)
    body.append('source', source)
    try {
      const response = await fetch('/api/scans', { method: 'POST', body })
      let data = {}
      try { data = await response.json() } catch {}
      if (!response.ok) {
        const detail = data.error || data.detail || data.message || `Request failed (${response.status})`
        throw new Error(response.status === 403
          ? `Request forbidden (403). Check backend CORS for localhost:5174. ${detail}`
          : detail)
      }
      setResult(data)
      await refresh()
    } catch (requestError) {
      setError(requestError?.message || 'Scan failed.')
      await refreshSystem()
    } finally {
      setBusy(false)
    }
  }

  const aiReady = Boolean(system?.ai?.ready)
  const aiStatus = !system ? 'Checking AI status…'
    : system?.api !== 'online' ? 'Java backend is not responding'
    : aiReady ? ''
    : system?.ai?.message || 'AI is initializing…' 
  const onDrop = event => {
    event.preventDefault()
    setDragging(false)
    const dropped = event.dataTransfer.files?.[0]
    if (dropped) acceptFile(dropped, 'upload')
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="wordmark" href="#scanner" aria-label="BankNoteGuard">
          <img className="wordmark-icon" src="/taka-logo.svg" alt="" aria-hidden="true" />
          <span>BANKNOTE GUARD</span>
        </a>
        <nav className="navigation" aria-label="Main navigation">
          <a href="#scanner">Scanner</a>
          <a href="#history">History</a>
        </nav>
      </header>

      <main className="main-content">
        <section className="scanner-section" id="scanner">
          <div className="section-heading">
            <h1>Scanner</h1>
          </div>

          <div
            className={`scanner-square ${dragging ? 'dragging' : ''}`}
            onDragOver={event => { event.preventDefault(); setDragging(true) }}
            onDragLeave={event => { event.preventDefault(); setDragging(false) }}
            onDrop={onDrop}
          >
            {cameraOn ? (
              <video ref={videoRef} className="scanner-media camera-media" autoPlay muted playsInline />
            ) : preview ? (
              <img className="scanner-media image-media" src={preview} alt="Selected banknote" />
            ) : (
              <div className="scanner-empty" aria-hidden="true">
                <ScanLine size={58} strokeWidth={1.1} />
              </div>
            )}

            {(preview || cameraOn) && <div className="scan-corners" aria-hidden="true"><i /><i /><i /><i /></div>}

            {preview && (
              <button className="clear-image" onClick={clearSelection} aria-label="Remove image" title="Remove image">
                <X size={18} />
              </button>
            )}
          </div>

          <input
            ref={inputRef}
            className="file-input"
            type="file"
            accept="image/jpeg,image/png,image/webp"
            onChange={event => {
              acceptFile(event.target.files?.[0], 'upload')
              event.target.value = ''
            }}
          />

          <div className="scanner-controls">
            <button className="control-button" onClick={() => inputRef.current?.click()}>
              <ImagePlus size={18} /> Upload
            </button>
            <button className="control-button" onClick={cameraOn ? stopCamera : startCamera}>
              <Camera size={18} /> {cameraOn ? 'Close camera' : 'Camera'}
            </button>
            <button
              className="control-button primary"
              onClick={cameraOn ? capture : analyze}
              disabled={cameraOn ? false : (!file || busy || clearing)}
            >
              {cameraOn ? <Check size={18} /> : <ScanLine size={18} />}
              {cameraOn ? 'Capture' : busy ? 'Scanning' : 'Analyze'}
            </button>
          </div>

          {!aiReady && !cameraError && !error && <div className="inline-message" role="status">{aiStatus}</div>}
          {(cameraError || error) && <div className="inline-message" role="alert">{cameraError || error}</div>}

          {result && (
            <section className={`scan-result ${String(result.status).toLowerCase()}`} aria-live="polite">
              <div className="result-primary">
                <strong>{statusLabel(result)}</strong>
                <span>{result.status === 'REJECTED' ? (result.reason || 'Not detected') : typeLabel(result)}</span>
              </div>
              <div className="result-meta">
                <span>{result.confidence == null ? '—' : percent(result.confidence)}</span>
                <span>{result.durationMs ?? '—'} ms</span>
                <span>#{shortId(result.id || result.requestId)}</span>
              </div>
            </section>
          )}
        </section>

        <section className="history-section" id="history">
          <div className="history-heading">
            <h2>History</h2>
            <div className="history-actions"><button className="clear-history-button" disabled={busy || clearing} onClick={clearHistory}>{clearing ? 'Clearing…' : 'Clear History'}</button>
            <button className="refresh-button" disabled={clearing} onClick={refresh} aria-label="Refresh history" title="Refresh history">
              <RefreshCw size={17} />
            </button></div>
          </div>

          {historyError && <div className="inline-message" role="alert">{historyError}</div>}
          <div className="history-table">
            <div className="history-header-row">
              <span>DATE</span>
              <span>TYPE</span>
              <span>IMAGE</span>
              <span>RESULT</span>
              <span>CONFIDENCE</span>
              <span>TIME</span>
            </div>
            {history.length ? history.map(item => (
              <div className="history-row" key={item.id}>
                <time>{dateLabel(item.createdAt)}</time>
                <span>{typeLabel(item)}</span>
                <div className="history-thumbnail"><img src={`/api/scans/${encodeURIComponent(item.id)}/image`} alt="Scanned note" loading="lazy" onError={event => { event.currentTarget.hidden = true; event.currentTarget.nextElementSibling.hidden = false }} /><span hidden>Unavailable</span></div>
                <strong>{statusLabel(item)}</strong>
                <span>{percent(item.confidence)}</span>
                <span>{item.durationMs == null ? '—' : `${item.durationMs} ms`}</span>
              </div>
            )) : <div className="history-empty"><History size={20} /></div>}
          </div>
        </section>
      </main>
    </div>
  )
}
