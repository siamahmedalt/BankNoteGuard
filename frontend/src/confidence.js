// Direct or averaged model outputs, not proof of authenticity.
export const isProbability = value => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1
export const formatPercent = value => isProbability(value) ? `${(value * 100).toFixed(1)}%` : '—'
export const isLegacyReview = item => item.status === 'REJECTED' && String(item.inferencePath || '').startsWith('review_')

export function scoreRows(item) {
  return [
    { label: 'REAL', value: item.realProbability, kind: 'real' },
    { label: 'FAKE', value: item.fakeProbability, kind: 'fake' },
    { label: 'INVALID (non-banknote)', value: item.invalidProbability, kind: 'invalid' },
  ]
}

export function topModelScore(item) {
  const rows = scoreRows(item)
  if (!rows.every(row => isProbability(row.value))) return null
  return [...rows].sort((a, b) => b.value - a.value)[0]
}

export function scanConfidence(item) {
  if (isLegacyReview(item)) {
    const top = topModelScore(item)
    return { label: 'Top model score', value: top?.value ?? null, detail: top ? top.label : null }
  }
  return { label: 'Prediction confidence', value: isProbability(item.confidence) ? item.confidence : null, detail: null }
}
