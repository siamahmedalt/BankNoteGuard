import { test } from 'node:test'
import assert from 'node:assert/strict'
import { formatPercent, scanConfidence, scoreRows, topModelScore } from '../src/confidence.js'

for (const [status, path, scores, confidence] of [
  ['REAL', 'rotation_crop_consensus', [.91,.07,.02], .91],
  ['FAKE', 'rotation_crop_consensus', [.10,.85,.05], .85],
  ['REJECTED', 'non_banknote_confirmed', [.01,.02,.97], .97],
  ['REJECTED', 'review_uncertain', [.40,.58,.02], null],
  ['REJECTED', 'review_uncertain', [.75,.24,.01], null],
]) {
  test(`${status} ${path} ${scores}`, () => {
    const item = { status, inferencePath:path, confidence, realProbability:scores[0],fakeProbability:scores[1],invalidProbability:scores[2] }
    assert.deepEqual(scoreRows(item).map(r=>r.value),scores)
    const result = scanConfidence(item)
    assert.equal(result.value, confidence ?? Math.max(...scores))
    assert.equal(result.label, confidence === null ? 'Top model score' : 'Prediction confidence')
  })
}
test('legacy/no-model score is not invented', () => {
  const item = {status:'REJECTED',inferencePath:'review_uncertain',fakeProbability:.55}
  assert.equal(topModelScore(item), null)
  assert.equal(formatPercent(scanConfidence(item).value), '—')
  assert.equal(formatPercent(0), '0.0%')
  assert.equal(formatPercent(undefined), '—')
  assert.equal(formatPercent(NaN), '—')
  assert.equal(formatPercent(1.1), '—')
})
