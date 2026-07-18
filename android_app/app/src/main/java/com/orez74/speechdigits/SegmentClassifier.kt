package com.orez74.speechdigits

/**
 * Single-digit classification engine: preprocessing, multi-candidate normalization,
 * inference, and confidence-gated decision logic (Digit/Unsure/TooQuiet).
 */
class SegmentClassifier(
    private val modelLoader: TfliteModelLoader,
    private val preprocessor: AudioPreprocessor = AudioPreprocessor
) {

    companion object {
        private const val TAG = "SegmentClassifier"

        const val MIN_CONFIDENCE_THRESHOLD = 0.55f
        const val MIN_TOP1_TOP2_MARGIN = 0.10f
        const val TARGET_SEGMENT_SAMPLES = 16000
        const val MIN_SEGMENT_SAMPLES = 3200
        const val MAX_SEGMENT_SAMPLES = 12800
    }

    fun classifyWithConfidence(
        pcm16: ShortArray,
        sampleRate: Int = AudioPreprocessor.TARGET_SAMPLE_RATE
    ): Pair<String, Float>? {
        val output = classifyWithMetrics(pcm16, sampleRate) ?: return null
        if (output.result == RawDigitClassifier.ClassificationResult.TOO_QUIET) return null
        return Pair(output.label ?: "_unsure_", output.confidence)
    }

    fun classifyWithMetrics(
        pcm16: ShortArray,
        sampleRate: Int = AudioPreprocessor.TARGET_SAMPLE_RATE
    ): RawDigitClassifier.ClassificationOutput? {
        AppLog.v(TAG) { "CLASSIFY: ${pcm16.size} samples @ $sampleRate Hz" }

        val resampled = if (sampleRate != preprocessor.TARGET_SAMPLE_RATE) {
            AppLog.v(TAG) { "Resampling: $sampleRate Hz -> ${preprocessor.TARGET_SAMPLE_RATE} Hz" }
            preprocessor.resampleLinear(pcm16, sampleRate, preprocessor.TARGET_SAMPLE_RATE)
        } else {
            pcm16
        }

        val processed = preprocessor.padOrTruncate(resampled, preprocessor.EXPECTED_SAMPLES)

        val peak = preprocessor.calculatePeak(processed)
        val rms = preprocessor.calculateRms(processed)
        AppLog.v(TAG) { "Audio: Peak=$peak, RMS=${"%.1f".format(rms)}" }

        if (peak < preprocessor.PEAK_TOO_QUIET_THRESHOLD || rms < preprocessor.RMS_TOO_QUIET_THRESHOLD) {
            AppLog.d(TAG) { "TOO_QUIET: Peak=$peak, RMS=${"%.1f".format(rms)}" }
            return RawDigitClassifier.ClassificationOutput(
                result = RawDigitClassifier.ClassificationResult.TOO_QUIET,
                label = null,
                confidence = 0f,
                rmsAmplitude = rms,
                peakAmplitude = peak,
                allProbabilities = FloatArray(12) { 0f }
            )
        }

        val normalized = preprocessor.normalize(processed)
        val inputBuffer = modelLoader.createInputBuffer(normalized)
        val probabilities = modelLoader.runInference(inputBuffer)

        val labels = modelLoader.getLabels()
        val sortedIndices = probabilities.indices.sortedByDescending { probabilities[it] }
        val topIndex = sortedIndices[0]
        val top2Index = sortedIndices[1]
        val confidence = probabilities[topIndex]
        val secondBest = probabilities[top2Index]
        val margin = confidence - secondBest

        val label = if (topIndex < labels.size) labels[topIndex] else topIndex.toString()

        AppLog.v(TAG) { "Top-1: $label (${(confidence*100).toInt()}%), Top-2: ${if (top2Index < labels.size) labels[top2Index] else top2Index} (${(secondBest*100).toInt()}%), Margin: ${(margin*100).toInt()}%" }

        val result: RawDigitClassifier.ClassificationResult
        val finalLabel: String?

        when {
            label == "_silence_" || label == "_unknown_" -> {
                result = RawDigitClassifier.ClassificationResult.UNSURE
                finalLabel = null
                AppLog.d(TAG) { "UNSURE: Special class '$label' (${(confidence*100).toInt()}%)" }
            }
            confidence < MIN_CONFIDENCE_THRESHOLD -> {
                result = RawDigitClassifier.ClassificationResult.UNSURE
                finalLabel = null
                AppLog.d(TAG) { "UNSURE: Low confidence $label (${(confidence*100).toInt()}%)" }
            }
            margin < MIN_TOP1_TOP2_MARGIN -> {
                result = RawDigitClassifier.ClassificationResult.UNSURE
                finalLabel = null
                AppLog.d(TAG) { "UNSURE: Small margin $label (${(margin*100).toInt()}%)" }
            }
            else -> {
                result = RawDigitClassifier.ClassificationResult.DIGIT
                finalLabel = label
                AppLog.d(TAG) { "DIGIT: $label (${(confidence*100).toInt()}%)" }
            }
        }

        return RawDigitClassifier.ClassificationOutput(
            result = result,
            label = finalLabel,
            confidence = confidence,
            rmsAmplitude = rms,
            peakAmplitude = peak,
            allProbabilities = probabilities
        )
    }

    // ==========================================
    // Segment-level classification (PIN pipeline)
    // ==========================================

    fun extractAndNormalizeSegment(
        pcm16: ShortArray,
        segment: RawDigitClassifier.SpeechSegment
    ): ShortArray {
        val safeStart = segment.startSample.coerceIn(0, pcm16.size - 1)
        val safeEnd = segment.endSample.coerceIn(safeStart + 1, pcm16.size)
        val rawSegment = pcm16.sliceArray(safeStart until safeEnd)
        val rawLength = rawSegment.size

        AppLog.v(TAG) { "Segment: ${rawLength} samples (${rawLength * 1000 / preprocessor.TARGET_SAMPLE_RATE}ms)" }

        val candidates = generateNormalizationCandidates(rawSegment)
        return selectBestCandidate(candidates)
    }

    private data class NormCandidate(val name: String, val audio: ShortArray)

    private fun generateNormalizationCandidates(rawSegment: ShortArray): List<NormCandidate> {
        val rawLength = rawSegment.size
        val candidates = mutableListOf<NormCandidate>()

        if (rawLength >= TARGET_SEGMENT_SAMPLES) {
            val excess = rawLength - TARGET_SEGMENT_SAMPLES

            candidates.add(NormCandidate("crop_start", rawSegment.copyOfRange(0, TARGET_SEGMENT_SAMPLES)))

            val centerStart = excess / 2
            candidates.add(NormCandidate("crop_center", rawSegment.copyOfRange(centerStart, centerStart + TARGET_SEGMENT_SAMPLES)))

            candidates.add(NormCandidate("crop_end", rawSegment.copyOfRange(excess, rawLength)))

            val peakPos = findPeakPosition(rawSegment)
            val peakOffset = (TARGET_SEGMENT_SAMPLES * 0.4).toInt()
            val peakStart = (peakPos - peakOffset).coerceIn(0, excess)
            candidates.add(NormCandidate("crop_peak40", rawSegment.copyOfRange(peakStart, peakStart + TARGET_SEGMENT_SAMPLES)))

        } else {
            val padTotal = TARGET_SEGMENT_SAMPLES - rawLength

            val padStart = ShortArray(TARGET_SEGMENT_SAMPLES)
            rawSegment.copyInto(padStart, 0)
            candidates.add(NormCandidate("pad_start", padStart))

            val padCenter = ShortArray(TARGET_SEGMENT_SAMPLES)
            val padBefore = padTotal / 2
            rawSegment.copyInto(padCenter, padBefore)
            candidates.add(NormCandidate("pad_center", padCenter))

            val padEnd = ShortArray(TARGET_SEGMENT_SAMPLES)
            rawSegment.copyInto(padEnd, padTotal)
            candidates.add(NormCandidate("pad_end", padEnd))
        }

        return candidates
    }

    private fun findPeakPosition(segment: ShortArray): Int {
        val windowSize = 320
        val hopSize = 160
        var maxRms = 0.0
        var peakPos = segment.size / 2

        var pos = 0
        while (pos + windowSize <= segment.size) {
            var sum = 0.0
            for (i in pos until pos + windowSize) {
                sum += segment[i].toDouble() * segment[i].toDouble()
            }
            val rms = kotlin.math.sqrt(sum / windowSize)
            if (rms > maxRms) {
                maxRms = rms
                peakPos = pos + windowSize / 2
            }
            pos += hopSize
        }
        return peakPos
    }

    private fun selectBestCandidate(candidates: List<NormCandidate>): ShortArray {
        if (candidates.isEmpty()) {
            return ShortArray(TARGET_SEGMENT_SAMPLES)
        }
        if (candidates.size == 1) {
            AppLog.v(TAG) { "Single candidate: ${candidates[0].name}" }
            return candidates[0].audio
        }

        val labels = modelLoader.getLabels()
        var bestCandidate = candidates[0]
        var bestScore = Float.NEGATIVE_INFINITY
        var bestLabel = ""

        for (candidate in candidates) {
            val normalized = preprocessor.normalize(candidate.audio)
            val inputBuffer = modelLoader.createInputBuffer(normalized)
            val probabilities = modelLoader.runInference(inputBuffer)

            val sortedIndices = probabilities.indices.sortedByDescending { probabilities[it] }
            val topIndex = sortedIndices[0]
            val top2Index = sortedIndices[1]
            val confidence = probabilities[topIndex]
            val margin = confidence - probabilities[top2Index]
            val label = if (topIndex < labels.size) labels[topIndex] else "?"

            var score = margin * 0.7f + confidence * 0.3f

            if (label == "_unknown_" || label == "_silence_") {
                score -= 0.5f
            }

            AppLog.v(TAG) { "  Candidate ${candidate.name}: $label conf=${(confidence*100).toInt()}% margin=${(margin*100).toInt()}% score=${(score*100).toInt()}" }

            if (score > bestScore) {
                bestScore = score
                bestCandidate = candidate
                bestLabel = label
            }
        }

        AppLog.d(TAG) { "-> Best candidate: ${bestCandidate.name} ($bestLabel, score=${(bestScore*100).toInt()})" }
        return bestCandidate.audio
    }

    fun classifySegment(
        audio: ShortArray,
        segmentIndex: Int,
        segment: RawDigitClassifier.SpeechSegment
    ): RawDigitClassifier.SegmentResult {
        val normalized = preprocessor.normalize(audio)
        val inputBuffer = modelLoader.createInputBuffer(normalized)
        val probabilities = modelLoader.runInference(inputBuffer)

        val labels = modelLoader.getLabels()
        val sortedIndices = probabilities.indices.sortedByDescending { probabilities[it] }
        val topIndex = sortedIndices[0]
        val top2Index = sortedIndices[1]
        val confidence = probabilities[topIndex]
        val margin = confidence - probabilities[top2Index]

        val label = if (topIndex < labels.size) labels[topIndex] else topIndex.toString()

        AppLog.v(TAG) {
            val top3 = sortedIndices.take(3).joinToString(", ") { idx ->
                val lbl = if (idx < labels.size) labels[idx] else "?"
                "$lbl: ${(probabilities[idx]*100).toInt()}%"
            }
            "Top 3 predictions: $top3"
        }

        val isDigit = label.length == 1 && label[0].isDigit()
        val segmentRms = segment.peakRms
        val durationMs = segment.durationMs

        val tooQuiet = segmentRms < PinAssemblyEngine.PIN_SEGMENT_MIN_RMS
        val tooShort = durationMs < PinAssemblyEngine.PIN_MIN_SEGMENT_DURATION_MS

        if (tooQuiet || tooShort) {
            val reason = when {
                tooQuiet && tooShort -> "low energy AND short"
                tooQuiet -> "low energy"
                else -> "short"
            }
            AppLog.v(TAG) { "SEGMENT[$segmentIndex]: UNSURE ($reason) - label=$label, rms=${segmentRms.toInt()} (min=${PinAssemblyEngine.PIN_SEGMENT_MIN_RMS}), dur=${durationMs}ms (min=${PinAssemblyEngine.PIN_MIN_SEGMENT_DURATION_MS})" }
            return RawDigitClassifier.SegmentResult(
                segmentIndex = segmentIndex,
                segment = segment,
                label = label,
                confidence = confidence,
                margin = margin,
                status = RawDigitClassifier.SegmentStatus.UNSURE
            )
        }

        val hardOk = isDigit &&
                     confidence >= PinAssemblyEngine.MIN_SEGMENT_CONFIDENCE &&
                     margin >= PinAssemblyEngine.MIN_SEGMENT_MARGIN

        val softOk = isDigit &&
                     confidence >= PinAssemblyEngine.SOFT_CONFIDENCE &&
                     margin >= PinAssemblyEngine.SOFT_MARGIN &&
                     segmentRms >= PinAssemblyEngine.SOFT_RMS

        val status = if (hardOk || softOk) RawDigitClassifier.SegmentStatus.OK else RawDigitClassifier.SegmentStatus.UNSURE

        when {
            hardOk -> AppLog.v(TAG) { "SEGMENT[$segmentIndex]: OK (HARD) - label=$label, conf=${"%.0f".format(confidence*100)}%, margin=${"%.0f".format(margin*100)}%, rms=${segmentRms.toInt()}, dur=${durationMs}ms" }
            softOk -> AppLog.v(TAG) { "SEGMENT[$segmentIndex]: OK (SOFT) - label=$label, conf=${"%.0f".format(confidence*100)}%, margin=${"%.0f".format(margin*100)}%, rms=${segmentRms.toInt()}, dur=${durationMs}ms" }
            else -> {
                val confOk = confidence >= PinAssemblyEngine.MIN_SEGMENT_CONFIDENCE
                val marginOk = margin >= PinAssemblyEngine.MIN_SEGMENT_MARGIN
                val reason = when {
                    !isDigit -> "not a digit ($label)"
                    !confOk && !marginOk -> "low conf AND low margin"
                    !confOk -> "low conf (${"%.0f".format(confidence*100)}% < ${(PinAssemblyEngine.MIN_SEGMENT_CONFIDENCE*100).toInt()}%)"
                    !marginOk -> "low margin (${"%.0f".format(margin*100)}% < ${(PinAssemblyEngine.MIN_SEGMENT_MARGIN*100).toInt()}%)"
                    else -> "unknown"
                }
                AppLog.v(TAG) { "SEGMENT[$segmentIndex]: UNSURE ($reason) - label=$label, conf=${"%.0f".format(confidence*100)}%, margin=${"%.0f".format(margin*100)}%, rms=${segmentRms.toInt()}, dur=${durationMs}ms" }
            }
        }

        return RawDigitClassifier.SegmentResult(
            segmentIndex = segmentIndex,
            segment = segment,
            label = label,
            confidence = confidence,
            margin = margin,
            status = status
        )
    }
}
