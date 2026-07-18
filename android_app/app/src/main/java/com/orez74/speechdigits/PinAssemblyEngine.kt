package com.orez74.speechdigits

import android.content.Context
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * High-level PIN classification engine: VAD-based segmentation,
 * Best-4 subsequence selection, Hard-OK/Soft-OK gating, and PIN assembly.
 */
class PinAssemblyEngine(
    private val segmentClassifier: SegmentClassifier,
    private val context: Context
) {

    companion object {
        private const val TAG = "PinAssemblyEngine"

        const val PIN_RECORD_DURATION_MS = 4000
        const val PIN_TOTAL_SAMPLES = 64000

        const val VAD_SPEECH_START_RMS = 200f
        const val VAD_SPEECH_END_RMS = 100f
        const val VAD_MIN_SPEECH_MS = 200
        const val VAD_MIN_SILENCE_MS = 120
        const val VAD_FRAME_MS = 20
        const val VAD_PRE_ROLL_FRAMES = 2
        const val VAD_POST_ROLL_FRAMES = 2

        const val PIN_MIN_SEGMENT_DURATION_MS = 120

        const val MIN_SEGMENT_CONFIDENCE = 0.65f
        const val MIN_SEGMENT_MARGIN = 0.15f

        const val SOFT_CONFIDENCE = 0.55f
        const val SOFT_MARGIN = 0.20f
        const val SOFT_RMS = 900f

        const val EXPECTED_PIN_LENGTH = 4
        const val MIN_OK_DIGITS = 3

        const val PIN_GLOBAL_MIN_RMS = 220f
        const val PIN_GLOBAL_MIN_PEAK = 800

        const val PIN_SEGMENT_MIN_RMS = 600f

        const val USE_WEBRTC_VAD = true
    }

    private var lastDebugExport: RawDigitClassifier.DebugExport? = null

    fun getLastDebugExport(): RawDigitClassifier.DebugExport? = lastDebugExport

    fun classifyPin(
        pcm16: ShortArray,
        sampleRate: Int = AudioPreprocessor.TARGET_SAMPLE_RATE
    ): RawDigitClassifier.PinResult {
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())

        AppLog.separator(TAG)
        AppLog.d(TAG) { "PIN MODE (VAD): Classifying ${pcm16.size} samples @ $sampleRate Hz" }
        AppLog.separator(TAG)

        require(sampleRate == AudioPreprocessor.TARGET_SAMPLE_RATE) {
            "PIN mode requires ${AudioPreprocessor.TARGET_SAMPLE_RATE} Hz, got $sampleRate Hz"
        }

        // ==========================================
        // GLOBAL NO-SPEECH GATE
        // ==========================================
        var globalMaxAbs = 0
        var globalSumSquares = 0.0
        for (s in pcm16) {
            val v = s.toInt()
            val abs = kotlin.math.abs(v)
            if (abs > globalMaxAbs) globalMaxAbs = abs
            globalSumSquares += v.toDouble() * v.toDouble()
        }
        val globalRms = kotlin.math.sqrt(globalSumSquares / pcm16.size).toFloat()

        AppLog.d(TAG) { "PIN Global: peak=$globalMaxAbs, rms=${globalRms.toInt()}, thresholds: minPeak=$PIN_GLOBAL_MIN_PEAK, minRms=${PIN_GLOBAL_MIN_RMS.toInt()}" }

        val globalRmsOk = globalRms >= PIN_GLOBAL_MIN_RMS
        val globalPeakOk = globalMaxAbs >= PIN_GLOBAL_MIN_PEAK

        if (!globalRmsOk || !globalPeakOk) {
            AppLog.d(TAG) { "PIN Global Gate: NO SPEECH (peak=$globalMaxAbs, rms=${globalRms.toInt()}) -> Returning empty PIN" }
            AppLog.separator(TAG)
            return RawDigitClassifier.PinResult(
                pin = "",
                segmentResults = emptyList(),
                segments = emptyList(),
                totalSegments = 0,
                validDigits = 0,
                allSegmentsOk = false,
                pipeline = "VAD"
            )
        }
        AppLog.d(TAG) { "PIN Global Gate: PASSED (speech detected)" }

        // DEBUG: Save complete PIN recording
        val fullRecordingPath = WavExporter.saveDebugWav(context, "pin_full", pcm16, sampleRate)

        // Step 1: VAD
        val vadLogBuilder = StringBuilder()
        val vadPipeline = if (USE_WEBRTC_VAD) "WEBRTC" else "RMS"
        AppLog.d(TAG) { "VAD pipeline: $vadPipeline" }
        vadLogBuilder.append("VAD pipeline: $vadPipeline\n")

        val vadStartMs = if (AppLog.DEBUG_ENABLED) System.currentTimeMillis() else 0L
        val segments = if (USE_WEBRTC_VAD) {
            WebRtcVadSegmenter.detectSpeechSegments(pcm16, sampleRate, vadLogBuilder)
        } else {
            detectSpeechSegments(pcm16, sampleRate, vadLogBuilder)
        }
        if (AppLog.DEBUG_ENABLED) {
            val vadLatencyMs = System.currentTimeMillis() - vadStartMs
            AppLog.d(TAG) { "[PERFORMANCE] VAD processing: ${vadLatencyMs}ms (${segments.size} segments)" }
        }
        AppLog.d(TAG) { "VAD detected ${segments.size} speech segments" }

        // Step 2: Classify each segment
        val segmentResults = mutableListOf<RawDigitClassifier.SegmentResult>()
        val segmentPaths = mutableListOf<String>()

        for ((index, segment) in segments.withIndex()) {
            AppLog.d(TAG) { "--- Segment $index: ${segment.startSample}..${segment.endSample} (${segment.durationMs}ms, peakRMS=${segment.peakRms.toInt()}) ---" }

            val segmentAudio = segmentClassifier.extractAndNormalizeSegment(pcm16, segment)

            if (WavExporter.DEBUG_MODE_ENABLED) {
                val safeStart = segment.startSample.coerceIn(0, pcm16.size - 1)
                val safeEnd = segment.endSample.coerceIn(safeStart + 1, pcm16.size)
                val rawSegment = pcm16.sliceArray(safeStart until safeEnd)
                val path = WavExporter.saveDebugWav(context, "pin_seg$index", rawSegment, sampleRate)
                if (path != null) segmentPaths.add(path)
            }

            val result = segmentClassifier.classifySegment(segmentAudio, index, segment)
            segmentResults.add(result)

            val statusIcon = if (result.status == RawDigitClassifier.SegmentStatus.OK) "[OK]" else "[UNSURE]"
            AppLog.d(TAG) { "Result: ${result.label} (conf=${(result.confidence*100).toInt()}%, margin=${(result.margin*100).toInt()}%) $statusIcon" }

            vadLogBuilder.append("Segment $index: ${result.label} (${(result.confidence*100).toInt()}%) $statusIcon\n")
        }

        // Step 3: FILTER short/noise segments
        val filteredSegmentResults = segmentResults.filter { result ->
            val durationMs = result.segment.durationMs
            val peakRms = result.segment.peakRms

            val isShortUnsure = result.status == RawDigitClassifier.SegmentStatus.UNSURE &&
                    durationMs < PIN_MIN_SEGMENT_DURATION_MS

            val isShortUnsureDigit = !result.isValidDigit && result.label.matches(Regex("[0-9]")) && durationMs < PIN_MIN_SEGMENT_DURATION_MS

            val isLowEnergyUnsure = result.status == RawDigitClassifier.SegmentStatus.UNSURE &&
                    peakRms < PIN_SEGMENT_MIN_RMS

            val keep = !isShortUnsure && !isShortUnsureDigit && !isLowEnergyUnsure
            if (!keep) {
                AppLog.d(TAG) { "FILTER: Segment ${result.segmentIndex} removed (${result.label}, dur=${durationMs}ms, rms=${peakRms.toInt()}, status=${result.status})" }
            }
            keep
        }

        AppLog.d(TAG) { "Segments after filter: ${filteredSegmentResults.size} (before: ${segmentResults.size})" }

        // Step 4: PIN aggregation
        val rawDigitSegments = filteredSegmentResults.filter { it.label.length == 1 && it.label[0].isDigit() }

        val digitSegments = if (rawDigitSegments.size > EXPECTED_PIN_LENGTH) {
            AppLog.d(TAG) { "Best-4-Selection: ${rawDigitSegments.size} candidates, selecting best $EXPECTED_PIN_LENGTH" }
            selectBestSubsequence(rawDigitSegments, EXPECTED_PIN_LENGTH)
        } else {
            rawDigitSegments
        }

        val okDigits = digitSegments.filter { it.status == RawDigitClassifier.SegmentStatus.OK }
        val okCount = okDigits.size
        val validCount = digitSegments.size
        val unsureCount = digitSegments.count { it.status == RawDigitClassifier.SegmentStatus.UNSURE }

        val pinAccepted = (validCount == EXPECTED_PIN_LENGTH) && (okCount >= MIN_OK_DIGITS)

        val pin = if (pinAccepted) {
            digitSegments.take(EXPECTED_PIN_LENGTH).joinToString("") { it.label }
        } else {
            ""
        }

        val validDigits = okCount
        val allSegmentsOk = digitSegments.isNotEmpty() && digitSegments.all { it.status == RawDigitClassifier.SegmentStatus.OK }

        AppLog.separator(TAG)
        AppLog.d(TAG) { "PIN RESULT (VAD): \"$pin\" - Digits: $validCount, OK: $okCount, UNSURE: $unsureCount, Accepted: $pinAccepted (need $EXPECTED_PIN_LENGTH digits, min $MIN_OK_DIGITS OK)" }
        filteredSegmentResults.forEach { r ->
            val statusIcon = if (r.status == RawDigitClassifier.SegmentStatus.OK) "[OK]" else "[UNSURE]"
            AppLog.v(TAG) { "[${r.segmentIndex}] ${r.label}: conf=${(r.confidence*100).toInt()}%, margin=${(r.margin*100).toInt()}%, rms=${r.segment.peakRms.toInt()}, dur=${r.segment.durationMs}ms $statusIcon" }
        }
        AppLog.separator(TAG)

        if (WavExporter.DEBUG_MODE_ENABLED) {
            lastDebugExport = RawDigitClassifier.DebugExport(
                timestamp = timestamp,
                fullRecordingPath = fullRecordingPath,
                segmentPaths = segmentPaths,
                vadLog = vadLogBuilder.toString()
            )
        }

        return RawDigitClassifier.PinResult(
            pin = pin,
            segmentResults = filteredSegmentResults,
            segments = segments,
            totalSegments = filteredSegmentResults.size,
            validDigits = validDigits,
            allSegmentsOk = allSegmentsOk,
            pipeline = "VAD"
        )
    }

    private fun selectBestSubsequence(
        candidates: List<RawDigitClassifier.SegmentResult>,
        targetCount: Int
    ): List<RawDigitClassifier.SegmentResult> {
        if (candidates.size <= targetCount) return candidates

        val filtered = candidates.filter { segment ->
            val dur = segment.segment.durationMs
            val confOk = segment.confidence >= 0.70f
            val marginOk = segment.margin >= 0.40f
            dur >= 200 && (confOk || marginOk)
        }

        AppLog.d(TAG) { "Best-4-Window: ${candidates.size} candidates, ${filtered.size} after filter (dur>=200ms, conf>=70% OR margin>=40%)" }

        if (filtered.size <= targetCount) {
            AppLog.d(TAG) { "Best-4-Window: Using filtered ${filtered.size} segments (no window selection needed)" }
            return filtered.take(targetCount)
        }

        data class ScoredSegment(val segment: RawDigitClassifier.SegmentResult, val score: Float, val index: Int)

        val scored = filtered.mapIndexed { index, segment ->
            val score = segment.confidence + 0.5f * segment.margin
            ScoredSegment(segment, score, index)
        }

        var bestWindowStart = 0
        var bestWindowSum = 0f

        for (start in 0..(scored.size - targetCount)) {
            val windowSum = scored.subList(start, start + targetCount).sumOf { it.score.toDouble() }.toFloat()
            if (windowSum > bestWindowSum) {
                bestWindowSum = windowSum
                bestWindowStart = start
            }
        }

        val bestSegments = scored.subList(bestWindowStart, bestWindowStart + targetCount).map { it.segment }

        AppLog.d(TAG) { "Best-4-Window Selection:" }
        scored.forEachIndexed { idx, s ->
            val inWindow = idx >= bestWindowStart && idx < bestWindowStart + targetCount
            val mark = if (inWindow) "[OK]" else "[X]"
            AppLog.v(TAG) { "$mark [${s.index}] ${s.segment.label}: score=${String.format("%.2f", s.score)} (conf=${(s.segment.confidence*100).toInt()}%, margin=${(s.segment.margin*100).toInt()}%, dur=${s.segment.segment.durationMs}ms)" }
        }
        AppLog.d(TAG) { "Selected window: [$bestWindowStart..${bestWindowStart+targetCount-1}], sum=${String.format("%.2f", bestWindowSum)}" }

        return bestSegments
    }

    // ==========================================
    // RMS-basierte VAD (Legacy-Fallback)
    // ==========================================

    private fun detectSpeechSegments(
        pcm16: ShortArray,
        sampleRate: Int,
        vadLog: StringBuilder? = null
    ): List<RawDigitClassifier.SpeechSegment> {
        val frameSize = (VAD_FRAME_MS * sampleRate) / 1000
        val minSpeechFrames = VAD_MIN_SPEECH_MS / VAD_FRAME_MS
        val minSilenceFrames = VAD_MIN_SILENCE_MS / VAD_FRAME_MS

        AppLog.v(TAG) { "VAD: frameSize=$frameSize, minSpeech=$minSpeechFrames frames, minSilence=$minSilenceFrames frames" }
        vadLog?.append("VAD Config: frameSize=$frameSize, minSpeech=$minSpeechFrames, minSilence=$minSilenceFrames\n")
        vadLog?.append("Thresholds: start=$VAD_SPEECH_START_RMS, end=$VAD_SPEECH_END_RMS\n")

        val frameRms = mutableListOf<Float>()
        var offset = 0
        while (offset + frameSize <= pcm16.size) {
            val frame = pcm16.sliceArray(offset until offset + frameSize)
            val rms = kotlin.math.sqrt(frame.map { it.toDouble() * it.toDouble() }.average()).toFloat()
            frameRms.add(rms)
            offset += frameSize
        }

        AppLog.v(TAG) { "VAD: ${frameRms.size} frames, RMS range: ${frameRms.minOrNull()?.toInt()}..${frameRms.maxOrNull()?.toInt()}" }
        vadLog?.append("RMS range: ${frameRms.minOrNull()?.toInt()}..${frameRms.maxOrNull()?.toInt()}\n")

        val segments = mutableListOf<RawDigitClassifier.SpeechSegment>()
        var inSpeech = false
        var speechStartFrame = 0
        var silenceCounter = 0
        var peakRms = 0f

        for ((frameIdx, rms) in frameRms.withIndex()) {
            if (!inSpeech) {
                if (rms >= VAD_SPEECH_START_RMS) {
                    inSpeech = true
                    speechStartFrame = maxOf(0, frameIdx - VAD_PRE_ROLL_FRAMES)
                    silenceCounter = 0
                    peakRms = rms
                    AppLog.v(TAG) { "VAD: Speech START at frame $frameIdx (preroll to $speechStartFrame) (RMS=${"%.0f".format(rms)})" }
                    vadLog?.append("  START frame $frameIdx->$speechStartFrame (RMS=${rms.toInt()})\n")
                }
            } else {
                peakRms = maxOf(peakRms, rms)

                if (rms < VAD_SPEECH_END_RMS) {
                    silenceCounter++
                    if (silenceCounter >= minSilenceFrames) {
                        val rawEndFrame = frameIdx - silenceCounter
                        val endFrame = minOf(frameRms.size - 1, rawEndFrame + VAD_POST_ROLL_FRAMES)
                        val speechFrames = endFrame - speechStartFrame + 1

                        if (speechFrames >= minSpeechFrames) {
                            val startSample = speechStartFrame * frameSize
                            val endSample = minOf((endFrame + 1) * frameSize - 1, pcm16.size - 1)
                            val durationMs = (speechFrames * VAD_FRAME_MS)

                            segments.add(RawDigitClassifier.SpeechSegment(startSample, endSample, durationMs, peakRms))
                            AppLog.v(TAG) { "VAD: Speech END at frame $endFrame (raw=$rawEndFrame, ${speechFrames} frames, ${durationMs}ms)" }
                            vadLog?.append("  END frame $endFrame (${durationMs}ms, peak=${peakRms.toInt()})\n")
                        } else {
                            AppLog.v(TAG) { "VAD: Discarding short segment (${speechFrames} frames < $minSpeechFrames)" }
                            vadLog?.append("  DISCARD short (${speechFrames} < $minSpeechFrames)\n")
                        }

                        inSpeech = false
                        silenceCounter = 0
                        peakRms = 0f
                    }
                } else {
                    silenceCounter = 0
                }
            }
        }

        if (inSpeech) {
            val endFrame = frameRms.size - 1
            val speechFrames = endFrame - speechStartFrame + 1

            if (speechFrames >= minSpeechFrames) {
                val startSample = speechStartFrame * frameSize
                val endSample = minOf((endFrame + 1) * frameSize - 1, pcm16.size - 1)
                val durationMs = (speechFrames * VAD_FRAME_MS)

                segments.add(RawDigitClassifier.SpeechSegment(startSample, endSample, durationMs, peakRms))
                AppLog.v(TAG) { "VAD: Speech END at buffer end (${speechFrames} frames, ${durationMs}ms)" }
            }
        }

        return segments
    }
}
