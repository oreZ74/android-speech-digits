package com.orez74.speechdigits

import com.konovalov.vad.webrtc.VadWebRTC
import com.konovalov.vad.webrtc.config.FrameSize
import com.konovalov.vad.webrtc.config.Mode
import com.konovalov.vad.webrtc.config.SampleRate

/**
 * WebRTC-based Voice Activity Detection for speech segmentation.
 *
 * Uses the native WebRTC VAD algorithm which is more robust than simple RMS thresholds,
 * especially for detecting speech endpoints (e.g., final consonants like /t/ in "acht").
 *
 * Configuration:
 * - Frame size: 10ms (160 samples @ 16kHz) - higher resolution for digit pauses
 * - Mode: VERY_AGGRESSIVE (3) for better silence detection
 * - RMS-Gating: additional energy threshold to prevent false positives
 */
object WebRtcVadSegmenter {

    private const val TAG = "WebRtcVadSegmenter"

    // ==========================================
    // WEBRTC VAD PARAMETERS
    // ==========================================

    /** VAD aggressiveness mode (0=least aggressive, 3=most aggressive) */
    var vadMode: Int = 3  // VERY_AGGRESSIVE for better silence detection

    /** Minimum consecutive speech frames to start a segment */
    const val MIN_SPEECH_FRAMES = 3      // 30ms (at 10ms frames) - fast start

    /** Minimum consecutive silence frames to end a segment */
    const val MIN_SILENCE_FRAMES = 8     // 80ms - clearly separated digits

    /** Frames to add before detected speech start */
    // PHASE 1: Increased from 6 (60ms) to 12 (120ms) for fricative onsets
    const val PRE_ROLL_FRAMES = 12       // 120ms - for onsets like 'Z' in 'zwei', 'D' in 'drei', 'S' in 'sechs'

    /** Frames to add after detected speech end */
    const val POST_ROLL_FRAMES = 7       // 70ms - for codas like /t/ in 'acht'

    /** Maximum gap between segments to merge them */
    const val MERGE_GAP_FRAMES = 0       // No merge - each digit stays separated

    /** Minimum segment duration in frames */
    const val MIN_SEGMENT_FRAMES = 15    // 150ms (at 10ms frames)

    /** Frame size in ms */
    const val FRAME_MS = 10              // 10ms for higher resolution

    // ==========================================
    // 2-LEVEL RMS GATE PARAMETERS (Hysterese)
    // ==========================================

    /** Start gate: lower threshold so that quiet onsets are detected */
    const val RMS_GATE_MIN_START = 120f
    const val RMS_GATE_FACTOR_START = 1.15f

    /** Keep gate: slightly higher, prevents noise from slipping through during speech */
    const val RMS_GATE_MIN_KEEP = 160f
    const val RMS_GATE_FACTOR_KEEP = 1.30f

    /** Grace frames: when VAD=true but RMS slightly below keep gate, still count as speech */
    const val GRACE_FRAMES_BELOW_KEEP = 2

    // ==========================================
    // NOISE FLOOR ESTIMATION
    // ==========================================

    /** Frames for noise floor estimation (first N frames or max available) */
    const val NOISE_ESTIMATION_FRAMES = 100  // 1s at 10ms frames

    /** Percentile of lowest RMS values for noise floor (30% = robust against early speech) */
    const val NOISE_FLOOR_PERCENTILE = 0.30f

    // ==========================================
    // FALLBACK VAD MODE
    // ==========================================

    /** Enable automatic fallback to less aggressive VAD mode */
    const val ENABLE_VAD_FALLBACK = true

    /** Fallback VAD mode (AGGRESSIVE=2) when VERY_AGGRESSIVE finds nothing */
    const val FALLBACK_VAD_MODE = 2

    /** Minimum maxRms to consider fallback (if maxRms is below this, audio is truly silent) */
    const val FALLBACK_MIN_MAX_RMS = 400f

    // ==========================================
    // BACKTRACKING
    // ==========================================

    /** Maximum frames to backtrack for onset refinement */
    const val BACKTRACK_FRAMES = 15 // 150ms - for fricative onsets like 'Z', 'D', 'S'

    // ==========================================
    // ONSET TRIMMING (Post-VAD cleanup)
    // ==========================================

    /** Enable onset trimming to remove leading silence/noise from segments */
    const val ENABLE_ONSET_TRIM = true

    /** Maximum frames to scan for onset (from segment start) */
    const val ONSET_TRIM_MAX_FRAMES = 12 // 120ms max scan

    /** RMS threshold multiplier for onset detection (relative to segment peak) */
    const val ONSET_TRIM_THRESHOLD_FACTOR = 0.08f // 8% of peak RMS

    /** Minimum RMS for onset (absolute floor) */
    const val ONSET_TRIM_MIN_RMS = 200f

    private const val TAG_REFINE = "VAD_REFINE"
    private const val TAG_TRIM = "VAD_TRIM"
    private const val TAG_GATE = "VAD_GATE"
    private const val TAG_MODE = "VAD_MODE"

    /**
     * Backtrack from confirmed speech start to find the true onset.
     * Searches backwards for the last frame below energy threshold.
     *
     * @param confirmedStart Frame index where speech was confirmed
     * @param frameRms RMS values for all frames
     * @param noiseFloorRms Estimated noise floor RMS
     * @param vadLog Optional StringBuilder for logging
     * @return Refined start frame (always <= confirmedStart)
     */
    private fun refineStartWithBacktrack(
        confirmedStart: Int,
        frameRms: FloatArray,
        noiseFloorRms: Float,
        vadLog: StringBuilder?
    ): Int {
        // Energy threshold: low enough to catch fricatives but above noise
        // Use 60% of Start-Gate as baseline for backtracking
        val energyThresh = maxOf(RMS_GATE_MIN_START * 0.6f, noiseFloorRms * 1.05f)

        // Define backtrack range
        val backtrackStart = maxOf(0, confirmedStart - BACKTRACK_FRAMES)
        val backtrackRange = backtrackStart until confirmedStart

        // Find last frame below threshold (= silence before onset)
        var lastSilenceFrame = -1
        for (f in backtrackRange) {
            if (frameRms[f] < energyThresh) {
                lastSilenceFrame = f
            }
        }

        // Refined start is one frame after last silence, or confirmedStart if no silence found
        val refinedStart = if (lastSilenceFrame >= 0) {
            (lastSilenceFrame + 1).coerceAtMost(confirmedStart)
        } else {
            confirmedStart
        }

        // Collect RMS values in backtrack range for logging (rounded to int)
        val backtrackRmsValues = backtrackRange.map { frameRms[it].toInt() }

        // Detailed logging
        val logMsg = buildString {
            append("confirmedStart=$confirmedStart, ")
            append("refinedStart=$refinedStart, ")
            append("delta=${confirmedStart - refinedStart}, ")
            append("noiseFloor=${noiseFloorRms.toInt()}, ")
            append("energyThresh=${energyThresh.toInt()}, ")
            append("backtrackFrames=$BACKTRACK_FRAMES, ")
            append("rmsInRange=$backtrackRmsValues")
        }
        AppLog.d(TAG_REFINE) { logMsg }
        vadLog?.append("REFINE: $logMsg\n")

        return refinedStart
    }

    /**
     * Detect speech segments using WebRTC VAD.
     * Includes automatic fallback to less aggressive mode if no speech found.
     *
     * @param pcm16 Raw PCM16 audio samples
     * @param sampleRate Sample rate (must be 16000)
     * @param vadLog Optional StringBuilder for logging
     * @return List of detected speech segments
     */
    fun detectSpeechSegments(
        pcm16: ShortArray,
        sampleRate: Int,
        vadLog: StringBuilder? = null
    ): List<RawDigitClassifier.SpeechSegment> {
        require(sampleRate == 16000) { "WebRTC VAD requires 16kHz, got $sampleRate" }

        val frameSize = (FRAME_MS * sampleRate) / 1000  // 160 samples at 10ms
        val totalFrames = pcm16.size / frameSize

        // ===== FIRST PASS: Calculate RMS for all frames =====
        val frameRms = FloatArray(totalFrames)
        for (frameIdx in 0 until totalFrames) {
            val startSample = frameIdx * frameSize
            val frame = pcm16.sliceArray(startSample until startSample + frameSize)
            var sumSquares = 0.0
            for (sample in frame) {
                sumSquares += sample.toDouble() * sample.toDouble()
            }
            frameRms[frameIdx] = kotlin.math.sqrt(sumSquares / frameSize).toFloat()
        }

        // ===== ROBUST NOISE FLOOR ESTIMATION =====
        // Use only first N frames (or all if shorter), take lowest NOISE_FLOOR_PERCENTILE
        val noiseEstFrames = minOf(totalFrames, NOISE_ESTIMATION_FRAMES)
        val noiseEstWindow = frameRms.take(noiseEstFrames).sorted()
        val noiseFloorIdx = (noiseEstWindow.size * NOISE_FLOOR_PERCENTILE).toInt().coerceIn(0, noiseEstWindow.lastIndex)
        val noiseFloorRms = noiseEstWindow[noiseFloorIdx]

        // ===== RMS STATISTICS FOR LOGGING =====
        val sortedAll = frameRms.sorted()
        val minRms = sortedAll.first()
        val p10 = sortedAll[(sortedAll.size * 0.10f).toInt().coerceIn(0, sortedAll.lastIndex)]
        val p20 = sortedAll[(sortedAll.size * 0.20f).toInt().coerceIn(0, sortedAll.lastIndex)]
        val p50 = sortedAll[(sortedAll.size * 0.50f).toInt().coerceIn(0, sortedAll.lastIndex)]
        val p90 = sortedAll[(sortedAll.size * 0.90f).toInt().coerceIn(0, sortedAll.lastIndex)]
        val maxRms = sortedAll.last()

        // ===== 2-LEVEL RMS GATES =====
        val rmsGateStart = maxOf(RMS_GATE_MIN_START, noiseFloorRms * RMS_GATE_FACTOR_START)
        val rmsGateKeep = maxOf(RMS_GATE_MIN_KEEP, noiseFloorRms * RMS_GATE_FACTOR_KEEP)

        // Detailed gate logging
        AppLog.d(TAG_GATE) { "noiseFloor=${noiseFloorRms.toInt()}, gateStart=${rmsGateStart.toInt()}, gateKeep=${rmsGateKeep.toInt()}, rms[p10,p20,p50,p90,max]=[${p10.toInt()},${p20.toInt()},${p50.toInt()},${p90.toInt()},${maxRms.toInt()}]" }
        vadLog?.append("VAD_GATE: noiseFloor=${noiseFloorRms.toInt()}, gateStart=${rmsGateStart.toInt()}, gateKeep=${rmsGateKeep.toInt()}, rms[p10,p20,p50,p90,max]=[${p10.toInt()},${p20.toInt()},${p50.toInt()},${p90.toInt()},${maxRms.toInt()}]\n")

        AppLog.d(TAG) { "WebRTC VAD: mode=$vadMode, frames=$totalFrames, frameSize=$frameSize, frameMs=$FRAME_MS" }
        vadLog?.append("WebRTC VAD Config: mode=$vadMode, minSpeech=$MIN_SPEECH_FRAMES, minSilence=$MIN_SILENCE_FRAMES\n")
        vadLog?.append("PreRoll=$PRE_ROLL_FRAMES, PostRoll=$POST_ROLL_FRAMES, MergeGap=$MERGE_GAP_FRAMES, FrameMs=$FRAME_MS\n")

        // ===== PRIMARY VAD RUN =====
        val primaryResult = runVadWithMode(
            pcm16 = pcm16,
            frameRms = frameRms,
            noiseFloorRms = noiseFloorRms,
            rmsGateStart = rmsGateStart,
            rmsGateKeep = rmsGateKeep,
            currentVadMode = vadMode,
            frameSize = frameSize,
            totalFrames = totalFrames,
            vadLog = vadLog
        )

        // ===== FALLBACK LOGIC =====
        if (ENABLE_VAD_FALLBACK && primaryResult.isEmpty() && maxRms > FALLBACK_MIN_MAX_RMS) {
            // Audio has energy but no segments found -> try less aggressive mode
            AppLog.d(TAG_MODE) { "primary=$vadMode segments=0 maxRms=${maxRms.toInt()} -> fallback=$FALLBACK_VAD_MODE reason=maxRms high but no speech" }
            vadLog?.append("VAD_MODE: primary=$vadMode segments=0 -> fallback=$FALLBACK_VAD_MODE reason=maxRms=${maxRms.toInt()} high but no speech\n")

            val fallbackResult = runVadWithMode(
                pcm16 = pcm16,
                frameRms = frameRms,
                noiseFloorRms = noiseFloorRms,
                rmsGateStart = rmsGateStart,
                rmsGateKeep = rmsGateKeep,
                currentVadMode = FALLBACK_VAD_MODE,
                frameSize = frameSize,
                totalFrames = totalFrames,
                vadLog = vadLog
            )

            // Accept fallback if plausible (1-10 segments, reasonable duration)
            if (fallbackResult.isNotEmpty() && fallbackResult.size <= 10) {
                AppLog.d(TAG_MODE) { "fallback=$FALLBACK_VAD_MODE accepted, segments=${fallbackResult.size}" }
                vadLog?.append("VAD_MODE: fallback accepted, segments=${fallbackResult.size}\n")
                return fallbackResult
            } else {
                AppLog.d(TAG_MODE) { "fallback=$FALLBACK_VAD_MODE rejected, segments=${fallbackResult.size}" }
                vadLog?.append("VAD_MODE: fallback rejected, segments=${fallbackResult.size}\n")
            }
        }

        return primaryResult
    }

    /**
     * Run VAD with a specific mode and 2-level RMS gating.
     */
    private fun runVadWithMode(
        pcm16: ShortArray,
        frameRms: FloatArray,
        noiseFloorRms: Float,
        rmsGateStart: Float,
        rmsGateKeep: Float,
        currentVadMode: Int,
        frameSize: Int,
        totalFrames: Int,
        vadLog: StringBuilder?
    ): List<RawDigitClassifier.SpeechSegment> {

        val mode = when (currentVadMode) {
            0 -> Mode.NORMAL
            1 -> Mode.LOW_BITRATE
            2 -> Mode.AGGRESSIVE
            else -> Mode.VERY_AGGRESSIVE
        }

        // Get raw VAD decisions first (without RMS gating)
        val vadDecisions = BooleanArray(totalFrames)
        VadWebRTC(
            sampleRate = SampleRate.SAMPLE_RATE_16K,
            frameSize = FrameSize.FRAME_SIZE_160,
            mode = mode
        ).use { vad ->
            for (frameIdx in 0 until totalFrames) {
                val startSample = frameIdx * frameSize
                val frame = pcm16.sliceArray(startSample until startSample + frameSize)
                vadDecisions[frameIdx] = vad.isSpeech(frame)
            }
        }

        // Count VAD speech frames before RMS gating
        val vadOnlyCount = vadDecisions.count { it }

        // ===== STATE MACHINE WITH 2-LEVEL RMS GATING =====
        val rawSegments = mutableListOf<IntRange>()
        var inSpeech = false
        var speechStartFrame = 0
        var speechFrameCount = 0
        var silenceFrameCount = 0
        var graceFrameCount = 0  // Frames where VAD=true but RMS below keep gate

        for (frameIdx in 0 until totalFrames) {
            val vadDecision = vadDecisions[frameIdx]
            val rms = frameRms[frameIdx]

            // 2-Level RMS gating based on current state
            val isSpeech = if (!inSpeech) {
                // Looking for start: use lower gate
                vadDecision && rms > rmsGateStart
            } else {
                // In speech: use higher gate with grace window
                if (vadDecision && rms > rmsGateKeep) {
                    graceFrameCount = 0
                    true
                } else if (vadDecision && rms > rmsGateStart && graceFrameCount < GRACE_FRAMES_BELOW_KEEP) {
                    // VAD says speech, RMS between start and keep gate -> grace period
                    graceFrameCount++
                    true
                } else {
                    graceFrameCount = 0
                    false
                }
            }

            if (!inSpeech) {
                // Looking for speech start
                if (isSpeech) {
                    speechFrameCount++
                    if (speechFrameCount >= MIN_SPEECH_FRAMES) {
                        inSpeech = true
                        graceFrameCount = 0
                        // Start from first speech frame
                        val firstSpeechFrame = frameIdx - speechFrameCount + 1

                        // ===== START BACKTRACKING =====
                        val refinedStart = refineStartWithBacktrack(
                            confirmedStart = firstSpeechFrame,
                            frameRms = frameRms,
                            noiseFloorRms = noiseFloorRms,
                            vadLog = vadLog
                        )
                        speechStartFrame = maxOf(0, refinedStart - PRE_ROLL_FRAMES)
                        // ===== END BACKTRACKING =====

                        silenceFrameCount = 0

                        // Detailed START logging with ms values
                        val confirmMs = frameIdx * FRAME_MS
                        val firstSpeechMs = firstSpeechFrame * FRAME_MS
                        val refinedMs = refinedStart * FRAME_MS
                        val finalStartMs = speechStartFrame * FRAME_MS
                        val deltaBacktrack = firstSpeechFrame - refinedStart
                        val deltaTotalMs = confirmMs - finalStartMs

                        AppLog.d(TAG) { "WebRTC: Speech START - confirmFrame=$frameIdx (${confirmMs}ms), firstSpeech=$firstSpeechFrame (${firstSpeechMs}ms), refined=$refinedStart (${refinedMs}ms), preroll=$PRE_ROLL_FRAMES -> finalStart=$speechStartFrame (${finalStartMs}ms) [backtrackDelta=$deltaBacktrack, totalDelta=${deltaTotalMs}ms] (mode=$currentVadMode)" }
                        vadLog?.append("  START confirm=$frameIdx (${confirmMs}ms) firstSpeech=$firstSpeechFrame refined=$refinedStart preroll=$PRE_ROLL_FRAMES -> frame $speechStartFrame (${finalStartMs}ms) [delta=${deltaTotalMs}ms]\n")
                    }
                } else {
                    speechFrameCount = 0
                }
            } else {
                // In speech, looking for end
                if (!isSpeech) {
                    silenceFrameCount++
                    if (silenceFrameCount >= MIN_SILENCE_FRAMES) {
                        val endFrame = minOf(totalFrames - 1, frameIdx + POST_ROLL_FRAMES)
                        val durationFrames = endFrame - speechStartFrame + 1
                        val durationMs = durationFrames * FRAME_MS

                        if (durationFrames >= MIN_SEGMENT_FRAMES) {
                            rawSegments.add(speechStartFrame..endFrame)
                            AppLog.d(TAG) { "WebRTC: Segment END - start=$speechStartFrame end=$endFrame dur=${durationMs}ms" }
                            vadLog?.append("  END frame $endFrame (dur=${durationMs}ms)\n")
                        } else {
                            AppLog.v(TAG) { "WebRTC: Discarding short segment (${durationMs}ms)" }
                            vadLog?.append("  DISCARD short (${durationMs}ms)\n")
                        }

                        inSpeech = false
                        speechFrameCount = 0
                        silenceFrameCount = 0
                        graceFrameCount = 0
                    }
                } else {
                    silenceFrameCount = 0
                }
            }
        }

        // Handle segment at end of buffer
        if (inSpeech) {
            val endFrame = totalFrames - 1
            if (endFrame - speechStartFrame + 1 >= MIN_SEGMENT_FRAMES) {
                rawSegments.add(speechStartFrame..endFrame)
                AppLog.v(TAG) { "WebRTC: Speech END at buffer end (frame $endFrame)" }
                vadLog?.append("  END at buffer end (frame $endFrame)\n")
            }
        }

        // Merge nearby segments
        val mergedSegments = mergeSegments(rawSegments, vadLog)

        // Convert to SpeechSegment with sample positions and peak RMS
        val result = mergedSegments.map { range ->
            val startSample = range.first * frameSize
            val endSample = minOf((range.last + 1) * frameSize - 1, pcm16.size - 1)
            val durationMs = (range.last - range.first + 1) * FRAME_MS

            // Calculate peak RMS within segment
            var peakRms = 0f
            for (f in range) {
                if (f < frameRms.size && frameRms[f] > peakRms) {
                    peakRms = frameRms[f]
                }
            }

            RawDigitClassifier.SpeechSegment(startSample, endSample, durationMs, peakRms)
        }

        // ===== ONSET TRIMMING =====
        val trimmedResult = if (ENABLE_ONSET_TRIM) {
            result.map { seg ->
                trimOnset(seg, pcm16, frameSize, vadLog)
            }
        } else {
            result
        }

        // Log speech frame statistics
        val speechCountAfterGating = rawSegments.sumOf { it.last - it.first + 1 }
        AppLog.d(TAG) { "WebRTC VAD (mode=$currentVadMode): vadOnly=$vadOnlyCount, afterGating=$speechCountAfterGating, ${rawSegments.size} raw -> ${trimmedResult.size} merged segments" }
        vadLog?.append("Result (mode=$currentVadMode): vadOnly=$vadOnlyCount, ${rawSegments.size} raw -> ${trimmedResult.size} merged\n")

        // Log each segment
        trimmedResult.forEachIndexed { idx, seg ->
            AppLog.d(TAG) { "  Segment $idx: ${seg.startSample}..${seg.endSample} (${seg.durationMs}ms, peakRms=${seg.peakRms.toInt()})" }
        }

        return trimmedResult
    }

    /**
     * Trim leading silence/noise from a segment.
     * Scans first N frames and skips until RMS exceeds threshold.
     * This fixes issues where pre-roll captures noise before actual speech.
     */
    private fun trimOnset(
        segment: RawDigitClassifier.SpeechSegment,
        pcm16: ShortArray,
        frameSize: Int,
        vadLog: StringBuilder?
    ): RawDigitClassifier.SpeechSegment {
        // Calculate threshold: 8% of peak RMS or minimum floor
        val threshold = maxOf(ONSET_TRIM_MIN_RMS, segment.peakRms * ONSET_TRIM_THRESHOLD_FACTOR)

        // Scan frames from start
        val maxScanSamples = ONSET_TRIM_MAX_FRAMES * frameSize
        val scanEnd = minOf(segment.startSample + maxScanSamples, segment.endSample)

        var trimmedStart = segment.startSample
        val rmsLog = mutableListOf<Int>()

        // Scan in 10ms frames (160 samples at 16kHz)
        var samplePos = segment.startSample
        while (samplePos + frameSize <= scanEnd) {
            // Calculate RMS for this frame
            var sumSquares = 0.0
            for (i in 0 until frameSize) {
                val sample = pcm16[samplePos + i].toDouble()
                sumSquares += sample * sample
            }
            val frameRms = kotlin.math.sqrt(sumSquares / frameSize).toFloat()
            rmsLog.add(frameRms.toInt())

            if (frameRms >= threshold) {
                // Found onset - use this position
                trimmedStart = samplePos
                break
            }

            // Move to next frame
            samplePos += frameSize
        }

        // Don't trim if no significant energy found (keep original)
        if (samplePos >= scanEnd && trimmedStart == segment.startSample) {
            // No frame exceeded threshold - don't trim
            AppLog.d(TAG_TRIM) { "noTrim: seg=${segment.startSample}..${segment.endSample}, thresh=${threshold.toInt()}, scanned=${rmsLog.size} frames, rms=$rmsLog" }
            vadLog?.append("TRIM: noTrim (no frame >= ${threshold.toInt()} in first ${rmsLog.size} frames)\n")
            return segment
        }

        // Calculate how much we trimmed
        val trimmedSamples = trimmedStart - segment.startSample
        val trimmedMs = (trimmedSamples * 1000) / 16000

        if (trimmedSamples > 0) {
            val newDurationMs = ((segment.endSample - trimmedStart + 1) * 1000) / 16000
            AppLog.d(TAG_TRIM) { "trimmed: ${trimmedSamples} samples (${trimmedMs}ms), old=${segment.startSample}..${segment.endSample}, new=$trimmedStart..${segment.endSample}, thresh=${threshold.toInt()}, rms=$rmsLog" }
            vadLog?.append("TRIM: -${trimmedMs}ms (${trimmedSamples} samples), new start=$trimmedStart, thresh=${threshold.toInt()}\n")

            return RawDigitClassifier.SpeechSegment(
                startSample = trimmedStart,
                endSample = segment.endSample,
                durationMs = newDurationMs,
                peakRms = segment.peakRms
            )
        }

        return segment
    }

    /**
     * Merge segments that are close together (within MERGE_GAP_FRAMES).
     */
    private fun mergeSegments(
        segments: List<IntRange>,
        vadLog: StringBuilder?
    ): List<IntRange> {
        if (segments.size <= 1) return segments

        val merged = mutableListOf<IntRange>()
        var current = segments[0]

        for (i in 1 until segments.size) {
            val next = segments[i]
            val gap = next.first - current.last - 1

            if (gap <= MERGE_GAP_FRAMES) {
                // Merge
                current = current.first..next.last
                AppLog.v(TAG) { "WebRTC: MERGED segments (gap=$gap frames)" }
                vadLog?.append("  MERGE gap=$gap\n")
            } else {
                merged.add(current)
                current = next
            }
        }
        merged.add(current)

        return merged
    }
}
