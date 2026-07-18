package com.orez74.speechdigits

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class WebRtcVadSegmenterTest {

    @Test
    fun silenceFrames_rmsBelowStartGate() {
        val silenceFrame = ShortArray(160) // 10ms @ 16kHz, all zeros
        val rms = rmsOfFrame(silenceFrame)
        assertEquals(0.0f, rms)
        assertTrue(
            "Silence RMS $rms should be below start gate ${WebRtcVadSegmenter.RMS_GATE_MIN_START}",
            rms < WebRtcVadSegmenter.RMS_GATE_MIN_START
        )
    }

    @Test
    fun loudSignalFrames_rmsExceedsStartGate() {
        val loudFrame = ShortArray(160) { 5000 }
        val rms = rmsOfFrame(loudFrame)
        assertTrue(
            "Loud RMS $rms should exceed start gate ${WebRtcVadSegmenter.RMS_GATE_MIN_START}",
            rms > WebRtcVadSegmenter.RMS_GATE_MIN_START
        )
    }

    @Test
    fun moderateSignal_rmsBetweenStartAndKeepGate() {
        val moderate = ShortArray(160) { 159 } // RMS ≈ 159
        val rms = rmsOfFrame(moderate)
        assertTrue(
            "RMS $rms should be above start gate ${WebRtcVadSegmenter.RMS_GATE_MIN_START}",
            rms >= WebRtcVadSegmenter.RMS_GATE_MIN_START
        )
        assertTrue(
            "RMS $rms should be below keep gate (min ${WebRtcVadSegmenter.RMS_GATE_MIN_KEEP})",
            rms < WebRtcVadSegmenter.RMS_GATE_MIN_KEEP
        )
    }

    @Test
    fun veryLoudSignal_rmsExceedsKeepGate() {
        val veryLoud = ShortArray(160) { 500 }
        val rms = rmsOfFrame(veryLoud)
        assertTrue(
            "Very loud RMS $rms should exceed keep gate ${WebRtcVadSegmenter.RMS_GATE_MIN_KEEP}",
            rms > WebRtcVadSegmenter.RMS_GATE_MIN_KEEP
        )
    }

    @Test
    fun vadConstants_areConsistent() {
        assertTrue("Start gate must be ≤ keep gate",
            WebRtcVadSegmenter.RMS_GATE_MIN_START <= WebRtcVadSegmenter.RMS_GATE_MIN_KEEP)
        assertTrue("Start factor must be < keep factor",
            WebRtcVadSegmenter.RMS_GATE_FACTOR_START < WebRtcVadSegmenter.RMS_GATE_FACTOR_KEEP)
        assertTrue("Min speech frames must be positive",
            WebRtcVadSegmenter.MIN_SPEECH_FRAMES > 0)
        assertTrue("Min silence frames must be positive",
            WebRtcVadSegmenter.MIN_SILENCE_FRAMES > 0)
        assertEquals(16000, AudioPreprocessor.TARGET_SAMPLE_RATE)
    }

    @Test
    fun frameRmsOfSilenceVsLoud_differenceExceedsThreshold() {
        val silence = ShortArray(160) // all zeros
        val loud = ShortArray(160) { 1000 }

        val silenceRms = rmsOfFrame(silence)
        val loudRms = rmsOfFrame(loud)

        assertTrue("Loud RMS must exceed silence RMS", loudRms > silenceRms)
        assertTrue("Difference must exceed RMS gate min start",
            (loudRms - silenceRms) > WebRtcVadSegmenter.RMS_GATE_MIN_START)
    }

    private fun rmsOfFrame(frame: ShortArray): Float {
        var sumSquares = 0.0
        for (sample in frame) {
            sumSquares += sample.toDouble() * sample.toDouble()
        }
        return kotlin.math.sqrt(sumSquares / frame.size).toFloat()
    }
}
