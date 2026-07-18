package com.orez74.speechdigits

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AudioPreprocessorTest {

    @Test
    fun padOrTruncate_targetLength_exactMatch_returnsIdentical() {
        val input = ShortArray(16000) { it.toShort() }
        val result = AudioPreprocessor.padOrTruncate(input, 16000)
        assertEquals(16000, result.size)
        assertTrue(input.contentEquals(result))
    }

    @Test
    fun padOrTruncate_shorter_than_target_padsWithZeros() {
        val input = ShortArray(8000) { 42 }
        val result = AudioPreprocessor.padOrTruncate(input, 16000)
        assertEquals(16000, result.size)
        assertEquals(42.toShort(), result[0])
        assertEquals(42.toShort(), result[7999])
        assertEquals(0.toShort(), result[8000])
        assertEquals(0.toShort(), result[15999])
    }

    @Test
    fun padOrTruncate_longer_than_target_truncates() {
        val input = ShortArray(32000) { 7 }
        val result = AudioPreprocessor.padOrTruncate(input, 16000)
        assertEquals(16000, result.size)
        assertEquals(7.toShort(), result.first())
        assertEquals(7.toShort(), result.last())
    }

    @Test
    fun normalize_pcm16ToFloat_rangeIsWithinMinusOneToOne() {
        val pcm16 = ShortArray(1024) { (it % 32767).toShort() }
        val result = AudioPreprocessor.normalize(pcm16)
        assertEquals(1024, result.size)
        for (v in result) {
            assertTrue("Value $v out of range [-1,1]", v in -1.0f..1.0f)
        }
    }

    @Test
    fun normalize_zeroInput_yieldsAllZeros() {
        val pcm16 = ShortArray(512) // all zeros
        val result = AudioPreprocessor.normalize(pcm16)
        assertEquals(512, result.size)
        result.forEach { assertEquals(0.0f, it) }
    }

    @Test
    fun calculateRms_silence_isBelowTooQuietThreshold() {
        val silence = ShortArray(16000) // all zeros
        val rms = AudioPreprocessor.calculateRms(silence)
        assertEquals(0.0f, rms)
        assertTrue(rms < AudioPreprocessor.RMS_TOO_QUIET_THRESHOLD)
    }

    @Test
    fun calculateRms_loudSignal_exceedsTooQuietThreshold() {
        val loud = ShortArray(16000) { 10000 }
        val rms = AudioPreprocessor.calculateRms(loud)
        assertTrue("RMS $rms should exceed threshold ${AudioPreprocessor.RMS_TOO_QUIET_THRESHOLD}",
            rms > AudioPreprocessor.RMS_TOO_QUIET_THRESHOLD)
    }

    @Test
    fun calculatePeak_returnsMaxAbsoluteValue() {
        val samples = shortArrayOf(100, -500, 300, -1000, 200)
        val peak = AudioPreprocessor.calculatePeak(samples)
        assertEquals(1000, peak)
    }

    @Test
    fun resampleLinear_identityRate_returnsSameArray() {
        val input = ShortArray(8000) { it.toShort() }
        val result = AudioPreprocessor.resampleLinear(input, 16000, 16000)
        assertTrue(input.contentEquals(result))
    }
}
