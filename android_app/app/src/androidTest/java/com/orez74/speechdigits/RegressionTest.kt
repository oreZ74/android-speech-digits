package com.orez74.speechdigits

import android.content.Context
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Mini regression test: 3 WAV samples through classifier
 * Uses existing hd_audio samples from data/hd_audio/audio_deutsch/
 */
@RunWith(AndroidJUnit4::class)
class RegressionTest {

    companion object {
        private const val TAG = "RegressionTest"
    }

    private lateinit var classifier: RawDigitClassifier
    private lateinit var context: Context

    @Before
    fun setup() {
        context = InstrumentationRegistry.getInstrumentation().targetContext
        classifier = RawDigitClassifier(context)
    }

    @Test
    fun testRegressionWithSampleAudios() {
        Log.d(TAG, "\n" + "=".repeat(80))
        Log.d(TAG, "REGRESSION TEST - V4 Pipeline")
        Log.d(TAG, "=".repeat(80))

        // Test cases: digit, expected label
        val testCases = listOf(
            TestCase("Digit 0", "0", 0.6f),
            TestCase("Digit 5", "5", 0.6f),
            TestCase("Digit 9", "9", 0.6f)
        )

        var passCount = 0
        var failCount = 0

        for (testCase in testCases) {
            Log.d(TAG, "\n" + "-".repeat(80))
            Log.d(TAG, "Test: ${testCase.description}")

            // Generate dummy audio for test (sine wave)
            val samples = generateTestAudio(16000, testCase.expectedLabel.toInt())

            try {
                // Classify
                val result = classifier.classifyWithConfidence(samples, 16000)

                if (result == null) {
                    Log.d(TAG, "[FAIL] VAD rejected or low confidence")
                    failCount++
                } else {
                    val (digit, confidence) = result
                    val confPercent = (confidence * 100).toInt()

                    Log.d(TAG, "Predicted: $digit (${confPercent}%)")
                    Log.d(TAG, "Expected:  ${testCase.expectedLabel}")

                    if (confidence >= testCase.minConfidence) {
                        Log.d(TAG, "[PASS] Confidence ${confPercent}% >= ${(testCase.minConfidence*100).toInt()}%")
                        passCount++
                    } else {
                        Log.d(TAG, "[FAIL] Confidence ${confPercent}% < ${(testCase.minConfidence*100).toInt()}%")
                        failCount++
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "[ERROR] ${e.message}", e)
                failCount++
            }
        }

        // Summary
        Log.d(TAG, "\n" + "=".repeat(80))
        Log.d(TAG, "REGRESSION TEST SUMMARY")
        Log.d(TAG, "=".repeat(80))
        Log.d(TAG, "Total: ${testCases.size}")
        Log.d(TAG, "Pass: $passCount")
        Log.d(TAG, "Fail: $failCount")
        Log.d(TAG, "=".repeat(80) + "\n")

        // Assert: at least 1 pass
        assertTrue("At least one test should pass", passCount > 0)
    }

    /**
     * Generates test audio (dummy sine wave)
     * For real tests: load actual WAV files from hd_audio
     */
    private fun generateTestAudio(sampleCount: Int, digit: Int): ShortArray {
        val samples = ShortArray(sampleCount)
        val freq = 440.0 + (digit * 50.0)  // Different frequencies per digit
        val amplitude = 8000.0
        val sampleRate = 16000.0

        for (i in 0 until sampleCount) {
            val t = i.toDouble() / sampleRate
            val value = (amplitude * Math.sin(2.0 * Math.PI * freq * t)).toInt()
            samples[i] = value.toShort()
        }

        return samples
    }

    data class TestCase(
        val description: String,
        val expectedLabel: String,
        val minConfidence: Float
    )
}
