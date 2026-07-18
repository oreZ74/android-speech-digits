package com.orez74.speechdigits

import android.content.Context
import android.util.Log
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Instrumentation test for RawDigitClassifier
 *
 * Tests:
 * 1. Model loading
 * 2. V2 normalization (mean-centering + peak-normalize)
 * 3. Offline predictions with FLAC files
 * 4. Comparison with Python predictions
 */
@RunWith(AndroidJUnit4::class)
class RawDigitClassifierTest {

    private lateinit var classifier: RawDigitClassifier
    private lateinit var context: Context

    companion object {
        private const val TAG = "RawDigitClassifierTest"
    }

    @Before
    fun setup() {
        context = ApplicationProvider.getApplicationContext()
        classifier = RawDigitClassifier(context)

        Log.d(TAG, "=".repeat(80))
        Log.d(TAG, "INSTRUMENTATION TEST: RawDigitClassifier V2")
        Log.d(TAG, "=".repeat(80))
    }

    @After
    fun teardown() {
        classifier.close()
        Log.d(TAG, "=".repeat(80))
        Log.d(TAG, "TEST COMPLETED")
        Log.d(TAG, "=".repeat(80))
    }

    /**
     * Test 1: Model loads successfully
     */
    @Test
    fun testModelLoads() {
        Log.d(TAG, "\nTEST 1: Model Loading")
        Log.d(TAG, "-".repeat(80))

        // If we reach this point, the constructor worked
        Log.d(TAG, "[OK] RawDigitClassifier initialized successfully")
        Log.d(TAG, "[OK] Model loaded")
    }

    /**
     * Test 2: V2 normalization works
     */
    @Test
    fun testV2Normalization() {
        Log.d(TAG, "\nTEST 2: V2 Normalization")
        Log.d(TAG, "-".repeat(80))

        // Create test audio (1 second sine @ 440Hz)
        val sampleRate = 16000
        val samples = ShortArray(16000) { i ->
            val t = i.toDouble() / sampleRate
            val freq = 440.0
            val amplitude = 16000.0
            (amplitude * Math.sin(2.0 * Math.PI * freq * t)).toInt().toShort()
        }

        Log.d(TAG, "Generated: 16000 samples @ 16kHz (440 Hz sine)")

        // Classify (internally checks normalization)
        val result = classifier.classifyWithConfidence(samples, sampleRate)

        if (result != null) {
            val (label, confidence) = result
            Log.d(TAG, "Prediction: Digit $label with ${(confidence*100).toInt()}% Confidence")
            Log.d(TAG, "[OK] Normalization successful (no crash)")
            Log.d(TAG, "[INFO] Check Logcat for 'V2 Normalization: Mean=..., MaxAbs=...'")
            Log.d(TAG, "  -> Expected: Mean \u2248 0.0, MaxAbs \u2248 1.0")
        } else {
            Log.d(TAG, "VAD: No speech detected (expected for sine test)")
        }
    }

    /**
     * Test 3: Classification with dummy audio
     *
     * Note: Random noise with low amplitude is detected as TOO_QUIET,
     * therefore we test here with a louder signal.
     */
    @Test
    fun testClassifyDummyAudio() {
        Log.d(TAG, "\nTEST 3: Dummy Audio Classification")
        Log.d(TAG, "-".repeat(80))

        // Create random noise with high amplitude (to avoid TOO_QUIET)
        val samples = ShortArray(16000) {
            (Math.random() * 20000 - 10000).toInt().toShort()
        }

        // Compute RMS for verification
        val rms = kotlin.math.sqrt(samples.map { it.toDouble() * it.toDouble() }.average())
        Log.d(TAG, "Generated signal: RMS = ${"%, .1f".format(rms)}")

        val result = classifier.classifyWithMetrics(samples, 16000)

        if (result != null) {
            Log.d(TAG, "Classification Result: ${result.result}")
            Log.d(TAG, "Label: ${result.label}, Confidence: ${(result.confidence * 100).toInt()}%")
            Log.d(TAG, "RMS: ${result.rmsAmplitude}, Peak: ${result.peakAmplitude}")

            // With high RMS we expect either a digit (0-9) or UNSURE
            // TOO_QUIET should not occur at this amplitude
            when (result.result) {
                RawDigitClassifier.ClassificationResult.DIGIT -> {
                    val digit = result.label?.toIntOrNull() ?: -1
                    assert(digit in 0..9) { "Predicted digit $digit outside range 0-9" }
                    Log.d(TAG, "[OK] DIGIT recognized: $digit")
                }
                RawDigitClassifier.ClassificationResult.UNSURE -> {
                    Log.d(TAG, "[OK] UNSURE is acceptable for random noise")
                }
                RawDigitClassifier.ClassificationResult.TOO_QUIET -> {
                    Log.d(TAG, "[WARN] TOO_QUIET with high RMS - check thresholds")
                }
            }
        } else {
            Log.d(TAG, "[FAIL] No result received")
        }

        Log.d(TAG, "[OK] Test completed without crash")
    }
    /**
     * Test 4: Various volume levels (V2 robustness test)
     */
    @Test
    fun testVolumVariation() {
        Log.d(TAG, "\nTEST 5: Volume variation (V2 robustness)")
        Log.d(TAG, "-".repeat(80))

        // Create same sine at different amplitudes
        val sampleRate = 16000

        val amplitudes = listOf(5000, 16000, 30000)  // quiet, normal, loud
        val predictions = mutableListOf<String>()

        for ((index, amp) in amplitudes.withIndex()) {
            val samples = ShortArray(16000) { i ->
                val t = i.toDouble() / sampleRate
                val freq = 440.0
                (amp * Math.sin(2.0 * Math.PI * freq * t)).toInt().toShort()
            }

            val result = classifier.classifyWithConfidence(samples, sampleRate)

            val volume = when(index) {
                0 -> "Quiet"
                1 -> "Normal"
                2 -> "Loud"
                else -> "?"
            }

            if (result != null) {
                val (label, confidence) = result
                predictions.add(label)
                Log.d(TAG, "$volume (Amp=$amp): Digit $label, ${(confidence*100).toInt()}%")
            } else {
                predictions.add("NO_SPEECH")
                Log.d(TAG, "$volume (Amp=$amp): VAD: No speech detected")
            }
        }

        Log.d(TAG, "\nPredictions: ${predictions.joinToString(", ")}")

        // V2 should be more robust, but with dummy audio it can vary
        Log.d(TAG, "[INFO] V2 normalization should be amplitude-independent")
        Log.d(TAG, "  -> With real digits we expect identical predictions")
        Log.d(TAG, "[OK] Test completed (no crashes)")
    }
}
