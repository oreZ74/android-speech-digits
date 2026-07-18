package com.orez74.speechdigits

/**
 * Pure audio signal processing utilities:
 * resampling, padding/truncation, PCM16-to-Float32 normalization.
 *
 * All normalization MUST match Python training:
 * float = pcm16 / 32768.0f (NO mean-centering, NO peak-normalize)
 */
object AudioPreprocessor {

    const val TAG = "AudioPreprocessor"

    const val TARGET_SAMPLE_RATE = 16000
    const val TARGET_DURATION = 1.0f
    const val EXPECTED_SAMPLES = 16000
    const val NORMALIZATION_FACTOR = 32768.0f

    const val RMS_TOO_QUIET_THRESHOLD = 30.0f
    const val PEAK_TOO_QUIET_THRESHOLD = 300f

    /**
     * Linear resampling (simple but fast).
     */
    fun resampleLinear(input: ShortArray, fromRate: Int, toRate: Int): ShortArray {
        if (fromRate == toRate) return input

        val ratio = fromRate.toDouble() / toRate.toDouble()
        val outputLength = (input.size / ratio).toInt()
        val output = ShortArray(outputLength)

        for (i in output.indices) {
            val srcIndex = (i * ratio).toInt()
            output[i] = if (srcIndex < input.size) input[srcIndex] else 0
        }

        AppLog.v(TAG) { "Resampled: ${input.size} -> ${output.size} samples" }
        return output
    }

    /**
     * Pad with zeros or truncate to target length.
     */
    fun padOrTruncate(input: ShortArray, targetLength: Int): ShortArray {
        return when {
            input.size == targetLength -> {
                AppLog.v(TAG) { "Length OK: ${input.size} samples" }
                input
            }
            input.size < targetLength -> {
                AppLog.v(TAG) { "Padding: ${input.size} -> $targetLength samples" }
                input.copyOf(targetLength)
            }
            else -> {
                AppLog.v(TAG) { "Truncating: ${input.size} -> $targetLength samples" }
                input.copyOf(targetLength)
            }
        }
    }

    /**
     * Normalize PCM16 to Float32 [-1.0, 1.0].
     *
     * NORMALIZATION (MUST match Python training):
     * 1. Convert PCM16 to Float32: float = pcm16 / 32768.0
     * 2. Optional: Clamp to [-1.0, 1.0] (safety for edge cases)
     *
     * NO mean-centering, NO peak-normalization!
     */
    fun normalize(pcm16: ShortArray): FloatArray {
        val normalized = FloatArray(pcm16.size) { i ->
            val raw = pcm16[i] / NORMALIZATION_FACTOR
            raw.coerceIn(-1.0f, 1.0f)
        }

        val min = normalized.minOrNull() ?: 0f
        val max = normalized.maxOrNull() ?: 0f
        val mean = normalized.average().toFloat()
        val first = normalized.firstOrNull() ?: 0f
        val last = normalized.lastOrNull() ?: 0f

        AppLog.v(TAG) { "Normalization: Min=${"%.4f".format(min)}, Max=${"%.4f".format(max)}, Mean=${"%.4f".format(mean)}, First=${"%.4f".format(first)}, Last=${"%.4f".format(last)}" }

        return normalized
    }

    fun calculatePeak(samples: ShortArray): Int {
        return samples.maxOfOrNull { kotlin.math.abs(it.toInt()) } ?: 0
    }

    fun calculateRms(samples: ShortArray): Float {
        return kotlin.math.sqrt(samples.map { it.toDouble() * it.toDouble() }.average()).toFloat()
    }
}
