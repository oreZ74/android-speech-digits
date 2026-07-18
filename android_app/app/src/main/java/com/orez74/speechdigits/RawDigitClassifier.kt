package com.orez74.speechdigits

import android.content.Context
import android.os.Environment
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Facade for the digit recognition pipeline.
 *
 * Orchestrates:
 * - TfliteModelLoader: model lifecycle and inference
 * - SegmentClassifier: single-digit classification with multi-candidate normalization
 * - PinAssemblyEngine: PIN recording → VAD → classification → assembly
 * - InferenceBenchmark: performance benchmarking
 *
 * All data classes remain here for backward compatibility with existing callers.
 */
class RawDigitClassifier(private val context: Context) {

    private val modelLoader = TfliteModelLoader(context)
    private val segmentClassifier = SegmentClassifier(modelLoader)
    private val pinAssembly = PinAssemblyEngine(segmentClassifier, context)

    // ==========================================
    // Data classes (public API for callers)
    // ==========================================

    enum class SegmentStatus { OK, UNSURE }

    enum class ClassificationResult { TOO_QUIET, UNSURE, DIGIT }

    data class SpeechSegment(
        val startSample: Int,
        val endSample: Int,
        val durationMs: Int,
        val peakRms: Float
    )

    data class SegmentResult(
        val segmentIndex: Int,
        val segment: SpeechSegment,
        val label: String,
        val confidence: Float,
        val margin: Float,
        val status: SegmentStatus
    ) {
        val isValidDigit: Boolean
            get() = status == SegmentStatus.OK && label.length == 1 && label[0].isDigit()
    }

    data class ClassificationOutput(
        val result: ClassificationResult,
        val label: String?,
        val confidence: Float,
        val rmsAmplitude: Float,
        val peakAmplitude: Int,
        val allProbabilities: FloatArray
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false
            other as ClassificationOutput
            return result == other.result && label == other.label &&
                   confidence == other.confidence && rmsAmplitude == other.rmsAmplitude &&
                   peakAmplitude == other.peakAmplitude && allProbabilities.contentEquals(other.allProbabilities)
        }
        override fun hashCode(): Int {
            var result1 = result.hashCode()
            result1 = 31 * result1 + (label?.hashCode() ?: 0)
            result1 = 31 * result1 + confidence.hashCode()
            result1 = 31 * result1 + rmsAmplitude.hashCode()
            result1 = 31 * result1 + peakAmplitude
            result1 = 31 * result1 + allProbabilities.contentHashCode()
            return result1
        }
    }

    data class PinResult(
        val pin: String,
        val segmentResults: List<SegmentResult>,
        val segments: List<SpeechSegment>,
        val totalSegments: Int,
        val validDigits: Int,
        val allSegmentsOk: Boolean,
        val pipeline: String = "VAD"
    )

    data class DebugExport(
        val timestamp: String,
        val fullRecordingPath: String?,
        val segmentPaths: List<String>,
        val vadLog: String
    )

    data class BenchmarkResult(
        val warmupRuns: Int,
        val measureRuns: Int,
        val medianMs: Double,
        val p95Ms: Double,
        val minMs: Double,
        val maxMs: Double,
        val modelPath: String,
        val androidVersion: String,
        val deviceModel: String
    )

    // ==========================================
    // Facade methods (delegated to components)
    // ==========================================

    fun classifyWithConfidence(pcm16: ShortArray, sampleRate: Int = AudioPreprocessor.TARGET_SAMPLE_RATE): Pair<String, Float>? {
        return segmentClassifier.classifyWithConfidence(pcm16, sampleRate)
    }

    fun classifyWithMetrics(pcm16: ShortArray, sampleRate: Int = AudioPreprocessor.TARGET_SAMPLE_RATE): ClassificationOutput? {
        return segmentClassifier.classifyWithMetrics(pcm16, sampleRate)
    }

    fun classifyPin(pcm16: ShortArray, sampleRate: Int = AudioPreprocessor.TARGET_SAMPLE_RATE): PinResult {
        return pinAssembly.classifyPin(pcm16, sampleRate)
    }

    fun getLabels(): List<String> = modelLoader.getLabels()

    fun getLastDebugExport(): DebugExport? = pinAssembly.getLastDebugExport()

    fun runInferenceBenchmark(warmupRuns: Int = 20, measureRuns: Int = 100): BenchmarkResult {
        return InferenceBenchmark.runBenchmark(modelLoader, warmupRuns, measureRuns)
    }

    fun runAndSaveBenchmark(): String? {
        val result = runInferenceBenchmark()

        val manufacturer = android.os.Build.MANUFACTURER
        val model = android.os.Build.MODEL
        val abi = android.os.Build.SUPPORTED_ABIS.firstOrNull() ?: "unknown"
        val sdkInt = android.os.Build.VERSION.SDK_INT
        val inputShapeStr = modelLoader.inputShape.joinToString("x")

        val json = """
{
  "benchmark": {
    "warmup_runs": ${result.warmupRuns},
    "measure_runs": ${result.measureRuns},
    "median_ms": ${"%.3f".format(result.medianMs)},
    "p95_ms": ${"%.3f".format(result.p95Ms)},
    "min_ms": ${"%.3f".format(result.minMs)},
    "max_ms": ${"%.3f".format(result.maxMs)},
    "input_type": "silence"
  },
  "device": {
    "manufacturer": "$manufacturer",
    "model": "$model",
    "android_version": "${result.androidVersion}",
    "sdk_int": $sdkInt,
    "abi": "$abi"
  },
  "model": {
    "path": "${result.modelPath}",
    "input_shape": "$inputShapeStr"
  },
  "timestamp": "${SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(Date())}"
}
""".trimIndent()

        return try {
            val dir = context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)
            val file = File(dir, "inference_latency.json")
            FileOutputStream(file).use { it.write(json.toByteArray()) }
            AppLog.i(TAG) { "Benchmark saved to: ${file.absolutePath}" }
            file.absolutePath
        } catch (e: Exception) {
            AppLog.e(TAG, e) { "Failed to save benchmark: ${e.message}" }
            null
        }
    }

    fun close() {
        modelLoader.close()
        AppLog.d(TAG) { "Classifier closed" }
    }

    companion object {
        const val TAG = "RawDigitClassifier"
    }
}
