package com.orez74.speechdigits

import android.content.Context
import android.os.Build
import androidx.core.content.pm.PackageInfoCompat
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.InputStreamReader
import java.io.BufferedReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.security.MessageDigest

/**
 * Manages TFLite model lifecycle: loading, SHA-256 verification, label parsing, inference.
 */
class TfliteModelLoader(private val context: Context) {

    private var interpreter: Interpreter? = null
    private var labels: List<String> = emptyList()

    var inputShape: IntArray = intArrayOf()
        private set
    var outputShape: IntArray = intArrayOf()
        private set

    val numClasses: Int
        get() = if (outputShape.isNotEmpty()) outputShape.last() else 0

    init {
        loadModel()
        loadLabels()
        inspectModel()

        AppLog.i(TAG) { "Model loader initialized - SHA-256: $EXPECTED_MODEL_SHA256" }
    }

    private fun loadModel() {
        try {
            val assetManager = context.assets

            val modelBytes: ByteArray = assetManager.open(MODEL_PATH).use { it.readBytes() }
            val modelSizeBytes = modelBytes.size

            val actualSha256 = calculateSha256(modelBytes)
            logModelFingerprint(MODEL_PATH, modelSizeBytes, actualSha256)
            verifyModelHash(actualSha256)

            val assetFileDescriptor = assetManager.openFd(MODEL_PATH)
            val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
            val fileChannel = fileInputStream.channel
            val startOffset = assetFileDescriptor.startOffset
            val declaredLength = assetFileDescriptor.declaredLength
            val modelBuffer: MappedByteBuffer = fileChannel.map(
                FileChannel.MapMode.READ_ONLY,
                startOffset,
                declaredLength
            )

            val options = Interpreter.Options().apply {
                setNumThreads(4)
            }

            interpreter = Interpreter(modelBuffer, options)
            logTensorInfo()

            AppLog.i(TAG) { "Model loaded and verified: $MODEL_PATH" }
        } catch (e: Exception) {
            AppLog.e(TAG, e) { "Failed to load model: ${e.message}" }
            throw RuntimeException("Model loading failed", e)
        }
    }

    private fun calculateSha256(bytes: ByteArray): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val hashBytes = digest.digest(bytes)
        return hashBytes.joinToString("") { "%02x".format(it) }
    }

    private fun logModelFingerprint(assetPath: String, sizeBytes: Int, sha256: String) {
        val sizeMb = sizeBytes / (1024.0 * 1024.0)
        val buildType = if (AppLog.DEBUG_ENABLED) "DEBUG" else "RELEASE"
        val packageInfo = context.packageManager.getPackageInfo(context.packageName, 0)
        val versionName = packageInfo.versionName ?: "unknown"
        val versionCode = PackageInfoCompat.getLongVersionCode(packageInfo).toString()

        val fingerprint = buildString {
            appendLine("")
            appendLine("╔══════════════════════════════════════════════════════════════════╗")
            appendLine("║                    MODEL FINGERPRINT                             ║")
            appendLine("╠══════════════════════════════════════════════════════════════════╣")
            appendLine("║ Asset Path:    $assetPath")
            appendLine("║ Size:          $sizeBytes bytes (${"%.2f".format(sizeMb)} MB)")
            appendLine("║ SHA-256:       $sha256")
            appendLine("║ Expected SHA:  $EXPECTED_MODEL_SHA256")
            appendLine("║ Hash Match:    ${if (sha256 == EXPECTED_MODEL_SHA256) "[OK] YES" else "[FAIL] NO (MISMATCH!)"}")
            appendLine("╠══════════════════════════════════════════════════════════════════╣")
            appendLine("║ Build Type:    $buildType")
            appendLine("║ Version:       $versionName ($versionCode)")
            appendLine("║ Device:        ${Build.MANUFACTURER} ${Build.MODEL}")
            appendLine("║ Android:       ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})")
            appendLine("╚══════════════════════════════════════════════════════════════════╝")
        }

        AppLog.i(TAG_FINGERPRINT) { fingerprint }
    }

    private fun verifyModelHash(actualSha256: String) {
        if (actualSha256 != EXPECTED_MODEL_SHA256) {
            val message = "Wrong TFLite model loaded! expected=$EXPECTED_MODEL_SHA256 actual=$actualSha256"
            AppLog.e(TAG_FINGERPRINT) { "[WARNING] MODEL HASH MISMATCH: $message" }

            if (AppLog.DEBUG_ENABLED) {
                throw IllegalStateException(message)
            }
        } else {
            AppLog.i(TAG_FINGERPRINT) { "[OK] Model hash verified successfully" }
        }
    }

    private fun logTensorInfo() {
        interpreter?.let { interp ->
            val inputTensor = interp.getInputTensor(0)
            val outputTensor = interp.getOutputTensor(0)

            val tensorInfo = buildString {
                appendLine("")
                appendLine("╔══════════════════════════════════════════════════════════════════╗")
                appendLine("║                    TENSOR INFO                                   ║")
                appendLine("╠══════════════════════════════════════════════════════════════════╣")
                appendLine("║ INPUT TENSOR [0]:")
                appendLine("║   Shape:      ${inputTensor.shape().contentToString()}")
                appendLine("║   DataType:   ${inputTensor.dataType()}")
                appendLine("║   NumBytes:   ${inputTensor.numBytes()}")
                try {
                    val inputParams = inputTensor.quantizationParams()
                    appendLine("║   Quant:      scale=${inputParams.scale}, zeroPoint=${inputParams.zeroPoint}")
                } catch (e: Exception) {
                    appendLine("║   Quant:      N/A (Float model)")
                }
                appendLine("║")
                appendLine("║ OUTPUT TENSOR [0]:")
                appendLine("║   Shape:      ${outputTensor.shape().contentToString()}")
                appendLine("║   DataType:   ${outputTensor.dataType()}")
                appendLine("║   NumBytes:   ${outputTensor.numBytes()}")
                try {
                    val outputParams = outputTensor.quantizationParams()
                    appendLine("║   Quant:      scale=${outputParams.scale}, zeroPoint=${outputParams.zeroPoint}")
                } catch (e: Exception) {
                    appendLine("║   Quant:      N/A (Float model)")
                }
                appendLine("╚══════════════════════════════════════════════════════════════════╝")
            }

            AppLog.i(TAG_FINGERPRINT) { tensorInfo }
        }
    }

    private fun loadLabels() {
        try {
            context.assets.open(LABEL_PATH).use { inputStream ->
                BufferedReader(InputStreamReader(inputStream)).use { reader ->
                    labels = reader.lineSequence()
                        .map { it.trim() }
                        .filter { it.isNotEmpty() }
                        .toList()
                }
            }
            AppLog.d(TAG) { "Labels loaded: ${labels.size} (${labels.joinToString(", ")})" }
        } catch (e: Exception) {
            AppLog.e(TAG, e) { "Failed to load labels: ${e.message}" }
            labels = (0..9).map { it.toString() }
            AppLog.w(TAG) { "Using default labels: ${labels.joinToString(", ")}" }
        }
    }

    private fun inspectModel() {
        interpreter?.let { interp ->
            inputShape = interp.getInputTensor(0).shape()
            val inputDataType = interp.getInputTensor(0).dataType()
            outputShape = interp.getOutputTensor(0).shape()
            val outputDataType = interp.getOutputTensor(0).dataType()

            AppLog.d(TAG) {
                """MODEL SPEC: Input=${inputShape.contentToString()} ($inputDataType),
                   |Output=${outputShape.contentToString()} ($outputDataType)""".trimMargin()
            }
        }
    }

    fun createInputBuffer(samples: FloatArray): ByteBuffer {
        val bufferSize = samples.size * 4
        val buffer = ByteBuffer.allocateDirect(bufferSize).apply {
            order(ByteOrder.nativeOrder())
        }
        samples.forEach { sample -> buffer.putFloat(sample) }
        buffer.rewind()
        AppLog.v(TAG) { "Input buffer: ${buffer.capacity()} bytes (${samples.size} floats)" }
        return buffer
    }

    fun runInference(inputBuffer: ByteBuffer): FloatArray {
        val numClasses = outputShape.last()
        val outputBuffer = Array(1) { FloatArray(numClasses) }

        if (AppLog.DEBUG_ENABLED) {
            val startTimeMs = System.currentTimeMillis()
            interpreter?.run(inputBuffer, outputBuffer)
                ?: throw IllegalStateException("Interpreter not initialized")
            val latencyMs = System.currentTimeMillis() - startTimeMs
            AppLog.d(TAG) { "[PERFORMANCE] Inference latency: ${latencyMs}ms" }
        } else {
            interpreter?.run(inputBuffer, outputBuffer)
                ?: throw IllegalStateException("Interpreter not initialized")
        }

        return outputBuffer[0]
    }

    fun getLabels(): List<String> = labels

    fun close() {
        interpreter?.close()
        interpreter = null
        AppLog.d(TAG) { "Model loader closed" }
    }

    companion object {
        init {
            try {
                System.loadLibrary("tensorflowlite_flex_jni")
                System.loadLibrary("tensorflowlite_jni")
                android.util.Log.i("TFLiteInit", "Native TFLite 2.16.1 libraries linked successfully in sequence.")
            } catch (e: UnsatisfiedLinkError) {
                android.util.Log.e("TFLiteInit", "Fatal error loading native dependencies: ${e.message}")
            }
        }

        const val TAG = "TfliteModelLoader"
        const val TAG_FINGERPRINT = "MODEL_FINGERPRINT"

        const val EXPECTED_MODEL_SHA256 = "903d2d34091af5f9c91233d87cd2510939c0fe4486d2c65bc834279f102c1d2b"
        const val MODEL_PATH = "digits_rawwave_12cls.tflite"
        const val LABEL_PATH = "raw_digits_labels.txt"
        const val EXPECTED_SAMPLES = 16000
    }
}
