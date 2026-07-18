package com.orez74.speechdigits

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.os.Environment
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.github.orez74.speechdigits.R
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.concurrent.thread

/**
 * Dataset Recorder Activity
 *
 * Enables recording of individual digits (0-9) for training data.
 *
 * Features:
 * - Grid with digit buttons (0-9)
 * - 1.5s recording at 16kHz mono
 * - Automatic trimming to 1.0s (16000 samples)
 * - WAV export to app directory
 *
 * Storage location: DatasetDigits/<digit>/<digit>_spk-default_dev-android_<timestamp>.wav
 */
class DigitDatasetRecorderActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "DatasetRecorder"

        // Audio-Parameter
        private const val SAMPLE_RATE = 16000
        private const val RECORD_DURATION_MS = 1500
        private const val TARGET_SAMPLES = 16000
        private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT

        // Debug mode: Saves raw + trimmed WAV for analysis
        private var debugDatasetRecorderEnabled = false

        // Asymmetric trim: 0.4s before peak, 0.6s after peak
        // (peak is typically at the beginning of the digit, not the center)
        private const val OFFSET_BEFORE_PEAK_SAMPLES = (0.4 * SAMPLE_RATE).toInt() // 6400 samples

        private const val PERMISSION_REQUEST_CODE = 200
        private const val DATASET_FOLDER = "DatasetDigits"
    }

    private lateinit var tvTitle: TextView
    private lateinit var tvActiveDigit: TextView
    private lateinit var tvSampleCounts: TextView
    private lateinit var btnRecord: Button
    private lateinit var digitButtons: List<Button>

    private var activeDigit: Int = 0
    private var isRecording = false
    private var audioRecord: AudioRecord? = null

    private val sampleCounts = mutableMapOf<Int, Int>().apply {
        for (i in 0..9) this[i] = 0
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        SettingsActivity.applyTheme(this)
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_digit_dataset_recorder)

        AppLog.init(this)
        debugDatasetRecorderEnabled = AppLog.DEBUG_ENABLED

        initViews()
        setupDigitButtons()
        setupRecordButton()
        updateUI()
        countExistingSamples()

        AppLog.d(TAG) { "DigitDatasetRecorderActivity created" }
    }

    private fun initViews() {
        tvTitle = findViewById(R.id.tvTitle)
        tvActiveDigit = findViewById(R.id.tvActiveDigit)
        tvSampleCounts = findViewById(R.id.tvSampleCounts)
        btnRecord = findViewById(R.id.btnRecord)

        digitButtons = listOf(
            findViewById(R.id.btn0),
            findViewById(R.id.btn1),
            findViewById(R.id.btn2),
            findViewById(R.id.btn3),
            findViewById(R.id.btn4),
            findViewById(R.id.btn5),
            findViewById(R.id.btn6),
            findViewById(R.id.btn7),
            findViewById(R.id.btn8),
            findViewById(R.id.btn9)
        )
    }

    private fun setupDigitButtons() {
        digitButtons.forEachIndexed { index, button ->
            button.setOnClickListener {
                activeDigit = index
                updateUI()
                AppLog.v(TAG) { "Selected digit: $activeDigit" }
            }
        }
    }

    private fun setupRecordButton() {
        btnRecord.setOnClickListener {
            if (!isRecording) {
                if (checkPermission()) {
                    startRecording()
                } else {
                    requestPermission()
                }
            }
        }
    }

    private fun updateUI() {
        tvActiveDigit.text = "Active digit: $activeDigit"

        digitButtons.forEachIndexed { index, button ->
            if (index == activeDigit) {
                button.setBackgroundColor(ContextCompat.getColor(this, R.color.colorPrimary))
                button.setTextColor(ContextCompat.getColor(this, R.color.white))
            } else {
                button.setBackgroundColor(ContextCompat.getColor(this, R.color.colorSurfaceVariant))
                button.setTextColor(ContextCompat.getColor(this, R.color.debug_text))
            }
        }

        val total = sampleCounts.values.sum()
        val countsText = buildString {
            append("Samples per digit:\n")
            for (row in 0..1) {
                for (col in 0..4) {
                    val digit = row * 5 + col
                    val count = sampleCounts[digit] ?: 0
                    append(String.format("%d:%3d", digit, count))
                    if (col < 4) append("  ")
                }
                append("\n")
            }
            append("----------------\n")
            append("Total: $total")
        }
        tvSampleCounts.text = countsText

        btnRecord.isEnabled = !isRecording
        btnRecord.text = if (isRecording) "Recording..." else "Start recording"
    }

    private fun countExistingSamples() {
        for (i in 0..9) sampleCounts[i] = 0

        val baseDir = File(
            getExternalFilesDir(Environment.DIRECTORY_MUSIC),
            DATASET_FOLDER
        )

        if (!baseDir.exists()) {
            AppLog.d(TAG) { "Dataset folder does not exist yet: $baseDir" }
            updateUI()
            return
        }

        var foundSubfolders = false
        for (digit in 0..9) {
            val digitDir = File(baseDir, digit.toString())
            if (digitDir.exists() && digitDir.isDirectory) {
                foundSubfolders = true
                val wavCount = digitDir.listFiles { file ->
                    file.name.endsWith(".wav", ignoreCase = true)
                }?.size ?: 0
                sampleCounts[digit] = wavCount
            }
        }

        if (!foundSubfolders) {
            baseDir.listFiles()?.forEach { file ->
                if (file.isFile && file.name.endsWith(".wav", ignoreCase = true)) {
                    val digitChar = file.name.firstOrNull()
                    if (digitChar?.isDigit() == true) {
                        val digit = digitChar.toString().toInt()
                        sampleCounts[digit] = (sampleCounts[digit] ?: 0) + 1
                    }
                }
            }
        }

        val total = sampleCounts.values.sum()
        AppLog.d(TAG) { "Counted samples: $sampleCounts (Total: $total)" }
        updateUI()
    }

    private fun checkPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun requestPermission() {
        ActivityCompat.requestPermissions(
            this,
            arrayOf(Manifest.permission.RECORD_AUDIO),
            PERMISSION_REQUEST_CODE
        )
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == PERMISSION_REQUEST_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startRecording()
            } else {
                Toast.makeText(this, "Microphone permission required!", Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun startRecording() {
        isRecording = true
        updateUI()

        thread {
            try {
                val bufferSize = AudioRecord.getMinBufferSize(
                    SAMPLE_RATE,
                    CHANNEL_CONFIG,
                    AUDIO_FORMAT
                )

                val recorder = AudioRecord(
                    MediaRecorder.AudioSource.MIC,
                    SAMPLE_RATE,
                    CHANNEL_CONFIG,
                    AUDIO_FORMAT,
                    bufferSize * 2
                )

                audioRecord = recorder

                if (recorder.state != AudioRecord.STATE_INITIALIZED) {
                    runOnUiThread {
                        Toast.makeText(this, "AudioRecord initialization failed", Toast.LENGTH_SHORT).show()
                        isRecording = false
                        updateUI()
                    }
                    return@thread
                }

                val totalSamples = (SAMPLE_RATE * RECORD_DURATION_MS) / 1000
                val recordBuffer = ShortArray(totalSamples)
                var samplesRead = 0

                recorder.startRecording()
                AppLog.d(TAG) { "Recording started for digit $activeDigit" }

                while (samplesRead < totalSamples && isRecording) {
                    val remaining = totalSamples - samplesRead
                    val toRead = minOf(bufferSize / 2, remaining)
                    val read = recorder.read(recordBuffer, samplesRead, toRead)
                    if (read > 0) {
                        samplesRead += read
                    }
                }

                recorder.stop()
                recorder.release()
                audioRecord = null

                AppLog.d(TAG) { "Recording stopped, got $samplesRead samples" }

                if (samplesRead > 0) {
                    // Debug: save raw audio before trimming
                    if (debugDatasetRecorderEnabled) {
                        saveDebugWav(recordBuffer.copyOf(samplesRead), activeDigit, "raw")
                    }

                    val trimmedSamples = processAndTrimAudio(recordBuffer, samplesRead)

                    // Debug: save trimmed audio
                    if (debugDatasetRecorderEnabled) {
                        saveDebugWav(trimmedSamples, activeDigit, "trimmed")
                    }

                    val savedPath = saveAsWav(trimmedSamples, activeDigit)

                    runOnUiThread {
                        if (savedPath != null) {
                            sampleCounts[activeDigit] = (sampleCounts[activeDigit] ?: 0) + 1
                            Toast.makeText(this, "Saved: Digit $activeDigit", Toast.LENGTH_SHORT).show()
                        } else {
                            Toast.makeText(this, "Error saving!", Toast.LENGTH_SHORT).show()
                        }
                        isRecording = false
                        updateUI()
                    }
                } else {
                    runOnUiThread {
                        Toast.makeText(this, "No audio data recorded", Toast.LENGTH_SHORT).show()
                        isRecording = false
                        updateUI()
                    }
                }

            } catch (e: Exception) {
                AppLog.e(TAG, e) { "Recording error" }
                runOnUiThread {
                    Toast.makeText(this, "Recording error: ${e.message}", Toast.LENGTH_SHORT).show()
                    isRecording = false
                    updateUI()
                }
            }
        }
    }

    private fun processAndTrimAudio(buffer: ShortArray, samplesRead: Int): ShortArray {
        val windowSize = SAMPLE_RATE / 20
        val hopSize = windowSize / 2

        var maxRms = 0.0
        var maxRmsCenter = samplesRead / 2

        var windowStart = 0
        while (windowStart + windowSize <= samplesRead) {
            var sumSquares = 0.0
            for (i in windowStart until windowStart + windowSize) {
                val sample = buffer[i].toDouble()
                sumSquares += sample * sample
            }
            val rms = kotlin.math.sqrt(sumSquares / windowSize)

            if (rms > maxRms) {
                maxRms = rms
                maxRmsCenter = windowStart + windowSize / 2
            }

            windowStart += hopSize
        }

        AppLog.v(TAG) { "Max RMS center at sample $maxRmsCenter (RMS=${"%.1f".format(maxRms)})" }

        // Asymmetric trim: 0.4s before peak, 0.6s after peak
        // (peak is typically at the beginning of the digit, not the center)
        var startIdx = maxRmsCenter - OFFSET_BEFORE_PEAK_SAMPLES
        var endIdx = startIdx + TARGET_SAMPLES

        if (startIdx < 0) {
            startIdx = 0
            endIdx = TARGET_SAMPLES
        }
        if (endIdx > samplesRead) {
            endIdx = samplesRead
            startIdx = maxOf(0, samplesRead - TARGET_SAMPLES)
        }

        AppLog.v(TAG) { "Asymmetric trim: start=$startIdx, end=$endIdx (peak at $maxRmsCenter, offset_before=${OFFSET_BEFORE_PEAK_SAMPLES})" }

        val actualLength = endIdx - startIdx
        val result = ShortArray(TARGET_SAMPLES)

        if (actualLength >= TARGET_SAMPLES) {
            System.arraycopy(buffer, startIdx, result, 0, TARGET_SAMPLES)
        } else {
            val padBefore = (TARGET_SAMPLES - actualLength) / 2
            System.arraycopy(buffer, startIdx, result, padBefore, actualLength)
        }

        AppLog.v(TAG) { "Trimmed to $TARGET_SAMPLES samples (from $samplesRead)" }
        return result
    }

    /**
     * Debug: Saves WAV files for analysis (raw + trimmed)
     * Folder: DatasetRecorderDebug/<digit>/
     */
    private fun saveDebugWav(samples: ShortArray, digit: Int, suffix: String) {
        try {
            val baseDir = File(
                getExternalFilesDir(Environment.DIRECTORY_MUSIC),
                "DatasetRecorderDebug"
            )

            val digitDir = File(baseDir, digit.toString())
            if (!digitDir.exists()) {
                digitDir.mkdirs()
            }

            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date())
            val filename = "${digit}_${suffix}_$timestamp.wav"
            val file = File(digitDir, filename)

            FileOutputStream(file).use { fos ->
                writeWavHeader(fos, samples.size, SAMPLE_RATE)
                val dataSize = samples.size * 2
                val buffer = ByteBuffer.allocate(dataSize).order(ByteOrder.LITTLE_ENDIAN)
                samples.forEach { buffer.putShort(it) }
                fos.write(buffer.array())
            }

            AppLog.d(TAG) { "Debug WAV saved: ${file.absolutePath} (${samples.size} samples)" }

        } catch (e: Exception) {
            AppLog.e(TAG, e) { "Failed to save debug WAV" }
        }
    }

    private fun saveAsWav(samples: ShortArray, digit: Int): String? {
        try {
            val baseDir = File(
                getExternalFilesDir(Environment.DIRECTORY_MUSIC),
                DATASET_FOLDER
            )

            val digitDir = File(baseDir, digit.toString())
            if (!digitDir.exists()) {
                digitDir.mkdirs()
                AppLog.d(TAG) { "Created digit folder: ${digitDir.absolutePath}" }
            }

            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val filename = "${digit}_spk-default_dev-android_$timestamp.wav"
            val file = File(digitDir, filename)

            AppLog.d(TAG) { "Saving WAV to: ${file.absolutePath}" }

            FileOutputStream(file).use { fos ->
                writeWavHeader(fos, samples.size, SAMPLE_RATE)

                val dataSize = samples.size * 2
                val buffer = ByteBuffer.allocate(dataSize).order(ByteOrder.LITTLE_ENDIAN)
                samples.forEach { buffer.putShort(it) }
                fos.write(buffer.array())
            }

            AppLog.d(TAG) { "Saved: ${file.absolutePath} (${samples.size} samples, ${file.length()} bytes)" }
            return file.absolutePath

        } catch (e: Exception) {
            AppLog.e(TAG, e) { "Failed to save WAV" }
            return null
        }
    }

    private fun writeWavHeader(fos: FileOutputStream, numSamples: Int, sampleRate: Int) {
        val channels = 1
        val bitsPerSample = 16
        val byteRate = sampleRate * channels * bitsPerSample / 8
        val blockAlign = channels * bitsPerSample / 8
        val dataSize = numSamples * channels * bitsPerSample / 8
        val fileSize = 36 + dataSize

        fos.write("RIFF".toByteArray(Charsets.US_ASCII))
        fos.write(intToLittleEndian(fileSize))
        fos.write("WAVE".toByteArray(Charsets.US_ASCII))

        fos.write("fmt ".toByteArray(Charsets.US_ASCII))
        fos.write(intToLittleEndian(16))
        fos.write(shortToLittleEndian(1))
        fos.write(shortToLittleEndian(channels))
        fos.write(intToLittleEndian(sampleRate))
        fos.write(intToLittleEndian(byteRate))
        fos.write(shortToLittleEndian(blockAlign))
        fos.write(shortToLittleEndian(bitsPerSample))

        fos.write("data".toByteArray(Charsets.US_ASCII))
        fos.write(intToLittleEndian(dataSize))
    }

    private fun intToLittleEndian(value: Int): ByteArray {
        return ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(value).array()
    }

    private fun shortToLittleEndian(value: Int): ByteArray {
        return ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN).putShort(value.toShort()).array()
    }

    override fun onDestroy() {
        super.onDestroy()
        audioRecord?.let {
            if (it.state == AudioRecord.STATE_INITIALIZED) {
                it.stop()
                it.release()
            }
        }
        audioRecord = null
    }
}
