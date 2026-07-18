package com.orez74.speechdigits

import android.content.Context
import android.os.Environment
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Exports PCM16 audio as WAV files for debugging (only when enabled).
 */
object WavExporter {

    const val TAG = "WavExporter"

    const val DEBUG_FOLDER = "DigitDebug"

    @JvmField
    var DEBUG_MODE_ENABLED: Boolean = false

    /**
     * Save PCM16 audio as WAV file (16kHz, mono, 16-bit).
     *
     * @return Absolute path of the saved file, or null on error / disabled
     */
    fun saveDebugWav(
        context: Context,
        name: String,
        samples: ShortArray,
        sampleRate: Int = 16000
    ): String? {
        if (!DEBUG_MODE_ENABLED) return null

        val timestamp = System.currentTimeMillis()
        val filename = "${timestamp}_${name}.wav"

        val dir = File(
            context.getExternalFilesDir(Environment.DIRECTORY_MUSIC),
            DEBUG_FOLDER
        )

        AppLog.d(TAG) { "Saving debug WAV '$name' with ${samples.size} samples to ${dir.absolutePath}" }

        try {
            if (!dir.exists()) {
                val created = dir.mkdirs()
                AppLog.d(TAG) { "Created directory: $created -> ${dir.absolutePath}" }
            }

            val file = File(dir, filename)

            FileOutputStream(file).use { fos ->
                writeWavHeader(fos, samples.size, sampleRate)

                val dataSize = samples.size * 2
                val buffer = ByteBuffer.allocate(dataSize).order(ByteOrder.LITTLE_ENDIAN)
                samples.forEach { buffer.putShort(it) }
                fos.write(buffer.array())
            }

            AppLog.d(TAG) { "Saved debug WAV: ${file.absolutePath}" }
            return file.absolutePath

        } catch (e: Exception) {
            AppLog.e(TAG, e) { "Failed to save debug WAV '$name'" }
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
        fos.write(intToLittleEndianBytes(fileSize))
        fos.write("WAVE".toByteArray(Charsets.US_ASCII))

        fos.write("fmt ".toByteArray(Charsets.US_ASCII))
        fos.write(intToLittleEndianBytes(16))
        fos.write(shortToLittleEndianBytes(1))
        fos.write(shortToLittleEndianBytes(channels))
        fos.write(intToLittleEndianBytes(sampleRate))
        fos.write(intToLittleEndianBytes(byteRate))
        fos.write(shortToLittleEndianBytes(blockAlign))
        fos.write(shortToLittleEndianBytes(bitsPerSample))

        fos.write("data".toByteArray(Charsets.US_ASCII))
        fos.write(intToLittleEndianBytes(dataSize))
    }

    private fun intToLittleEndianBytes(value: Int): ByteArray {
        return ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(value).array()
    }

    private fun shortToLittleEndianBytes(value: Int): ByteArray {
        return ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN).putShort(value.toShort()).array()
    }
}
