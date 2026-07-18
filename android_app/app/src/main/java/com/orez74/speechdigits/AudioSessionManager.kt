package com.orez74.speechdigits

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.media.audiofx.AcousticEchoCanceler
import android.media.audiofx.AutomaticGainControl
import android.media.audiofx.NoiseSuppressor
import androidx.annotation.RequiresPermission
import android.Manifest

/**
 * AudioSessionManager - Central audio setup for PIN recognition
 *
 * Responsibilities:
 * - Creation and configuration of AudioRecord instances
 * - Calculation of optimal buffer sizes
 * - Disabling DSP effects (AGC, NS, AEC) for ML consistency
 * - Reading PCM data from the microphone
 *
 * IMPORTANT for ML speech recognition:
 * - The model was trained with raw audio data (without DSP effects)
 * - AGC/NS/AEC modify amplitude and frequency response device-dependently
 * - Consistency between training and inference is essential
 */
class AudioSessionManager {

    companion object {
        private const val TAG = "AudioSessionManager"

        // Audio configuration (MUST match Python training!)
        const val SAMPLE_RATE = 16000
        const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
        const val AUDIO_SOURCE = MediaRecorder.AudioSource.VOICE_RECOGNITION
    }

    /**
     * Configuration for an audio recording session
     */
    data class RecordingConfig(
        val durationMs: Int,
        val sampleRate: Int = SAMPLE_RATE,
        val channelConfig: Int = CHANNEL_CONFIG,
        val audioFormat: Int = AUDIO_FORMAT,
        val audioSource: Int = AUDIO_SOURCE
    ) {
        /** Number of samples based on duration and sample rate */
        val bufferSizeSamples: Int
            get() = (durationMs * sampleRate) / 1000
    }

    /**
     * Result of an audio recording session
     */
    data class RecordingResult(
        val samples: ShortArray,
        val samplesRead: Int,
        val sampleRate: Int,
        val durationMs: Int
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false
            other as RecordingResult
            return samples.contentEquals(other.samples) &&
                   samplesRead == other.samplesRead &&
                   sampleRate == other.sampleRate &&
                   durationMs == other.durationMs
        }

        override fun hashCode(): Int {
            var result = samples.contentHashCode()
            result = 31 * result + samplesRead
            result = 31 * result + sampleRate
            result = 31 * result + durationMs
            return result
        }
    }

    // Active AudioRecord instance (for cleanup)
    private var activeRecord: AudioRecord? = null

    /**
     * Creates a configured AudioRecord for PIN recording
     *
     * @param config Recording configuration
     * @return Configured AudioRecord in STATE_INITIALIZED
     * @throws IllegalStateException if AudioRecord cannot be initialized
     */
    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    fun createRecorder(config: RecordingConfig): AudioRecord {
        val minBufferSize = AudioRecord.getMinBufferSize(
            config.sampleRate,
            config.channelConfig,
            config.audioFormat
        )
        val bufferSize = maxOf(minBufferSize, config.bufferSizeSamples * 2)

        AppLog.d(TAG) {
            "PIN RECORDING CONFIG: AudioSource=VOICE_RECOGNITION, SampleRate=${config.sampleRate} Hz, " +
            "Duration=${config.durationMs} ms (${config.bufferSizeSamples} samples), " +
            "BufferSize=$bufferSize bytes (min=$minBufferSize)"
        }

        val recorder = AudioRecord(
            config.audioSource,
            config.sampleRate,
            config.channelConfig,
            config.audioFormat,
            bufferSize
        )

        AppLog.v(TAG) { "AudioRecord actualSampleRate=${recorder.sampleRate}" }

        if (recorder.state != AudioRecord.STATE_INITIALIZED) {
            recorder.release()
            throw IllegalStateException("AudioRecord could not be initialized")
        }

        activeRecord = recorder
        return recorder
    }

    /**
     * Performs a complete audio recording
     *
     * @param config Recording configuration
     * @return RecordingResult with the recorded samples
     * @throws IllegalStateException on recording errors
     */
    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    fun recordAudio(config: RecordingConfig): RecordingResult {
        val recorder = createRecorder(config)

        try {
            // Disable DSP effects for ML consistency
            disableDspEffects(recorder.audioSessionId)

            recorder.startRecording()
            AppLog.d(TAG) { "Recording started (sessionId=${recorder.audioSessionId})" }

            val audioData = ShortArray(config.bufferSizeSamples)
            var samplesRead = 0

            while (samplesRead < config.bufferSizeSamples) {
                val remaining = config.bufferSizeSamples - samplesRead
                val read = recorder.read(audioData, samplesRead, remaining)

                if (read > 0) {
                    samplesRead += read
                } else {
                    AppLog.w(TAG) { "AudioRecord.read() returned $read" }
                    break
                }
            }

            recorder.stop()
            AppLog.d(TAG) { "Recording completed: $samplesRead samples" }

            return RecordingResult(
                samples = audioData,
                samplesRead = samplesRead,
                sampleRate = config.sampleRate,
                durationMs = config.durationMs
            )

        } finally {
            recorder.release()
            activeRecord = null
        }
    }

    /**
     * Disables DSP effects (AGC, NoiseSuppressor, AcousticEchoCanceler)
     * for consistent audio input without device-specific post-processing.
     *
     * IMPORTANT for ML speech recognition:
     * - The model was trained with raw audio data (without DSP effects)
     * - AGC/NS/AEC modify amplitude and frequency response device-dependently
     * - Consistency between training and inference is essential
     *
     * @param audioSessionId Session ID of the AudioRecord
     */
    fun disableDspEffects(audioSessionId: Int) {
        AppLog.d(TAG) { "Disabling DSP effects for sessionId=$audioSessionId (ML consistency)" }

        // Automatic Gain Control (AGC) - adjusts volume automatically
        if (AutomaticGainControl.isAvailable()) {
            try {
                val agc = AutomaticGainControl.create(audioSessionId)
                agc?.enabled = false
                AppLog.v(TAG) { "AGC disabled" }
            } catch (e: Exception) {
                AppLog.w(TAG) { "Could not disable AGC: ${e.message}" }
            }
        } else {
            AppLog.v(TAG) { "AGC not available on this device" }
        }

        // Noise Suppressor (NS) - suppresses background noise
        if (NoiseSuppressor.isAvailable()) {
            try {
                val ns = NoiseSuppressor.create(audioSessionId)
                ns?.enabled = false
                AppLog.v(TAG) { "NoiseSuppressor disabled" }
            } catch (e: Exception) {
                AppLog.w(TAG) { "Could not disable NoiseSuppressor: ${e.message}" }
            }
        } else {
            AppLog.v(TAG) { "NoiseSuppressor not available on this device" }
        }

        // Acoustic Echo Canceler (AEC) - removes echo
        if (AcousticEchoCanceler.isAvailable()) {
            try {
                val aec = AcousticEchoCanceler.create(audioSessionId)
                aec?.enabled = false
                AppLog.v(TAG) { "AEC disabled" }
            } catch (e: Exception) {
                AppLog.w(TAG) { "Could not disable AEC: ${e.message}" }
            }
        } else {
            AppLog.v(TAG) { "AEC not available on this device" }
        }

        AppLog.v(TAG) { "DSP effects configuration complete" }
    }

    /**
     * Stops and releases active recording resources
     */
    fun release() {
        activeRecord?.let { recorder ->
            try {
                if (recorder.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                    recorder.stop()
                }
                recorder.release()
            } catch (e: Exception) {
                AppLog.w(TAG) { "Error releasing AudioRecord: ${e.message}" }
            }
            activeRecord = null
        }
    }
}
