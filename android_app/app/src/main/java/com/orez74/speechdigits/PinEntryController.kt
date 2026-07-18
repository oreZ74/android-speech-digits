package com.orez74.speechdigits

import android.Manifest
import androidx.annotation.RequiresPermission
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * PinEntryController - Coordinates PIN recording and classification workflow
 *
 * Responsibilities:
 * - Start/Stop of PIN recording
 * - Coordination between AudioSessionManager and RawDigitClassifier
 * - Management of recording state
 * - Validation and aggregation of classification results
 * - Notification of UI via listener callbacks
 *
 * This class is independent of Android UI classes and can be tested.
 */
class PinEntryController(
    private val audioSessionManager: AudioSessionManager,
    private val classifier: RawDigitClassifier,
    private val listener: PinEntryListener,
    private val scope: CoroutineScope = CoroutineScope(Dispatchers.IO + Job())
) {
    companion object {
        private const val TAG = "PinEntryController"

        /** Expected PIN length */
        const val EXPECTED_PIN_LENGTH = 4
    }

    /**
     * Callback interface for PIN workflow events
     */
    interface PinEntryListener {
        /** Called when recording starts */
        fun onPinRecordingStarted()

        /** Called when recording is in progress and transitions to processing */
        fun onPinRecordingProcessing()

        /** Called when a PIN is successfully recognized */
        fun onPinRecognized(result: PinRecognitionResult)

        /** Called when no PIN was detected (silence/noise) */
        fun onNoPinDetected()

        /** Called when PIN was recognized but uncertain */
        fun onPinUnsure(result: PinRecognitionResult)

        /** Called on an error */
        fun onPinError(error: PinError)

        /** Called when recording is complete (regardless of success or error) */
        fun onPinRecordingComplete()
    }

    /**
     * Result of a PIN recognition attempt
     */
    data class PinRecognitionResult(
        val pin: String,
        val totalSegments: Int,
        val validDigits: Int,
        val segmentResults: List<RawDigitClassifier.SegmentResult>
    )

    /**
     * Error types for PIN recognition
     */
    sealed class PinError {
        data class ClassifierNotReady(val message: String) : PinError()
        data class RecordingFailed(val message: String, val cause: Throwable?) : PinError()
        data class ClassificationFailed(val message: String, val cause: Throwable?) : PinError()
    }

    // Recording state
    private var isRecording = false
    private var recordingJob: Job? = null

    /**
     * Starts PIN recording and classification
     *
     * @param durationMs Recording duration in milliseconds
     * @return true if recording started, false if one is already in progress
     */
    fun startPinRecording(durationMs: Int): Boolean {
        if (isRecording) {
            AppLog.w(TAG) { "Recording already in progress, ignoring start request" }
            return false
        }

        isRecording = true
        listener.onPinRecordingStarted()

        recordingJob = scope.launch(Dispatchers.IO) {
            try {
                performPinRecordingAndClassification(durationMs)
            } catch (e: Exception) {
                AppLog.e(TAG, e) { "Error during PIN recording/classification: ${e.message}" }
                withContext(Dispatchers.Main) {
                    listener.onPinError(PinError.RecordingFailed(
                        message = e.message ?: "Unknown error",
                        cause = e
                    ))
                }
            } finally {
                isRecording = false
                withContext(Dispatchers.Main) {
                    listener.onPinRecordingComplete()
                }
            }
        }

        return true
    }

    /**
     * Cancels the current recording
     */
    fun cancelRecording() {
        if (isRecording) {
            recordingJob?.cancel()
            audioSessionManager.release()
            isRecording = false
            AppLog.d(TAG) { "Recording cancelled" }
        }
    }

    /**
     * Checks if a recording is in progress
     */
    fun isRecordingActive(): Boolean = isRecording

    /**
     * Performs the actual recording and classification
     */
    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    private suspend fun performPinRecordingAndClassification(durationMs: Int) {
        // 1. Record audio
        val config = AudioSessionManager.RecordingConfig(durationMs = durationMs)
        val recordingResult = audioSessionManager.recordAudio(config)

        // 2. Notify UI about processing phase
        withContext(Dispatchers.Main) {
            listener.onPinRecordingProcessing()
        }

        // 3. Perform classification
        val pinResult = classifier.classifyPin(
            recordingResult.samples,
            recordingResult.sampleRate
        )

        // 4. Validate result and notify listener
        withContext(Dispatchers.Main) {
            processPinResult(pinResult)
        }
    }

    /**
     * Processes the classification result and notifies the listener
     */
    private fun processPinResult(pinResult: RawDigitClassifier.PinResult) {
        // Validation: Exactly 4 digits with status OK
        val validDigitSegments = pinResult.segmentResults.filter {
            it.isValidDigit && it.label.length == 1 && it.label[0].isDigit()
        }

        val pinLengthOk = pinResult.pin.length == EXPECTED_PIN_LENGTH
        val validCountOk = validDigitSegments.size == EXPECTED_PIN_LENGTH

        val result = PinRecognitionResult(
            pin = pinResult.pin,
            totalSegments = pinResult.totalSegments,
            validDigits = pinResult.validDigits,
            segmentResults = pinResult.segmentResults
        )

        when {
            pinResult.pin.isEmpty() -> {
                AppLog.d(TAG) { "No PIN detected (empty result)" }
                listener.onNoPinDetected()
            }
            !pinLengthOk || !validCountOk -> {
                AppLog.d(TAG) { "PIN uncertain: length=${pinResult.pin.length}, validDigits=${validDigitSegments.size}" }
                listener.onPinUnsure(result)
            }
            else -> {
                AppLog.i(TAG) { "PIN recognized: ${pinResult.pin}" }
                listener.onPinRecognized(result)
            }
        }
    }

    /**
     * Releases resources
     */
    fun release() {
        cancelRecording()
        audioSessionManager.release()
    }
}
