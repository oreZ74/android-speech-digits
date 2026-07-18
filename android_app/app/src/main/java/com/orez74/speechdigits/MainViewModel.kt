package com.orez74.speechdigits

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * ViewModel for the main PIN entry screen.
 *
 * Owns the classification pipeline and exposes UiState via StateFlow.
 * Survives configuration changes (screen rotation etc.).
 */
class MainViewModel(application: Application) : AndroidViewModel(application),
    PinEntryController.PinEntryListener {

    val classifier = RawDigitClassifier(application)
    private val audioSessionManager = AudioSessionManager()
    private val pinEntryController: PinEntryController

    private val _uiState = MutableStateFlow<UiState>(UiState.Idle)
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    companion object {
        private const val TAG = "MainViewModel"
    }

    init {
        pinEntryController = PinEntryController(
            audioSessionManager = audioSessionManager,
            classifier = classifier,
            listener = this,
            scope = viewModelScope
        )
    }

    fun startPinRecording(durationMs: Int): Boolean {
        return pinEntryController.startPinRecording(durationMs)
    }

    fun cancelRecording() {
        pinEntryController.cancelRecording()
    }

    fun isRecordingActive(): Boolean = pinEntryController.isRecordingActive()

    fun runBenchmark(onResult: (RawDigitClassifier.BenchmarkResult, String?) -> Unit) {
        viewModelScope.launch(Dispatchers.IO) {
            val result = classifier.runInferenceBenchmark()
            val filePath = classifier.runAndSaveBenchmark()
            onResult(result, filePath)
        }
    }

    // ==========================================
    // PinEntryListener implementation
    // ==========================================

    override fun onPinRecordingStarted() {
        _uiState.value = UiState.Listening
    }

    override fun onPinRecordingProcessing() {
        _uiState.value = UiState.Processing
    }

    override fun onPinRecognized(result: PinEntryController.PinRecognitionResult) {
        _uiState.value = UiState.PinDetected(
            pin = result.pin,
            totalSegments = result.totalSegments,
            validDigits = result.validDigits,
            segmentResults = result.segmentResults
        )
    }

    override fun onNoPinDetected() {
        _uiState.value = UiState.Error(
            type = ErrorType.NO_PIN,
            message = ""
        )
    }

    override fun onPinUnsure(result: PinEntryController.PinRecognitionResult) {
        _uiState.value = UiState.Error(
            type = ErrorType.UNSURE_RECOGNITION,
            message = ""
        )
    }

    override fun onPinError(error: PinEntryController.PinError) {
        _uiState.value = UiState.Error(
            type = ErrorType.UNKNOWN,
            message = when (error) {
                is PinEntryController.PinError.ClassifierNotReady -> error.message
                is PinEntryController.PinError.RecordingFailed -> error.message
                is PinEntryController.PinError.ClassificationFailed -> error.message
            }
        )
    }

    override fun onPinRecordingComplete() {
        // Recording finished - UI state already set by other callbacks
    }

    override fun onCleared() {
        pinEntryController.release()
        classifier.close()
    }
}
