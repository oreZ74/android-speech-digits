package com.orez74.speechdigits

/**
 * UI state representation for the main screen.
 * Single source of truth owned by MainViewModel, observed by MainActivity.
 */
sealed class UiState {
    object Idle : UiState()
    object Listening : UiState()
    object Processing : UiState()
    data class PinDetected(
        val pin: String,
        val totalSegments: Int,
        val validDigits: Int,
        val segmentResults: List<RawDigitClassifier.SegmentResult>
    ) : UiState()
    data class Error(
        val type: ErrorType,
        val message: String,
        val rms: Float = 0f,
        val peak: Int = 0
    ) : UiState()
}

enum class ErrorType {
    TOO_QUIET,
    NO_PIN,
    UNSURE_RECOGNITION,
    PERMISSION_DENIED,
    MODEL_ERROR,
    UNKNOWN
}
