package com.orez74.speechdigits

import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Tests the UiState logic and StateFlow behavior isolated from Android dependencies.
 *
 * A minimal ViewModel stub implements PinEntryListener to verify callback-to-state mappings.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class MainViewModelTest {

    private class ViewModelStub : PinEntryController.PinEntryListener {
        private val _uiState = MutableStateFlow<UiState>(UiState.Idle)
        val uiState: StateFlow<UiState> = _uiState.asStateFlow()

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
            _uiState.value = UiState.Error(type = ErrorType.NO_PIN, message = "")
        }

        override fun onPinUnsure(result: PinEntryController.PinRecognitionResult) {
            _uiState.value = UiState.Error(type = ErrorType.UNSURE_RECOGNITION, message = "")
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
            // No state change needed
        }
    }

    @Test
    fun initialState_isIdle() = runTest {
        val stub = ViewModelStub()
        assertEquals(UiState.Idle, stub.uiState.value)
    }

    @Test
    fun onPinRecordingStarted_setsStateToListening() = runTest {
        val stub = ViewModelStub()
        stub.onPinRecordingStarted()
        assertEquals(UiState.Listening, stub.uiState.value)
    }

    @Test
    fun onPinRecordingProcessing_setsStateToProcessing() = runTest {
        val stub = ViewModelStub()
        stub.onPinRecordingProcessing()
        assertEquals(UiState.Processing, stub.uiState.value)
    }

    @Test
    fun onPinRecognized_setsStateToPinDetected_withCorrectPin() = runTest {
        val stub = ViewModelStub()
        stub.onPinRecognized(
            PinEntryController.PinRecognitionResult(
                pin = "1234", totalSegments = 4, validDigits = 4,
                segmentResults = emptyList()
            )
        )
        assertTrue(stub.uiState.value is UiState.PinDetected)
        val pinDetected = stub.uiState.value as UiState.PinDetected
        assertEquals("1234", pinDetected.pin)
        assertEquals(4, pinDetected.totalSegments)
        assertEquals(4, pinDetected.validDigits)
    }

    @Test
    fun onNoPinDetected_setsStateToError_noPin() = runTest {
        val stub = ViewModelStub()
        stub.onNoPinDetected()
        assertTrue(stub.uiState.value is UiState.Error)
        assertEquals(ErrorType.NO_PIN, (stub.uiState.value as UiState.Error).type)
    }

    @Test
    fun onPinUnsure_setsStateToError_unsureRecognition() = runTest {
        val stub = ViewModelStub()
        stub.onPinUnsure(
            PinEntryController.PinRecognitionResult(
                pin = "12x", totalSegments = 3, validDigits = 2,
                segmentResults = emptyList()
            )
        )
        assertTrue(stub.uiState.value is UiState.Error)
        assertEquals(ErrorType.UNSURE_RECOGNITION, (stub.uiState.value as UiState.Error).type)
    }

    @Test
    fun onPinError_classifierNotReady_setsStateToError_withUnknown() = runTest {
        val stub = ViewModelStub()
        stub.onPinError(PinEntryController.PinError.ClassifierNotReady("model broken"))
        val error = stub.uiState.value as UiState.Error
        assertEquals(ErrorType.UNKNOWN, error.type)
        assertEquals("model broken", error.message)
    }

    @Test
    fun stateFlow_emitsAllStates_inSequence() = runTest {
        val stub = ViewModelStub()
        val states = mutableListOf<UiState>()

        val job = launch {
            stub.uiState.collect { states.add(it) }
        }

        // Let initial state be collected
        testScheduler.advanceUntilIdle()

        stub.onPinRecordingStarted()
        testScheduler.advanceUntilIdle()

        stub.onPinRecordingProcessing()
        testScheduler.advanceUntilIdle()

        stub.onPinRecognized(
            PinEntryController.PinRecognitionResult(
                pin = "9999", totalSegments = 4, validDigits = 4,
                segmentResults = emptyList()
            )
        )
        testScheduler.advanceUntilIdle()

        job.cancel()

        assertTrue("Expected at least initial + 3 states, got ${states.size}", states.size >= 4)
        assertEquals(UiState.Idle, states[0])
        assertTrue(states.any { it is UiState.Listening })
        assertTrue(states.any { it is UiState.Processing })
        assertTrue(states.any { it is UiState.PinDetected })
    }
}
