package com.orez74.speechdigits

import android.Manifest
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.KeyEvent
import android.view.Menu
import android.view.MenuItem
import android.view.View
import android.widget.Toast
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.github.orez74.speechdigits.R
import com.github.orez74.speechdigits.databinding.ActivityMainBinding
import kotlinx.coroutines.launch

/**
 * MainActivity for German digit recognition (0-9).
 *
 * Observes MainViewModel.uiState via StateFlow (MVVM pattern).
 * Delegates UI animations, accessibility, and debug rendering to helper classes.
 * State survives configuration changes via ViewModel.
 */
class MainActivity : AppCompatActivity() {

    // ==========================================
    // View Binding
    // ==========================================

    private lateinit var binding: ActivityMainBinding
    private var lastRecognizedPin: String? = null
    private var hasCopied = false

    // ==========================================
    // ViewModel
    // ==========================================

    private val viewModel: MainViewModel by viewModels()

    // ==========================================
    // Extracted Helpers
    // ==========================================

    private lateinit var micPulseAnimator: MicPulseAnimator
    private lateinit var a11yHelper: A11yAnnouncementHelper
    private lateinit var debugPanel: DebugPanelController
    private lateinit var feedbackManager: FeedbackManager
    private lateinit var prefs: SharedPreferences

    companion object {
        private const val TAG = "MainActivity"
        private const val PERMISSION_REQUEST_CODE = 100
        private const val PREFS_NAME = "audio_digit_prefs"
        private const val PREF_PIN_DURATION_MS = "pref_pin_duration_ms"
        private const val DEFAULT_PIN_DURATION_MS = 4000

        val DURATION_OPTIONS_MS = intArrayOf(4000, 6000, 8000, 10000)
    }

    private val pinRecordingDurationMs: Int
        get() = prefs.getInt(PREF_PIN_DURATION_MS, DEFAULT_PIN_DURATION_MS)

    private fun getDurationLabels(): Array<String> = arrayOf(
        getString(R.string.duration_4s),
        getString(R.string.duration_6s),
        getString(R.string.duration_8s),
        getString(R.string.duration_10s)
    )

    // ==========================================
    // Lifecycle
    // ==========================================

    override fun onCreate(savedInstanceState: Bundle?) {
        SettingsActivity.applyTheme(this)
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        AppLog.init(this)
        prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

        initHelpers()
        initViews()
        observeUiState()
        checkPermissions()

        binding.btnModePin.post {
            binding.btnModePin.requestFocus()
        }
    }

    private fun initHelpers() {
        micPulseAnimator = MicPulseAnimator(
            outerRing = binding.micOuterRing,
            middleRing = binding.micMiddleRing
        )
        micPulseAnimator.setup()

        a11yHelper = A11yAnnouncementHelper(
            liveRegionView = binding.a11yAnnouncement
        )

        debugPanel = DebugPanelController(
            context = this,
            debugCard = binding.debugCard,
            debugContentContainer = binding.debugContentContainer,
            ivDebugArrow = binding.ivDebugArrow,
            textDebugSummary = binding.textDebugSummary,
            textDebugAudio = binding.textDebugAudio,
            textDebugResult = binding.textDebugResult,
            llProbabilities = binding.llProbabilities
        )
        debugPanel.setup()

        feedbackManager = FeedbackManager(this, prefs)
    }

    private fun initViews() {
        setupClickListeners()
        updateAppInfoText()
        updateLanguageHintVisibility()
        applyFontScaleLayout()
        debugPanel.collapse()
    }

    private fun observeUiState() {
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                var previous: UiState? = null
                viewModel.uiState.collect { state ->
                    // Feedback on transitions
                    if (previous !is UiState.Listening && state is UiState.Listening) {
                        feedbackManager.onRecordingStart()
                    }
                    if (previous is UiState.Listening && state is UiState.Processing) {
                        feedbackManager.onRecordingStop()
                    }
                    previous = state
                    renderState(state)
                }
            }
        }
    }

    // ==========================================
    // UI Rendering (driven by StateFlow)
    // ==========================================

    private fun renderState(state: UiState) {
        when (state) {
            is UiState.Idle -> showIdleState()
            is UiState.Listening -> showListeningState()
            is UiState.Processing -> showProcessingState()
            is UiState.PinDetected -> showPinDetectedState(state)
            is UiState.Error -> showErrorState(state)
        }
    }

    private fun showIdleState() {
        micPulseAnimator.stop()

        binding.textStatus.text = getString(R.string.status_ready)
        binding.textStatus.setTextColor(ContextCompat.getColor(this, R.color.status_ready))
        binding.textInstruction.text = getString(R.string.instruction_press_button)
        binding.textInstruction.visibility = View.VISIBLE

        hideResultViews()
        resetButton()

        lastRecognizedPin = null
        hasCopied = false

        binding.btnModePin.post {
            binding.btnModePin.requestFocus()
        }
    }

    private fun showListeningState() {
        micPulseAnimator.start()

        binding.textStatus.text = getString(R.string.status_listening)
        binding.textStatus.setTextColor(ContextCompat.getColor(this, R.color.status_listening))

        val durationSeconds = pinRecordingDurationMs / 1000
        binding.textInstruction.text = getString(R.string.instruction_speak_pin_dynamic, durationSeconds)
        binding.textInstruction.visibility = View.VISIBLE

        hideResultViews()

        binding.btnModePin.isEnabled = true
        binding.tvButtonLabel.text = getString(R.string.btn_recording)
        binding.btnModePin.contentDescription = getString(R.string.cd_recording_stop)
        ViewCompat.setStateDescription(binding.btnModePin, getString(R.string.state_recording))
    }

    private fun showProcessingState() {
        binding.textStatus.text = getString(R.string.status_processing)
        binding.textStatus.setTextColor(ContextCompat.getColor(this, R.color.status_processing))
        binding.textInstruction.text = getString(R.string.instruction_wait)
        binding.textInstruction.visibility = View.VISIBLE
    }

    private fun showPinDetectedState(state: UiState.PinDetected) {
        micPulseAnimator.stop()

        binding.textStatus.text = getString(R.string.status_pin_detected)
        binding.textStatus.setTextColor(ContextCompat.getColor(this, R.color.status_success))

        val pinWithSpaces = state.pin.toCharArray().joinToString(" ")
        binding.textPinResult.text = pinWithSpaces
        binding.textPinResult.visibility = View.VISIBLE

        binding.textInstruction.visibility = View.GONE

        setupCopyButton(state.pin)
        resetButton()
        debugPanel.updatePinDebugPanel(state)

        a11yHelper.announce(getString(R.string.a11y_pin_detected_copy), A11yAnnouncementHelper.Mode.POLITE)
        a11yHelper.focusCopyButton(this, binding.btnCopyResult, state.pin)
    }

    private fun showErrorState(state: UiState.Error) {
        micPulseAnimator.stop()

        val (statusText, statusColor, instructionText) = getErrorDisplay(state)

        binding.textStatus.text = statusText
        binding.textStatus.setTextColor(ContextCompat.getColor(this, statusColor))
        binding.textInstruction.text = instructionText
        binding.textInstruction.visibility = View.VISIBLE

        hideResultViews()
        resetButton()

        a11yHelper.announce(instructionText, A11yAnnouncementHelper.Mode.ASSERTIVE)

        binding.scrollView.post {
            binding.scrollView.smoothScrollTo(0, binding.resultCard.top)
        }

        if (state.rms > 0) {
            binding.textDebugSummary.text = getString(R.string.debug_error_rms_peak, state.rms.toInt(), state.peak)
        }
    }

    // ==========================================
    // UI Helpers
    // ==========================================

    private fun hideResultViews() {
        binding.textPinResult.visibility = View.GONE
        binding.btnCopyResult.visibility = View.GONE
        binding.textCopySuccess.visibility = View.GONE
    }

    private fun resetButton() {
        binding.btnModePin.isEnabled = true
        binding.tvButtonLabel.text = getString(R.string.btn_pin_mode)
        binding.btnModePin.contentDescription = getString(R.string.cd_mic_button_idle)
        ViewCompat.setStateDescription(binding.btnModePin, null)
    }

    private fun setupCopyButton(pin: String) {
        lastRecognizedPin = pin
        hasCopied = false
        binding.btnCopyResult.text = getString(R.string.btn_copy_pin)
        binding.btnCopyResult.contentDescription = getString(R.string.cd_copy_pin)
        binding.btnCopyResult.isEnabled = true
        binding.btnCopyResult.visibility = View.VISIBLE
        binding.textCopySuccess.visibility = View.GONE
    }

    private fun getErrorDisplay(state: UiState.Error): Triple<String, Int, String> {
        return when (state.type) {
            ErrorType.TOO_QUIET -> Triple(
                getString(R.string.status_too_quiet),
                R.color.status_too_quiet,
                getString(R.string.error_too_quiet_closer)
            )
            ErrorType.NO_PIN -> Triple(
                getString(R.string.status_no_pin),
                R.color.status_warning,
                getString(R.string.error_no_pin_speak_again)
            )
            ErrorType.PERMISSION_DENIED -> Triple(
                getString(R.string.status_error),
                R.color.status_error,
                getString(R.string.error_permission)
            )
            ErrorType.MODEL_ERROR -> Triple(
                getString(R.string.status_error),
                R.color.status_error,
                getString(R.string.error_model_load)
            )
            ErrorType.UNSURE_RECOGNITION -> Triple(
                getString(R.string.status_unsure),
                R.color.status_warning,
                getString(R.string.error_unsure_recognition)
            )
            ErrorType.UNKNOWN -> Triple(
                getString(R.string.status_error),
                R.color.status_error,
                state.message
            )
        }
    }

    private fun updateLanguageHintVisibility() {
        val hint = getString(R.string.input_language_hint)
        binding.tvInputLanguageHint.visibility = if (hint.isBlank()) View.GONE else View.VISIBLE
    }

    private fun applyFontScaleLayout() {
        val fontScale = resources.configuration.fontScale
        val density = resources.displayMetrics.density

        val dims = when {
            fontScale > 1.5f -> LayoutDimensions(180, 160, 170, 140, 48)
            fontScale > 1.3f -> LayoutDimensions(200, 170, 185, 150, 52)
            else -> LayoutDimensions(220, 180, 200, 160, 56)
        }

        val infoScale = fontScale.coerceIn(1.0f, 1.4f)
        val infoButtonSize = (48 * infoScale * density).toInt()
        val infoIconSize = (24 * infoScale * density).toInt()
        binding.btnInfo.layoutParams = binding.btnInfo.layoutParams.apply {
            width = infoButtonSize
            height = infoButtonSize
        }
        binding.btnInfo.minWidth = infoButtonSize
        binding.btnInfo.minHeight = infoButtonSize
        binding.btnInfo.iconSize = infoIconSize

        binding.micButtonContainer.layoutParams = binding.micButtonContainer.layoutParams.apply {
            width = (dims.container * density).toInt()
            height = (dims.container * density).toInt()
        }
        binding.btnModePin.layoutParams = binding.btnModePin.layoutParams.apply {
            width = (dims.button * density).toInt()
            height = (dims.button * density).toInt()
        }
        binding.btnModePin.minWidth = (dims.button * density).toInt()
        binding.btnModePin.minHeight = (dims.button * density).toInt()
        binding.btnModePin.cornerRadius = (dims.button / 2 * density).toInt()

        binding.micOuterRing.layoutParams = binding.micOuterRing.layoutParams.apply {
            width = (dims.outerRing * density).toInt()
            height = (dims.outerRing * density).toInt()
        }
        binding.micMiddleRing.layoutParams = binding.micMiddleRing.layoutParams.apply {
            width = (dims.middleRing * density).toInt()
            height = (dims.middleRing * density).toInt()
        }
        binding.ivMicIcon.layoutParams = binding.ivMicIcon.layoutParams.apply {
            width = (dims.iconSize * density).toInt()
            height = (dims.iconSize * density).toInt()
        }

        AppLog.d(TAG) { "FontScale: $fontScale -> Button: ${dims.button}dp, Container: ${dims.container}dp" }
    }

    private fun setupClickListeners() {
        binding.btnModePin.setOnClickListener {
            if (viewModel.isRecordingActive()) {
                cancelRecording()
            } else {
                startPinRecording()
            }
        }
        binding.btnCopyResult.setOnClickListener { handleCopyClick() }
        binding.btnDurationSettings.setOnClickListener { showDurationSettingsDialog() }
        binding.btnInfo.setOnClickListener { showHelpDialog() }
        binding.debugToggleContainer.setOnClickListener { debugPanel.toggle() }
    }

    override fun onDestroy() {
        super.onDestroy()
        micPulseAnimator.release()
        if (::feedbackManager.isInitialized) {
            feedbackManager.release()
        }
    }

    // ==========================================
    // Permissions
    // ==========================================

    private fun checkPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.RECORD_AUDIO),
                PERMISSION_REQUEST_CODE
            )
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == PERMISSION_REQUEST_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                AppLog.i(TAG) { "Microphone permission granted" }
                Toast.makeText(this, getString(R.string.toast_permission_granted), Toast.LENGTH_SHORT).show()
            } else {
                AppLog.e(TAG) { "Microphone permission denied" }
                // State will be set by renderState via the collected flow
            }
        }
    }

    // ==========================================
    // PIN Recording (delegated to ViewModel)
    // ==========================================

    private fun startPinRecording() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(this, getString(R.string.error_permission), Toast.LENGTH_SHORT).show()
            checkPermissions()
            return
        }
        viewModel.startPinRecording(pinRecordingDurationMs)
    }

    private fun cancelRecording() {
        if (viewModel.isRecordingActive()) {
            viewModel.cancelRecording()
            feedbackManager.onRecordingStop()
            micPulseAnimator.stop()
            showIdleState()
            AppLog.i(TAG) { "Recording cancelled" }
        }
    }

    // ==========================================
    // Dialogs & Menus
    // ==========================================

    private fun showDurationSettingsDialog() {
        val currentDuration = pinRecordingDurationMs
        val currentIndex = DURATION_OPTIONS_MS.indexOf(currentDuration).coerceAtLeast(0)

        MaterialAlertDialogBuilder(this, R.style.ThemeOverlay_App_MaterialAlertDialog)
            .setTitle(getString(R.string.dialog_duration_title))
            .setSingleChoiceItems(getDurationLabels(), currentIndex) { dialog, which ->
                val selectedDuration = DURATION_OPTIONS_MS[which]
                prefs.edit().putInt(PREF_PIN_DURATION_MS, selectedDuration).apply()
                updateAppInfoText()

                val durationSeconds = selectedDuration / 1000
                val announcement = getString(R.string.a11y_duration_changed, durationSeconds)
                a11yHelper.announce(announcement, A11yAnnouncementHelper.Mode.POLITE)

                Toast.makeText(
                    this,
                    getString(R.string.toast_duration_changed, getDurationLabels()[which]),
                    Toast.LENGTH_SHORT
                ).show()
                dialog.dismiss()
            }
            .setNegativeButton(getString(R.string.dialog_cancel), null)
            .show()
    }

    private fun showHelpDialog() {
        MaterialAlertDialogBuilder(this, R.style.ThemeOverlay_App_MaterialAlertDialog)
            .setTitle(getString(R.string.dialog_help_title))
            .setMessage(getString(R.string.dialog_help_content))
            .setPositiveButton(getString(R.string.dialog_help_close), null)
            .show()
    }

    private fun showDeveloperMenu() {
        val options = arrayOf("Dataset Recorder", "Run Inference Benchmark", "Abbrechen")

        android.app.AlertDialog.Builder(this)
            .setTitle("Developer Tools")
            .setItems(options) { dialog, which ->
                when (which) {
                    0 -> startActivity(Intent(this, DigitDatasetRecorderActivity::class.java))
                    1 -> runInferenceBenchmark()
                }
                dialog.dismiss()
            }
            .show()
    }

    private fun runInferenceBenchmark() {
        Toast.makeText(this, "Running benchmark...", Toast.LENGTH_SHORT).show()

        viewModel.runBenchmark { result, filePath ->
            val message = "Median: ${"%.2f".format(result.medianMs)}ms\n" +
                          "P95: ${"%.2f".format(result.p95Ms)}ms\n" +
                          "Device: ${result.deviceModel}\n" +
                          "Saved: ${filePath ?: "failed"}"

            android.app.AlertDialog.Builder(this)
                .setTitle("Benchmark Result")
                .setMessage(message)
                .setPositiveButton("OK", null)
                .show()
        }
    }

    // ==========================================
    // Misc Actions
    // ==========================================

    private fun updateAppInfoText() {
        val durationSeconds = pinRecordingDurationMs / 1000
        val text = getString(R.string.app_info_dynamic, durationSeconds)
        binding.btnDurationSettings.text = text
        val contentDesc = getString(R.string.a11y_duration_changed, durationSeconds)
        binding.btnDurationSettings.contentDescription = contentDesc
    }

    private fun handleCopyClick() {
        val textToCopy = lastRecognizedPin
        if (textToCopy.isNullOrEmpty()) return

        copyToClipboard(textToCopy)
        hasCopied = true
        binding.textCopySuccess.visibility = View.VISIBLE
        a11yHelper.announce(getString(R.string.toast_copied_to_clipboard), A11yAnnouncementHelper.Mode.POLITE)
    }

    private fun copyToClipboard(text: String) {
        val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        val clip = ClipData.newPlainText("PIN", text)
        clipboard.setPrimaryClip(clip)
        AppLog.d(TAG) { "\"$text\" copied to clipboard" }
    }

    override fun onCreateOptionsMenu(menu: Menu?): Boolean {
        menuInflater.inflate(R.menu.main_menu, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            R.id.action_settings -> {
                startActivity(Intent(this, SettingsActivity::class.java))
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }

    // ==========================================
    // Hardware Key Support
    // ==========================================

    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        if (event?.repeatCount != 0) {
            return when (keyCode) {
                KeyEvent.KEYCODE_VOLUME_UP, KeyEvent.KEYCODE_VOLUME_DOWN -> true
                else -> super.onKeyDown(keyCode, event)
            }
        }
        return when (keyCode) {
            KeyEvent.KEYCODE_VOLUME_UP -> {
                if (!viewModel.isRecordingActive()) {
                    binding.textStatus.text = getString(R.string.status_hardware_key_triggered)
                    startPinRecording()
                }
                true
            }
            KeyEvent.KEYCODE_VOLUME_DOWN -> {
                if (viewModel.isRecordingActive()) {
                    cancelRecording()
                    binding.textStatus.text = getString(R.string.status_recording_cancelled)
                    a11yHelper.announce(getString(R.string.status_recording_cancelled), A11yAnnouncementHelper.Mode.POLITE)
                }
                true
            }
            else -> super.onKeyDown(keyCode, event)
        }
    }
}
