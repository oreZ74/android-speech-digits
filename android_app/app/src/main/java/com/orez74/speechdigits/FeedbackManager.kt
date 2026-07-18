package com.orez74.speechdigits

import android.content.Context
import android.content.SharedPreferences
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager

/**
 * Manages haptic and auditory feedback for recording events.
 *
 * Features:
 * - Short beep (DTMF tone, 50ms) for recording start/stop
 * - Vibration (30ms) synchronized with beep
 * - TalkBack-friendly (announcements handled by caller)
 * - Settings-aware (beep/vibration can be disabled by user)
 *
 * Thread-Safety: All methods are safe to call from UI thread.
 * Vibrator/ToneGenerator are thread-safe system services.
 *
 * @since 2026-01-06
 */
class FeedbackManager(
    private val context: Context,
    private val prefs: SharedPreferences
) {

    private val vibrator: Vibrator? by lazy {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val vibratorManager = context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as? VibratorManager
            vibratorManager?.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            context.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
        }
    }

    private val audioManager: AudioManager by lazy {
        context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
    }

    private val toneGenerator: ToneGenerator? by lazy {
        try {
            ToneGenerator(
                AudioManager.STREAM_NOTIFICATION,
                ToneGenerator.MAX_VOLUME / 2
            )
        } catch (e: Exception) {
            AppLog.w(TAG, e) { "ToneGenerator initialization failed: ${e.message}" }
            null
        }
    }

    companion object {
        private const val TAG = "FeedbackManager"

        // Settings keys
        const val PREF_FEEDBACK_SOUND_ENABLED = "pref_feedback_sound_enabled"
        const val PREF_FEEDBACK_VIBRATION_ENABLED = "pref_feedback_vibration_enabled"

        // Defaults
        const val DEFAULT_SOUND_ENABLED = true
        const val DEFAULT_VIBRATION_ENABLED = true

        // Durations
        private const val VIBRATION_DURATION_MS = 30L
        private const val BEEP_DURATION_MS = 50
    }

    /**
     * Checks if device is in silent/vibrate mode.
     * In silent mode, beeps should not play (vibration may still work).
     */
    fun isSilentMode(): Boolean {
        return audioManager.ringerMode == AudioManager.RINGER_MODE_SILENT
    }

    /**
     * Provides feedback for recording start.
     * Respects user settings and system state.
     */
    fun onRecordingStart() {
        val soundEnabled = prefs.getBoolean(PREF_FEEDBACK_SOUND_ENABLED, DEFAULT_SOUND_ENABLED)
        val vibrationEnabled = prefs.getBoolean(PREF_FEEDBACK_VIBRATION_ENABLED, DEFAULT_VIBRATION_ENABLED)

        if (vibrationEnabled) {
            vibrate()
        }
        if (soundEnabled && !isSilentMode()) {
            playBeep()
        }
        AppLog.d(TAG) { "Recording start feedback (sound=$soundEnabled, vib=$vibrationEnabled, silent=${isSilentMode()})" }
    }

    /**
     * Provides feedback for recording stop.
     * Respects user settings and system state.
     */
    fun onRecordingStop() {
        val soundEnabled = prefs.getBoolean(PREF_FEEDBACK_SOUND_ENABLED, DEFAULT_SOUND_ENABLED)
        val vibrationEnabled = prefs.getBoolean(PREF_FEEDBACK_VIBRATION_ENABLED, DEFAULT_VIBRATION_ENABLED)

        if (vibrationEnabled) {
            vibrate()
        }
        if (soundEnabled && !isSilentMode()) {
            playBeep()
        }
        AppLog.d(TAG) { "Recording stop feedback (sound=$soundEnabled, vib=$vibrationEnabled, silent=${isSilentMode()})" }
    }

    private fun vibrate() {
        vibrator?.let {
            try {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    val effect = VibrationEffect.createOneShot(
                        VIBRATION_DURATION_MS,
                        VibrationEffect.DEFAULT_AMPLITUDE
                    )
                    it.vibrate(effect)
                } else {
                    @Suppress("DEPRECATION")
                    it.vibrate(VIBRATION_DURATION_MS)
                }
            } catch (e: Exception) {
                AppLog.w(TAG, e) { "Vibration failed: ${e.message}" }
            }
        } ?: AppLog.d(TAG) { "Vibrator not available" }
    }

    private fun playBeep() {
        toneGenerator?.let {
            try {
                // Use DTMF_1 for short, distinct beep (~697Hz)
                it.startTone(ToneGenerator.TONE_DTMF_1, BEEP_DURATION_MS)
            } catch (e: Exception) {
                AppLog.w(TAG, e) { "Beep playback failed: ${e.message}" }
            }
        }
    }

    /**
     * Release audio resources.
     * Call in onDestroy() to avoid resource leaks.
     */
    fun release() {
        toneGenerator?.release()
        AppLog.d(TAG) { "Released audio resources" }
    }
}
