package com.orez74.speechdigits

import android.content.Context
import android.view.View
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityManager
import android.widget.TextView

/**
 * Central accessibility helper for Live Region announcements and TalkBack focus control.
 *
 * Replaces scattered announceForAccessibility() calls with rate-limited, thread-safe
 * Live Region updates. Provides TalkBack focus management for the copy button.
 *
 * Features:
 * - Rate-limiting: identical messages within 500ms are ignored
 * - Thread-safe: post {} on View
 * - Privacy: no PIN in contentDescription
 */
class A11yAnnouncementHelper(
    private val liveRegionView: TextView
) {
    private var lastText = ""
    private var lastTime = 0L
    private var lastFocusKey: String? = null

    /**
     * Central accessibility announcement via Live Region.
     *
     * @param message The message to announce
     * @param mode POLITE (default) or ASSERTIVE for errors
     */
    fun announce(message: String, mode: Mode = Mode.POLITE) {
        if (message.isBlank()) return

        val now = System.currentTimeMillis()
        if (message == lastText && (now - lastTime) < 500) {
            AppLog.d(TAG) { "A11y: Rate-limited duplicate: \"$message\"" }
            return
        }

        lastText = message
        lastTime = now

        val liveRegionMode = when (mode) {
            Mode.ASSERTIVE -> View.ACCESSIBILITY_LIVE_REGION_ASSERTIVE
            Mode.POLITE -> View.ACCESSIBILITY_LIVE_REGION_POLITE
        }

        liveRegionView.post {
            liveRegionView.accessibilityLiveRegion = liveRegionMode
            liveRegionView.text = ""
            liveRegionView.post {
                liveRegionView.text = message
                AppLog.d(TAG) { "A11y announced (${mode.name}): \"$message\"" }
            }
        }
    }

    /**
     * Checks whether TalkBack is active (Touch Exploration).
     */
    fun isTalkBackActive(context: Context): Boolean {
        val am = context.getSystemService(Context.ACCESSIBILITY_SERVICE) as? AccessibilityManager
        return am?.isEnabled == true && am.isTouchExplorationEnabled
    }

    /**
     * Sets focus on Copy button for TalkBack users after PIN recognition.
     * Only executed when TalkBack is active and a new result is available.
     *
     * @param context The application/activity context
     * @param copyButton The copy button view
     * @param pin The recognized PIN (as focus key)
     */
    fun focusCopyButton(context: Context, copyButton: View, pin: String) {
        if (pin == lastFocusKey) {
            AppLog.d(TAG) { "Focus-Shift skipped: Same result key" }
            return
        }

        if (!isTalkBackActive(context)) {
            AppLog.d(TAG) { "Focus-Shift skipped: TalkBack not active" }
            return
        }

        lastFocusKey = pin

        copyButton.post {
            copyButton.post {
                if (copyButton.isShown && copyButton.isEnabled) {
                    val focusSuccess = copyButton.requestFocus()
                    copyButton.sendAccessibilityEvent(AccessibilityEvent.TYPE_VIEW_FOCUSED)

                    AppLog.d(TAG) {
                        "Focus-Shift to Copy button: requestFocus=$focusSuccess, shown=${copyButton.isShown}, enabled=${copyButton.isEnabled}"
                    }
                } else {
                    AppLog.w(TAG) {
                        "Copy button not ready for focus (shown=${copyButton.isShown}, enabled=${copyButton.isEnabled})"
                    }
                }
            }
        }
    }

    enum class Mode {
        POLITE,
        ASSERTIVE
    }

    companion object {
        private const val TAG = "A11yHelper"
    }
}
