package com.orez74.speechdigits

import android.content.Context
import android.content.SharedPreferences
import android.os.Bundle
import android.view.MenuItem
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AppCompatDelegate
import androidx.core.os.LocaleListCompat
import com.github.orez74.speechdigits.R
import com.github.orez74.speechdigits.databinding.ActivitySettingsBinding

/**
 * Settings screen for feedback preferences, theme, and language selection.
 *
 * WCAG 2.2 AA compliant:
 * - All controls have proper labels and contentDescriptions
 * - Touch targets are >=48dp
 * - Focus order is logical
 * - Works with TalkBack
 */
class SettingsActivity : AppCompatActivity() {

    private lateinit var binding: ActivitySettingsBinding
    private lateinit var prefs: SharedPreferences

    private var lastAnnouncementText = ""
    private var lastAnnouncementTime = 0L

    private fun announceA11y(message: String) {
        if (message.isBlank()) return

        val now = System.currentTimeMillis()
        if (message == lastAnnouncementText && (now - lastAnnouncementTime) < 500) {
            return
        }

        lastAnnouncementText = message
        lastAnnouncementTime = now

        binding.a11yAnnouncement.post {
            binding.a11yAnnouncement.accessibilityLiveRegion = View.ACCESSIBILITY_LIVE_REGION_POLITE
            binding.a11yAnnouncement.text = ""
            binding.a11yAnnouncement.post {
                binding.a11yAnnouncement.text = message
                AppLog.d(TAG) { "A11y announced: \"$message\"" }
            }
        }
    }

    companion object {
        private const val TAG = "SettingsActivity"
        private const val PREFS_NAME = "audio_digit_prefs"

        const val PREF_THEME_MODE = "pref_theme_mode"

        const val THEME_DARK = AppCompatDelegate.MODE_NIGHT_YES
        const val THEME_LIGHT = AppCompatDelegate.MODE_NIGHT_NO
        const val THEME_SYSTEM = AppCompatDelegate.MODE_NIGHT_FOLLOW_SYSTEM
        const val DEFAULT_THEME = THEME_DARK

        fun applyTheme(context: Context) {
            val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            val mode = prefs.getInt(PREF_THEME_MODE, DEFAULT_THEME)
            AppCompatDelegate.setDefaultNightMode(mode)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        applyTheme(this)
        super.onCreate(savedInstanceState)
        binding = ActivitySettingsBinding.inflate(layoutInflater)
        setContentView(binding.root)

        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.title = getString(R.string.settings_title)

        prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

        initViews()
        loadSettings()
    }

    private fun initViews() {
        binding.switchSound.setOnCheckedChangeListener { _, isChecked ->
            prefs.edit()
                .putBoolean(FeedbackManager.PREF_FEEDBACK_SOUND_ENABLED, isChecked)
                .apply()
            AppLog.d(TAG) { "Sound feedback: $isChecked" }

            val announcement = if (isChecked) {
                getString(R.string.settings_sound_enabled)
            } else {
                getString(R.string.settings_sound_disabled)
            }
            announceA11y(announcement)
        }

        binding.switchVibration.setOnCheckedChangeListener { _, isChecked ->
            prefs.edit()
                .putBoolean(FeedbackManager.PREF_FEEDBACK_VIBRATION_ENABLED, isChecked)
                .apply()
            AppLog.d(TAG) { "Vibration feedback: $isChecked" }

            val announcement = if (isChecked) {
                getString(R.string.settings_vibration_enabled)
            } else {
                getString(R.string.settings_vibration_disabled)
            }
            announceA11y(announcement)
        }

        // Theme radio buttons
        val themeListener = { radioId: Int ->
            binding.radioThemeDark.isChecked = (radioId == R.id.radioThemeDark)
            binding.radioThemeLight.isChecked = (radioId == R.id.radioThemeLight)
            binding.radioThemeSystem.isChecked = (radioId == R.id.radioThemeSystem)

            val (mode, announcement) = when (radioId) {
                R.id.radioThemeLight -> THEME_LIGHT to getString(R.string.settings_theme_changed_light)
                R.id.radioThemeSystem -> THEME_SYSTEM to getString(R.string.settings_theme_changed_system)
                else -> THEME_DARK to getString(R.string.settings_theme_changed_dark)
            }

            prefs.edit().putInt(PREF_THEME_MODE, mode).apply()
            AppCompatDelegate.setDefaultNightMode(mode)
            announceA11y(announcement)
            AppLog.d(TAG) { "Theme changed to: $mode" }
        }

        binding.radioThemeDark.setOnClickListener { themeListener(R.id.radioThemeDark) }
        binding.radioThemeLight.setOnClickListener { themeListener(R.id.radioThemeLight) }
        binding.radioThemeSystem.setOnClickListener { themeListener(R.id.radioThemeSystem) }

        // Language radio buttons
        val languageListener = { radioId: Int ->
            binding.radioLanguageSystem.isChecked = (radioId == R.id.radioLanguageSystem)
            binding.radioLanguageGerman.isChecked = (radioId == R.id.radioLanguageGerman)
            binding.radioLanguageEnglish.isChecked = (radioId == R.id.radioLanguageEnglish)

            val (localeList, announcement) = when (radioId) {
                R.id.radioLanguageGerman -> LocaleListCompat.forLanguageTags("de") to getString(R.string.settings_language_changed_german)
                R.id.radioLanguageEnglish -> LocaleListCompat.forLanguageTags("en") to getString(R.string.settings_language_changed_english)
                else -> LocaleListCompat.getEmptyLocaleList() to getString(R.string.settings_language_changed_system)
            }

            AppCompatDelegate.setApplicationLocales(localeList)
            announceA11y(announcement)
            AppLog.d(TAG) { "Language changed to: ${if (localeList.isEmpty) "system" else localeList.toLanguageTags()}" }
        }

        binding.radioLanguageSystem.setOnClickListener { languageListener(R.id.radioLanguageSystem) }
        binding.radioLanguageGerman.setOnClickListener { languageListener(R.id.radioLanguageGerman) }
        binding.radioLanguageEnglish.setOnClickListener { languageListener(R.id.radioLanguageEnglish) }
    }

    private fun loadSettings() {
        binding.switchSound.isChecked = prefs.getBoolean(
            FeedbackManager.PREF_FEEDBACK_SOUND_ENABLED,
            FeedbackManager.DEFAULT_SOUND_ENABLED
        )
        binding.switchVibration.isChecked = prefs.getBoolean(
            FeedbackManager.PREF_FEEDBACK_VIBRATION_ENABLED,
            FeedbackManager.DEFAULT_VIBRATION_ENABLED
        )

        val themeMode = prefs.getInt(PREF_THEME_MODE, DEFAULT_THEME)
        when (themeMode) {
            THEME_LIGHT -> binding.radioThemeLight.isChecked = true
            THEME_SYSTEM -> binding.radioThemeSystem.isChecked = true
            else -> binding.radioThemeDark.isChecked = true
        }

        // Load current language setting from AppCompat's persisted state
        val currentLocales = AppCompatDelegate.getApplicationLocales()
        when {
            currentLocales.isEmpty -> binding.radioLanguageSystem.isChecked = true
            currentLocales.toLanguageTags().startsWith("en") -> binding.radioLanguageEnglish.isChecked = true
            currentLocales.toLanguageTags().startsWith("de") -> binding.radioLanguageGerman.isChecked = true
            else -> binding.radioLanguageSystem.isChecked = true
        }
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            android.R.id.home -> {
                finish()
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }
}
