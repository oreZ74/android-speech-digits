package com.orez74.speechdigits

import android.content.Context
import android.content.pm.ApplicationInfo
import android.util.Log

/**
 * Centralized logger for production-ready logging.
 *
 * Log-Level Strategy:
 * - DEBUG: Detailed technical info (debug builds only)
 * - INFO: Important events (model loaded, recording started/stopped)
 * - WARN: Unexpected but non-critical conditions
 * - ERROR: Actual errors with exceptions
 *
 * Usage:
 * ```kotlin
 * AppLog.d(TAG) { "expensive debug message: $value" }  // Debug builds only
 * AppLog.i(TAG) { "Model loaded successfully" }        // Always
 * AppLog.w(TAG) { "Audio very quiet, RMS=$rms" }       // Always
 * AppLog.e(TAG, exception) { "Failed to load" }        // Always
 * ```
 *
 * Lambda syntax prevents string concatenation in release builds.
 *
 * IMPORTANT: Call AppLog.init(context) once before any logging,
 * typically in Activity.onCreate().
 */
object AppLog {

    @Volatile
    private var initialized = false

    /**
     * Initialize debug mode from application flags.
     * Must be called once before any logging occurs.
     */
    fun init(context: Context) {
        if (initialized) return
        val flags = context.applicationContext.applicationInfo.flags
        DEBUG_ENABLED = (flags and ApplicationInfo.FLAG_DEBUGGABLE) != 0
        VERBOSE_ENABLED = DEBUG_ENABLED
        initialized = true
    }

    /**
     * Global debug flag.
     * In release builds: false -> all debug logs are suppressed.
     * In debug builds: true -> debug logs are emitted.
     */
    var DEBUG_ENABLED: Boolean = false
        private set

    /**
     * Verbose debug logging for hot-paths (audio loop, VAD, inference).
     * Can be disabled separately, even in debug builds.
     */
    var VERBOSE_ENABLED: Boolean = false

    /**
     * DEBUG level. Only active when [DEBUG_ENABLED] is true.
     * The lambda is not evaluated when debug is off.
     */
    inline fun d(tag: String, msg: () -> String) {
        if (DEBUG_ENABLED) {
            Log.d(tag, msg())
        }
    }

    /**
     * VERBOSE level for hot-paths. Only active when [VERBOSE_ENABLED] is true.
     */
    inline fun v(tag: String, msg: () -> String) {
        if (VERBOSE_ENABLED) {
            Log.v(tag, msg())
        }
    }

    /** INFO level. Always active (including release builds). */
    inline fun i(tag: String, msg: () -> String) {
        Log.i(tag, msg())
    }

    /** WARN level. Always active. Use for unexpected but non-critical states. */
    inline fun w(tag: String, msg: () -> String) {
        Log.w(tag, msg())
    }

    /** WARN level with throwable. Always active. */
    inline fun w(tag: String, t: Throwable, msg: () -> String) {
        Log.w(tag, msg(), t)
    }

    /** ERROR level. Always active. */
    inline fun e(tag: String, t: Throwable? = null, msg: () -> String) {
        if (t != null) {
            Log.e(tag, msg(), t)
        } else {
            Log.e(tag, msg())
        }
    }

    /**
     * ERROR-level with exception (alternative signature).
     */
    inline fun e(tag: String, msg: () -> String, t: Throwable) {
        Log.e(tag, msg(), t)
    }

    // Debug helpers

    /**
     * Logs a separator line (debug mode only).
     */
    fun separator(tag: String, char: Char = '=', length: Int = 60) {
        if (DEBUG_ENABLED) {
            Log.d(tag, char.toString().repeat(length))
        }
    }

    /**
     * Logs a banner block (debug mode only).
     */
    fun banner(tag: String, title: String, char: Char = '=', length: Int = 60) {
        if (DEBUG_ENABLED) {
            val line = char.toString().repeat(length)
            Log.d(tag, line)
            Log.d(tag, title)
            Log.d(tag, line)
        }
    }
}
