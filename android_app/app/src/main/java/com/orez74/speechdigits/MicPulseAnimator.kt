package com.orez74.speechdigits

import android.animation.AnimatorSet
import android.animation.ObjectAnimator
import android.animation.ValueAnimator
import android.view.View
import android.view.animation.LinearInterpolator

/**
 * Manages the two concentric pulse ring animations around the microphone button.
 *
 * Usage:
 * - Call setup() once after view binding is ready
 * - Call start() / stop() to control the animation
 */
class MicPulseAnimator(
    private val outerRing: View,
    private val middleRing: View
) {
    private lateinit var outerPulseAnimator: AnimatorSet
    private lateinit var middlePulseAnimator: AnimatorSet
    private var isPulsing = false

    val running: Boolean
        get() = isPulsing

    fun setup() {
        val outerScaleX = ObjectAnimator.ofFloat(outerRing, View.SCALE_X, 0.9f, 1.4f).apply {
            repeatCount = ValueAnimator.INFINITE
            repeatMode = ValueAnimator.RESTART
        }
        val outerScaleY = ObjectAnimator.ofFloat(outerRing, View.SCALE_Y, 0.9f, 1.4f).apply {
            repeatCount = ValueAnimator.INFINITE
            repeatMode = ValueAnimator.RESTART
        }
        val outerAlpha = ObjectAnimator.ofFloat(outerRing, View.ALPHA, 0f, 0.4f, 0f).apply {
            repeatCount = ValueAnimator.INFINITE
            repeatMode = ValueAnimator.RESTART
        }

        outerPulseAnimator = AnimatorSet().apply {
            playTogether(outerScaleX, outerScaleY, outerAlpha)
            duration = 1600
            interpolator = LinearInterpolator()
        }

        val middleScaleX = ObjectAnimator.ofFloat(middleRing, View.SCALE_X, 0.95f, 1.35f).apply {
            repeatCount = ValueAnimator.INFINITE
            repeatMode = ValueAnimator.RESTART
        }
        val middleScaleY = ObjectAnimator.ofFloat(middleRing, View.SCALE_Y, 0.95f, 1.35f).apply {
            repeatCount = ValueAnimator.INFINITE
            repeatMode = ValueAnimator.RESTART
        }
        val middleAlpha = ObjectAnimator.ofFloat(middleRing, View.ALPHA, 0f, 0.35f, 0f).apply {
            repeatCount = ValueAnimator.INFINITE
            repeatMode = ValueAnimator.RESTART
        }

        middlePulseAnimator = AnimatorSet().apply {
            playTogether(middleScaleX, middleScaleY, middleAlpha)
            duration = 1400
            startDelay = 350
            interpolator = LinearInterpolator()
        }

        hideRings()
    }

    fun start() {
        if (isPulsing) return

        outerRing.visibility = View.VISIBLE
        middleRing.visibility = View.VISIBLE

        outerRing.scaleX = 0.9f
        outerRing.scaleY = 0.9f
        outerRing.alpha = 0f

        middleRing.scaleX = 0.95f
        middleRing.scaleY = 0.95f
        middleRing.alpha = 0f

        outerPulseAnimator.start()
        middlePulseAnimator.start()

        isPulsing = true
        AppLog.v(TAG) { "Pulse animation started" }
    }

    fun stop() {
        if (!isPulsing) return

        outerPulseAnimator.cancel()
        outerPulseAnimator.removeAllListeners()
        middlePulseAnimator.cancel()
        middlePulseAnimator.removeAllListeners()

        hideRings()

        isPulsing = false
        AppLog.v(TAG) { "Pulse animation stopped" }
    }

    fun release() {
        if (::outerPulseAnimator.isInitialized) {
            outerPulseAnimator.cancel()
            outerPulseAnimator.removeAllListeners()
        }
        if (::middlePulseAnimator.isInitialized) {
            middlePulseAnimator.cancel()
            middlePulseAnimator.removeAllListeners()
        }
    }

    private fun hideRings() {
        outerRing.apply {
            clearAnimation()
            scaleX = 0.9f
            scaleY = 0.9f
            alpha = 0f
            visibility = View.GONE
        }
        middleRing.apply {
            clearAnimation()
            scaleX = 0.95f
            scaleY = 0.95f
            alpha = 0f
            visibility = View.GONE
        }
    }

    companion object {
        private const val TAG = "MicPulseAnimator"
    }
}
