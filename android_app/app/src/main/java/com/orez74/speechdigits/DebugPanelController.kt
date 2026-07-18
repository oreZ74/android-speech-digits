package com.orez74.speechdigits

import android.content.Context
import android.view.View
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.content.ContextCompat
import com.github.orez74.speechdigits.R


/**
 * Manages the debug panel UI: toggle, collapse, and per-segment probability display.
 *
 * The panel shows segment-level classification details during development/debugging.
 */
class DebugPanelController(
    private val context: Context,
    private val debugCard: View,
    private val debugContentContainer: View,
    private val ivDebugArrow: ImageView,
    private val textDebugSummary: TextView,
    private val textDebugAudio: TextView,
    private val textDebugResult: TextView,
    private val llProbabilities: LinearLayout
) {
    private var isExpanded = false

    fun setup() {
        debugCard.visibility = View.GONE
    }

    fun toggle() {
        isExpanded = !isExpanded
        debugContentContainer.visibility = if (isExpanded) View.VISIBLE else View.GONE
        updateArrow()

        ivDebugArrow.contentDescription = if (isExpanded)
            context.getString(R.string.debug_collapse)
        else
            context.getString(R.string.debug_expand)
    }

    fun collapse() {
        isExpanded = false
        updateArrow()
        debugContentContainer.visibility = View.GONE
    }

    private fun updateArrow() {
        ivDebugArrow.rotation = if (isExpanded) 180f else 0f
    }

    fun updatePinDebugPanel(result: UiState.PinDetected) {
        val okCount = result.segmentResults.count { it.status == RawDigitClassifier.SegmentStatus.OK }
        val unsureCount = result.segmentResults.count { it.status == RawDigitClassifier.SegmentStatus.UNSURE }

        textDebugSummary.text = "Segmente: ${result.totalSegments} | OK: $okCount | UNSURE: $unsureCount"
        textDebugAudio.text = "Pipeline: VAD | Thresholds: conf>=80%, margin>=15%"
        textDebugResult.text = "PIN: \"${result.pin}\" (${result.segmentResults.size} Segmente)"

        llProbabilities.removeAllViews()

        for (segmentResult in result.segmentResults) {
            val segmentView = TextView(context).apply {
                val statusIcon = when (segmentResult.status) {
                    RawDigitClassifier.SegmentStatus.OK -> "[OK]"
                    RawDigitClassifier.SegmentStatus.UNSURE -> "[UNSURE]"
                }
                text = "$statusIcon [${segmentResult.segmentIndex}] ${segmentResult.label}: " +
                       "${(segmentResult.confidence * 100).toInt()}% (margin=${(segmentResult.margin * 100).toInt()}%, ${segmentResult.segment.durationMs}ms)"
                textSize = 11f
                setTextColor(
                    when (segmentResult.status) {
                        RawDigitClassifier.SegmentStatus.OK ->
                            ContextCompat.getColor(context, R.color.colorAccent)
                        RawDigitClassifier.SegmentStatus.UNSURE ->
                            ContextCompat.getColor(context, R.color.colorWarning)
                    }
                )
                setPadding(8, 4, 8, 4)
            }
            llProbabilities.addView(segmentView)
        }
    }
}
