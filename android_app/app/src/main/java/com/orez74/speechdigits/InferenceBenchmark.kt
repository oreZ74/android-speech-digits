package com.orez74.speechdigits

/**
 * Benchmark utilities for measuring inference latency.
 *
 * Uses a TfliteModelLoader to run warmup + measurement phases
 * with synthetic silence input, then computes statistics.
 */
object InferenceBenchmark {

    const val TAG = "InferenceBenchmark"

    fun runBenchmark(
        modelLoader: TfliteModelLoader,
        warmupRuns: Int = 20,
        measureRuns: Int = 100
    ): RawDigitClassifier.BenchmarkResult {
        val testSamples = FloatArray(TfliteModelLoader.EXPECTED_SAMPLES) { 0.0f }
        val inputBuffer = modelLoader.createInputBuffer(testSamples)
        val numClasses = modelLoader.outputShape.last()
        val outputBuffer = Array(1) { FloatArray(numClasses) }

        AppLog.i(TAG) { "Benchmark: Starting warmup ($warmupRuns runs)..." }
        repeat(warmupRuns) {
            inputBuffer.rewind()
            modelLoader.runInference(inputBuffer)
        }

        AppLog.i(TAG) { "Benchmark: Measuring latency ($measureRuns runs)..." }
        val latencies = mutableListOf<Long>()

        repeat(measureRuns) {
            inputBuffer.rewind()
            val startNs = System.nanoTime()
            modelLoader.runInference(inputBuffer)
            val endNs = System.nanoTime()
            latencies.add(endNs - startNs)
        }

        val sortedLatencies = latencies.sorted()
        val medianNs = if (sortedLatencies.size % 2 == 0) {
            (sortedLatencies[sortedLatencies.size / 2 - 1] + sortedLatencies[sortedLatencies.size / 2]) / 2.0
        } else {
            sortedLatencies[sortedLatencies.size / 2].toDouble()
        }
        val p95Index = (sortedLatencies.size * 0.95).toInt().coerceAtMost(sortedLatencies.size - 1)
        val p95Ns = sortedLatencies[p95Index]
        val minNs = sortedLatencies.first()
        val maxNs = sortedLatencies.last()

        val medianMs = medianNs / 1_000_000.0
        val p95Ms = p95Ns / 1_000_000.0
        val minMs = minNs / 1_000_000.0
        val maxMs = maxNs / 1_000_000.0

        val androidVersion = android.os.Build.VERSION.RELEASE
        val deviceModel = "${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL}"

        val result = RawDigitClassifier.BenchmarkResult(
            warmupRuns = warmupRuns,
            measureRuns = measureRuns,
            medianMs = medianMs,
            p95Ms = p95Ms,
            minMs = minMs,
            maxMs = maxMs,
            modelPath = TfliteModelLoader.MODEL_PATH,
            androidVersion = androidVersion,
            deviceModel = deviceModel
        )

        AppLog.i(TAG) { "Benchmark Result: Median=${"%.2f".format(medianMs)}ms, P95=${"%.2f".format(p95Ms)}ms, Min=${"%.2f".format(minMs)}ms, Max=${"%.2f".format(maxMs)}ms" }
        AppLog.i(TAG) { "Device: $deviceModel (Android $androidVersion)" }

        return result
    }
}
