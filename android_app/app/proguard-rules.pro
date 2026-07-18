# ===== TensorFlow Lite / LiteRT =====
-keep class com.google.ai.edge.litert.** { *; }
-keep class org.tensorflow.lite.** { *; }
-keepclassmembers class org.tensorflow.lite.** { *; }

# ===== WebRTC VAD =====
# Keep native methods for VAD JNI
-keep class com.konovalov.vad.** { *; }
-keepclassmembers class com.konovalov.vad.** {
    native <methods>;
}

# ===== Kotlin Coroutines =====
-keepnames class kotlinx.coroutines.internal.MainDispatcherFactory {}
-keepnames class kotlinx.coroutines.CoroutineExceptionHandler {}
-keepclassmembers class kotlinx.coroutines.** {
    volatile <fields>;
}

# ===== General Android =====
-keepattributes Signature
-keepattributes Exceptions

# ===== Release optimizations =====
# Strip all Log calls in release builds
-assumenosideeffects class android.util.Log {
    public static boolean isLoggable(java.lang.String, int);
    public static int v(...);
    public static int d(...);
    public static int i(...);
    public static int w(...);
    public static int e(...);
}

# Strip AppLog calls (all inline, removed together with underlying Log calls)
-assumenosideeffects class com.orez74.speechdigits.AppLog {
    *** d(...);
    *** v(...);
    *** i(...);
    *** w(...);
    *** e(...);
    *** separator(...);
    *** banner(...);
}

# Keep ViewBinding classes
-keep class com.orez74.speechdigits.databinding.** { *; }
