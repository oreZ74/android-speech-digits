# Contributing to DigitVoice

Thanks for your interest in contributing! This document outlines the workflow
for submitting improvements.

## Development Workflow

1. **Fork** the repository and clone your fork.
2. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/my-feature
   ```
3. **Make your changes** following the code style below.
4. **Write or update tests** to cover your changes.
5. **Run the test suite** to ensure nothing is broken:
   ```bash
   cd android_app
   ./gradlew test
   ```
6. **Commit** changes using a descriptive, imperative message in the present tense, following the *Conventional Commits* specification:
   ```
   Add onset backtracking to WebRTC VAD segmenter
   ```
7. **Push** your branch and open a **Pull Request** against `main`.

## Code Style

- **Language:** All code comments and documentation must be in **English**.
- **Kotlin:** Follow the [official Kotlin coding conventions](https://kotlinlang.org/docs/coding-conventions.html).
- **Android:** Use `ViewModel` + `StateFlow` for new UI logic. Keep Activities lean.
- **Python:** Follow [PEP 8](https://pep8.org/). Use type hints where practical.
- **Formatting:** Consistent 4-space indentation. No trailing whitespace.
- **Logging:** Use the centralized `AppLog` helper instead of raw `android.util.Log` calls.

## Pull Request Checklist

- [ ] Code builds without warnings (`./gradlew assembleDebug`)
- [ ] Unit tests pass (`./gradlew test`)
- [ ] New code has corresponding unit tests
- [ ] No hardcoded UI strings (use `strings.xml`)
- [ ] Accessibility features are preserved (TalkBack, content descriptions)
- [ ] PR description explains _what_ and _why_

## Project Architecture

For an overview of the codebase organization, see the
[architecture section of the README](README.md#architecture).

Key principles:
- **Single Responsibility:** Each class has one clear purpose.
- **MVVM:** `MainViewModel` owns all UI state via `StateFlow<UiState>`.
- **Facade Pattern:** `RawDigitClassifier` orchestrates the extracted components
  (`TfliteModelLoader`, `SegmentClassifier`, `PinAssemblyEngine`, etc.).
- **Offline-first:** No network calls. Everything runs on-device.
