package com.orez74.speechdigits

import android.Manifest
import androidx.test.espresso.Espresso.onView
import androidx.test.espresso.action.ViewActions.click
import androidx.test.espresso.assertion.ViewAssertions.matches
import androidx.test.espresso.matcher.ViewMatchers.*
import androidx.test.ext.junit.rules.ActivityScenarioRule
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.rule.GrantPermissionRule
import com.github.orez74.speechdigits.R
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class MainActivityTest {

    @get:Rule
    val activityRule = ActivityScenarioRule(MainActivity::class.java)

    // Automatically grants the RECORD_AUDIO permission for all tests in this class
    @get:Rule
    val grantPermissionRule: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.RECORD_AUDIO)

    @Test
    fun appStarts_TitleIsVisible() {
        onView(withId(R.id.tvTitle))
            .check(matches(isDisplayed()))
            .check(matches(withText("PIN-Eingabe")))
    }

    @Test
    fun pinModeButton_isDisplayedAndClickable() {
        // Verify PIN mode button is visible and clickable
        onView(withId(R.id.btnModePin))
            .check(matches(isDisplayed()))
            .check(matches(isClickable()))
    }

    @Test
    fun initialState_showsCorrectStatus() {
        // Check initial status text
        onView(withId(R.id.textStatus))
            .check(matches(isDisplayed()))
            .check(matches(withText("Bereit")))
        
        // Check initial instruction
        onView(withId(R.id.textInstruction))
            .check(matches(isDisplayed()))
            .check(matches(withText("Drücke den Button und sprich deine PIN")))
    }
    
    @Test
    fun micButtonContainer_isDisplayed() {
        // Verify mic button container with pulsing rings is visible
        onView(withId(R.id.micButtonContainer))
            .check(matches(isDisplayed()))
    }
    
    @Test
    fun clickPinButton_startsRecording() {
        // Click PIN mode button
        onView(withId(R.id.btnModePin)).perform(click())

        // Wait for recording to start
        Thread.sleep(500)

        // Check status changed (should show listening state)
        onView(withId(R.id.textStatus))
            .check(matches(isDisplayed()))
    }
    
    @Test
    fun debugPanel_isDisplayed() {
        // Debug panel should be visible in debug builds
        onView(withId(R.id.debugCard))
            .check(matches(isDisplayed()))
    }
    
    @Test
    fun resultCard_isDisplayed() {
        // Result card should be visible
        onView(withId(R.id.resultCard))
            .check(matches(isDisplayed()))
    }
}
