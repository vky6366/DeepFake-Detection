package com.example.deepshield.Test

import androidx.compose.ui.test.ExperimentalTestApi
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.navigation.NavController
import androidx.navigation.compose.rememberNavController
import com.example.deepshield.MainActivity
import com.example.deepshield.data.Constants.TestTags
import com.example.deepshield.di.DiObject
import com.example.deepshield.presentation.Navigation.MyApp
import com.example.deepshield.presentation.Screens.VideoProcessingScreen
import dagger.hilt.android.testing.HiltAndroidRule
import dagger.hilt.android.testing.HiltAndroidTest
import dagger.hilt.android.testing.UninstallModules
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import org.junit.Before
import org.junit.Rule
import org.junit.Test


@HiltAndroidTest
@UninstallModules(DiObject::class)
class DeepShieldTest {

    @get:Rule(order = 0)
    val hiltRule = HiltAndroidRule(this)

    @OptIn(ExperimentalTestApi::class)
    @get:Rule(order = 1)
    val composeTestRule= createAndroidComposeRule<MainActivity>()

    @Before
    fun setUp(){
        hiltRule.inject()

    }

    @Test
    fun deepFakeVideoScreenTest(){
        composeTestRule.onNodeWithTag(TestTags.VIDEOCARD).assertExists()
        composeTestRule.onNodeWithTag(TestTags.VIDEOCARD).performClick()
        composeTestRule.onNodeWithTag(TestTags.VIDEOSELECTIONSCREEN).assertExists()
        composeTestRule.onNodeWithTag(TestTags.VIDEOSELECTIONSCREEN).performClick()
    }



}