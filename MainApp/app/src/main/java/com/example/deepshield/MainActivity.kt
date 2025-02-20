package com.example.deepshield

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text

import androidx.compose.ui.Modifier
import com.example.deepshield.presentation.Navigation.MyApp
import com.example.deepshield.presentation.Screens.VideoScreenSelector
import com.example.deepshield.presentation.Utils.LoadingIndicator

import com.example.deepshield.ui.theme.DeepShieldTheme
import dagger.hilt.android.AndroidEntryPoint

@AndroidEntryPoint
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        //installSplashScreen()
        setContent {
            DeepShieldTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    Box(modifier = Modifier.padding(innerPadding)) {
                            //MyApp()
                        VideoScreenSelector()


                    }

                }
            }
        }
    }

}


