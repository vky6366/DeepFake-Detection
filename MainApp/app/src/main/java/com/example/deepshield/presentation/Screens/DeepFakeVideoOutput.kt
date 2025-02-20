package com.example.deepshield.presentation.Screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun DeepFakeVideoOutput(   message: String, prediction: String, score: Double) {
    Column(modifier = Modifier.fillMaxSize(), verticalArrangement = Arrangement.Center,
        horizontalAlignment =Alignment.CenterHorizontally) {
        Text("$message")
        Spacer(modifier = Modifier.height (16.dp))
        Text("$prediction")
        Spacer(modifier = Modifier.height (16.dp))
        Text("$score")

    }


}