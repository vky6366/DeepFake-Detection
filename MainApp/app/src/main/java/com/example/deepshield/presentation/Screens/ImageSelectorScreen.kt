package com.example.deepshield.presentation.Screens

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.airbnb.lottie.compose.LottieAnimation
import com.airbnb.lottie.compose.LottieCompositionSpec
import com.airbnb.lottie.compose.LottieConstants
import com.airbnb.lottie.compose.animateLottieCompositionAsState
import com.airbnb.lottie.compose.rememberLottieComposition
import com.example.deepshield.R

@Composable
fun ImageSelectorScreen() {
    val videouri= remember { mutableStateOf("") }
    val lottiecomposition by rememberLottieComposition(LottieCompositionSpec.RawRes(R.raw.button))
    val progress2 by animateLottieCompositionAsState(
        composition = lottiecomposition,
        iterations = LottieConstants.IterateForever,
        speed = 0.75f
    )
    val medialauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia(), onResult = {uri->
            videouri.value = uri.toString()
        }
    )
    Column(modifier = Modifier.fillMaxSize()) {
        Text("Please Select an Image")
        Box(
            contentAlignment = Alignment.Center,  // Centers the text inside the animation
            modifier = Modifier
                .fillMaxWidth(0.95f)
                .height(50.dp)
                .clickable {
                    medialauncher.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly))
                }
        ) {
            // Lottie Animation
            LottieAnimation(
                composition = lottiecomposition,
                progress = { progress2 },
                modifier = Modifier.fillMaxWidth()  // Makes animation fill the Box
            )

            // Overlayed Text
            Text(
                text = "Select Video",  // Your desired text
                color = Color.White,  // Adjust color for visibility
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold
            )
        }


    }

}