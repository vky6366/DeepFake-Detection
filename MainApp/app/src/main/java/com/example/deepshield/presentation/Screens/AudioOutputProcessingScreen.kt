package com.example.deepshield.presentation.Screens

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.navigation.compose.hiltViewModel
import com.airbnb.lottie.compose.LottieAnimation
import com.airbnb.lottie.compose.LottieCompositionSpec
import com.airbnb.lottie.compose.LottieConstants
import com.airbnb.lottie.compose.animateLottieCompositionAsState
import com.airbnb.lottie.compose.rememberLottieComposition
import com.example.deepshield.R
import com.example.deepshield.presentation.Navigation.ALLSONGSCREEN
import com.example.deepshield.presentation.Utils.LoadingIndicator
import com.example.deepshield.presentation.viewModel.MyViewModel

@Composable
fun AudioOutputProcessingScreen(songTitle: String,song: String,viewModel: MyViewModel= hiltViewModel()) {
    val lottiecomposition by rememberLottieComposition(LottieCompositionSpec.RawRes(R.raw.button))
    val progress2 by animateLottieCompositionAsState(
        composition = lottiecomposition,
        iterations = LottieConstants.IterateForever,
        speed = 0.75f
    )
    val context = LocalContext.current
    val audioUploadState = viewModel.uploadAudioToServerState.collectAsState()
    when{
        audioUploadState.value.isLoading -> {
            LoadingIndicator()
        }
        audioUploadState.value.data != null -> {
            Text(text = audioUploadState.value.data.toString())
        }

    }

    Column(modifier = Modifier.fillMaxSize()) {
        Box(
            contentAlignment = Alignment.Center,  // Centers the text inside the animation
            modifier = Modifier
                .fillMaxWidth(0.95f)
                .height(50.dp)
                .clickable {
                    viewModel.uploadAudioToDeepFakeServer(context = context, audioUrl = song)

                }
        ) {
            LottieAnimation(
                composition = lottiecomposition,
                progress = { progress2 },
                modifier = Modifier.fillMaxWidth()  // Makes animation fill the Box
            )

            Text(
                text = "Upload",  // Your desired text
                color = Color.White,  // Adjust color for visibility
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold
            )
        }
    }

}