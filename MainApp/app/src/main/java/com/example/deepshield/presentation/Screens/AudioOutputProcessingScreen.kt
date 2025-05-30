package com.example.deepshield.presentation.Screens

import android.util.Log
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.media3.common.MediaItem
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.ui.PlayerView
import com.airbnb.lottie.compose.LottieAnimation
import com.airbnb.lottie.compose.LottieCompositionSpec
import com.airbnb.lottie.compose.LottieConstants
import com.airbnb.lottie.compose.animateLottieCompositionAsState
import com.airbnb.lottie.compose.rememberLottieComposition
import com.example.deepshield.R
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
    val primaryBlue = Color(0xFF4A84D4)
    val context = LocalContext.current
    val audioUploadState = viewModel.uploadAudioToServerState.collectAsState()
    val exoPlayer = remember {
        ExoPlayer.Builder(context).build().apply {
            val mediaItem = MediaItem.fromUri(song)
            setMediaItem(mediaItem)
            prepare()
            playWhenReady = false// Auto-play
        }
    }

    Column(modifier = Modifier.fillMaxSize(), horizontalAlignment = Alignment.CenterHorizontally) {
        Spacer(modifier = Modifier.height(8.dp))
        Text(songTitle, fontSize = 28.sp, fontWeight = FontWeight.Bold, color = primaryBlue)
        Spacer(modifier = Modifier.height(16.dp))
        AndroidView(
            factory = { ctx ->
                PlayerView(ctx).apply {
                    player = exoPlayer
                    useController = true // Show play/pause controls
                }
            },
            modifier = Modifier.fillMaxWidth(0.8f).height(450.dp)
        )
        Spacer(modifier = Modifier.height(16.dp))
        when{
            audioUploadState.value.isLoading -> {
                LoadingIndicator()
            }
            audioUploadState.value.data != null -> {
                Text(text = audioUploadState.value.data!!.prediction.toString(), fontSize = 28.sp, fontWeight = FontWeight.Bold, color = primaryBlue)
                Log.d("AUDIORESPONSE", "${audioUploadState.value.data}")
            }
            audioUploadState.value.error != null -> {
                Text(text = audioUploadState.value.error.toString())
            }

        }
        Spacer(modifier = Modifier.height(8.dp))

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
                text = "Upload",
                color = Color.White,
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold
            )
        }
    }
    DisposableEffect(Unit) {
        onDispose {
            exoPlayer.release() // Release player when composable is removed
        }
    }

}
