package com.example.deepshield.presentation.Screens

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.util.Log
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.net.toUri
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.navigation.NavController
import coil3.compose.rememberAsyncImagePainter
import com.example.deepshield.presentation.Utils.LoadingIndicator
import com.example.deepshield.presentation.viewModel.MyViewModel

@Composable
fun VideoProcessingScreen(
    viewmodel: MyViewModel = hiltViewModel(),
    imageUri: String,
    videoUri: String,
    navController: NavController
) {
    val deepfakeResponseState = viewmodel.uploadDeepFakeVideoState.collectAsState()
    val context = LocalContext.current
    val data = remember { mutableStateOf<Bitmap?>(null) }
    val infiniteTransition = rememberInfiniteTransition()
    val animatedAlpha by infiniteTransition.animateFloat(
        initialValue = 0.2f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 1000, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse
        )
    )

    LaunchedEffect(Unit) {
        Log.d("UPLOAD", "Uploading video: $videoUri")
        data.value = getVideoThumbnail(context, videoUri.toUri())
        viewmodel.uploadVideoToDeepFakeServer(context = context, videoUri = videoUri)
    }

    // Observe API Response

    Log.d("APIRESPONSE2", "${deepfakeResponseState.value.data?.prediction}")

    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {

        data.value?.let { bitmap ->
            Image(
                bitmap = bitmap.asImageBitmap(),
                contentDescription = "Video Thumbnail",
                modifier = Modifier.size(200.dp) .graphicsLayer(
                    alpha = animatedAlpha, // Apply animated transparency
                    scaleX = 0.8f, // Reduce size to 80%
                    scaleY = 0.8f
                )

            )
        }

        if(deepfakeResponseState.value.data != null) {
            Text("Message: ${deepfakeResponseState.value.data?.prediction}")
        }
    }
}
