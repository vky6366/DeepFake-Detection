package com.example.deepshield.presentation.Screens

import android.util.Log
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.example.deepshield.R
import com.example.deepshield.presentation.viewModel.MyViewModel

@Composable
fun DeepFakeVideoScreen(viewmodel: MyViewModel = hiltViewModel()) {
    val uploadVideoState = viewmodel.uploadDeepFakeVideoState.collectAsState()

    if (uploadVideoState.value.isLoading) {


    } else if (uploadVideoState.value.error != null) {

    } else {

        // Create an infinite transition for animation
        val infiniteTransition = rememberInfiniteTransition()
        val animatedAlpha by infiniteTransition.animateFloat(
            initialValue = 0.2f,
            targetValue = 1f,
            animationSpec = infiniteRepeatable(
                animation = tween(durationMillis = 1000, easing = LinearEasing),
                repeatMode = RepeatMode.Reverse
            )
        )

        Log.d("APIRESPONSE", "${uploadVideoState.value.data}")

        val videourl = remember { mutableStateOf("") }
        val mediapicker = rememberLauncherForActivityResult(
            ActivityResultContracts.PickVisualMedia(),
            onResult = { uri -> videourl.value = uri.toString() }
        )
        val context = LocalContext.current

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            // Circular Animated Image
            Image(
                painter = painterResource(id = R.drawable.faceloading),
                contentDescription = "Loading",
                modifier = Modifier
                    .size(400.dp)  // Increased size
                    .clip(CircleShape)  // Circular image
                    .graphicsLayer(
                        alpha = animatedAlpha, // Apply animated transparency
                        scaleX = 0.8f, // Reduce size to 80%
                        scaleY = 0.8f
                    )
            )

            Spacer(modifier = Modifier.height(24.dp)) // Spacing

            // Video Input Button
            Button(onClick = {
                mediapicker.launch(
                    PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly)
                )
            }, modifier = Modifier.fillMaxWidth(0.85f)) {
                Text("Select Video")
            }

            Spacer(modifier = Modifier.height(12.dp))

            // Upload Button
            Button(onClick = {
                if (videourl.value.isNotEmpty()) {
                    viewmodel.uploadVideoToDeepFakeServer(
                        context = context,
                        videoUri = videourl.value
                    )
                } else {
                    Toast.makeText(context, "Please select a video", Toast.LENGTH_SHORT).show()
                }
            }, modifier = Modifier.fillMaxWidth(0.85f)) {
                Text("Upload Video")
            }
        }
    }
}
//
//@Composable
//fun AnimatedLoadingImage() {
//    val animatedAlpha by rememberInfiniteTransition().animateFloat(
//        initialValue = 0.2f,
//        targetValue = 1f,
//        animationSpec = infiniteRepeatable(
//            animation = tween(1000, easing = LinearEasing),
//            repeatMode = RepeatMode.Reverse
//        )
//    )
//
//    Image(
//        painter = painterResource(id = R.drawable.faceloading),
//        contentDescription = "Loading",
//        modifier = Modifier
//            .size(250.dp)
//            .graphicsLayer(alpha = animatedAlpha)
//    )
//}

