package com.example.deepshield.presentation.Screens

import android.graphics.Color
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
import androidx.compose.foundation.clickable
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
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.navigation.NavController
import com.airbnb.lottie.compose.LottieAnimation
import com.airbnb.lottie.compose.LottieCompositionSpec
import com.airbnb.lottie.compose.LottieConstants
import com.airbnb.lottie.compose.animateLottieCompositionAsState
import com.airbnb.lottie.compose.rememberLottieAnimatable
import com.airbnb.lottie.compose.rememberLottieComposition
import com.example.deepshield.R
import com.example.deepshield.presentation.Navigation.DEEPFAKEVIDEOOUTPUTSCREEN
import com.example.deepshield.presentation.Utils.LoadingIndicator
import com.example.deepshield.presentation.viewModel.MyViewModel

@Composable
fun DeepFakeVideoScreen(viewmodel: MyViewModel = hiltViewModel(),navController: NavController) {
    val context= LocalContext.current
    val uploadVideoState = viewmodel.uploadDeepFakeVideoState.collectAsState()
    LaunchedEffect(uploadVideoState.value.data) {
        uploadVideoState.value.data?.let { data ->
            Toast.makeText(context, data.message, Toast.LENGTH_SHORT).show()
            navController.navigate(
                DEEPFAKEVIDEOOUTPUTSCREEN(
                    message = data.message,
                    prediction = data.prediction,
                    score = data.score
                )
            )
        }
    }

    if (uploadVideoState.value.isLoading) {
        LoadingIndicator()
    }
    else {
        val composition by rememberLottieComposition(
            spec = LottieCompositionSpec.RawRes(R.raw.clickhere)
        )
        val progress by animateLottieCompositionAsState(
            composition,
            iterations = LottieConstants.IterateForever,
            speed = 0.8f
        )

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
       // val context = LocalContext.current

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            // Circular Animated Image
            if (videourl.value == "" || videourl.value.isNullOrEmpty() || videourl.value.isBlank()) {
                LottieAnimation(
                    composition,
                    progress = { progress },
                    modifier = Modifier
                        .size(250.dp)
                        .clickable {
                            //todo
                            mediapicker.launch(
                                PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly)
                            )
                            Log.d("VALUE", "${videourl.value}")

                        }
                )
            } else {
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

            }


            Spacer(modifier = Modifier.height(18.dp))

            // Upload Button
            Button(
                onClick = {
                    if (videourl.value.isNotEmpty()) {
                        viewmodel.uploadVideoToDeepFakeServer(
                            context = context,
                            videoUri = videourl.value
                        )
                    } else {
                        Toast.makeText(context, "Please select a video", Toast.LENGTH_SHORT).show()
                    }
                }, modifier = Modifier
                    .fillMaxWidth(0.85f)
                    .height(50.dp), colors = ButtonDefaults.buttonColors(
                    containerColor = colorResource(id = R.color.themecolour) // Custom Hex Color
                )
            ) {
                Text("Upload Video", color = colorResource(id = R.color.black))
            }
        }
    }

}
//@Composable
//fun DeepFakeVideoScreen(viewmodel: MyViewModel = hiltViewModel(),navController: NavController) {
//    LocalContext.current
//    val uploadVideoState = viewmodel.uploadDeepFakeVideoState.collectAsState()
//    val stateOfNavigation= remember { mutableStateOf(false) }
//    if (uploadVideoState.value.isLoading) {
//        LoadingIndicator()
//    }else if(uploadVideoState.value.data!=null){
//
//        navController.navigate(DEEPFAKEVIDEOOUTPUTSCREEN(message = uploadVideoState.value.data!!.message,
//            prediction = uploadVideoState.value.data!!.prediction, score = uploadVideoState.value.data!!.score
//        ))
//
//
//
//    }
//    else {
//        val composition by rememberLottieComposition(
//            spec = LottieCompositionSpec.RawRes(R.raw.clickhere)
//        )
//        val progress by animateLottieCompositionAsState(
//            composition,
//            iterations = LottieConstants.IterateForever,
//            speed = 0.8f
//        )
//
//        // Create an infinite transition for animation
//        val infiniteTransition = rememberInfiniteTransition()
//        val animatedAlpha by infiniteTransition.animateFloat(
//            initialValue = 0.2f,
//            targetValue = 1f,
//            animationSpec = infiniteRepeatable(
//                animation = tween(durationMillis = 1000, easing = LinearEasing),
//                repeatMode = RepeatMode.Reverse
//            )
//        )
//
//        Log.d("APIRESPONSE", "${uploadVideoState.value.data}")
//
//        val videourl = remember { mutableStateOf("") }
//        val mediapicker = rememberLauncherForActivityResult(
//            ActivityResultContracts.PickVisualMedia(),
//            onResult = { uri -> videourl.value = uri.toString() }
//        )
//        val context = LocalContext.current
//
//        Column(
//            modifier = Modifier
//                .fillMaxSize()
//                .padding(16.dp),
//            horizontalAlignment = Alignment.CenterHorizontally,
//            verticalArrangement = Arrangement.Center
//        ) {
//            // Circular Animated Image
//            if (videourl.value == "" || videourl.value.isNullOrEmpty() || videourl.value.isBlank()) {
//                LottieAnimation(
//                    composition,
//                    progress = { progress },
//                    modifier = Modifier
//                        .size(250.dp)
//                        .clickable {
//                            //todo
//                            mediapicker.launch(
//                                PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly)
//                            )
//                            Log.d("VALUE", "${videourl.value}")
//
//                        }
//                )
//            } else {
//                Image(
//                    painter = painterResource(id = R.drawable.faceloading),
//                    contentDescription = "Loading",
//                    modifier = Modifier
//                        .size(400.dp)  // Increased size
//                        .clip(CircleShape)  // Circular image
//                        .graphicsLayer(
//                            alpha = animatedAlpha, // Apply animated transparency
//                            scaleX = 0.8f, // Reduce size to 80%
//                            scaleY = 0.8f
//                        )
//                )
//
//            }
//
//
//            Spacer(modifier = Modifier.height(18.dp))
//
//            // Upload Button
//            Button(
//                onClick = {
//                    if (videourl.value.isNotEmpty()) {
//                        viewmodel.uploadVideoToDeepFakeServer(
//                            context = context,
//                            videoUri = videourl.value
//                        )
//                        navController.navigate(DEEPFAKEVIDEOOUTPUTSCREEN(
////                            message = uploadVideoState.value.data.message,
////                            prediction = uploadVideoState.value.data!!.prediction,
////                            score = uploadVideoState.value.data!!.score
//                            message = uploadVideoState.value.data?.message ?: "No message",
//                            prediction = uploadVideoState.value.data?.prediction ?: "Unknown",
//                            score = uploadVideoState.value.data?.score ?: 0.0
//                        ))
//                    } else {
//                        Toast.makeText(context, "Please select a video", Toast.LENGTH_SHORT).show()
//                    }
//                }, modifier = Modifier
//                    .fillMaxWidth(0.85f)
//                    .height(50.dp), colors = ButtonDefaults.buttonColors(
//                    containerColor = colorResource(id = R.color.themecolour) // Custom Hex Color
//                )
//            ) {
//                Text("Upload Video", color = colorResource(id = R.color.black))
//            }
//        }
//    }
//
//}