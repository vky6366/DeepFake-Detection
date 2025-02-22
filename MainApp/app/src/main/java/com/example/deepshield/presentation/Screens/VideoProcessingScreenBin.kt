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
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.net.toUri
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.navigation.NavController
import coil3.compose.rememberAsyncImagePainter
import com.example.deepshield.R
import com.example.deepshield.presentation.Utils.AnimatedText
import com.example.deepshield.presentation.Utils.LoadingIndicator
import com.example.deepshield.presentation.viewModel.MyViewModel
import com.shashank.sony.fancytoastlib.FancyToast



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

    // Animation transition (Only runs when loading)
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
        horizontalAlignment = Alignment.CenterHorizontally

    ) {
        Spacer(modifier = Modifier.height(32.dp))
        data.value?.let { bitmap ->
            Card(
                shape = CircleShape,  // Makes it circular
                elevation = CardDefaults.elevatedCardElevation(8.dp),  // Adds shadow effect
                modifier = Modifier
                    .size(320.dp)  // Set size of the circular box
                    .padding(16.dp)
            ) {
                Image(
                    bitmap = bitmap.asImageBitmap(),
                    contentDescription = "Video Thumbnail",
                    modifier = Modifier
                        .size(300.dp)  // Image size inside the circle
                        .clip(CircleShape)  // Clips image to circular shape
                        .graphicsLayer(
                            alpha = if (deepfakeResponseState.value.isLoading) animatedAlpha else 1f,
                            scaleX = 0.9f,
                            scaleY = 0.9f
                        )
                )

            }
        }//end
        Spacer(modifier = Modifier.height(16.dp))
        if(deepfakeResponseState.value.isLoading) {
            FancyToast.makeText(context,"processing",FancyToast.LENGTH_SHORT,FancyToast.CONFUSING,false).show()
           // Text("Processing...", color = colorResource(id= R .color.themecolour))
            AnimatedText()
        }
        else if (deepfakeResponseState.value.data != null) {
            val score = deepfakeResponseState.value.data?.score ?: 0.0
            val formattedScore = String.format("%.3f", score)
            if( deepfakeResponseState.value.data?.prediction == "FAKE") {
                Text("Prediction: ${deepfakeResponseState.value.data?.prediction}", color = Color.Red,
                    fontSize = 35.sp,  // Keep font size constant
                    fontWeight = FontWeight.Bold,)
            }else{
                Text("Prediction: ${deepfakeResponseState.value.data?.prediction}", color = Color.Green,
                    fontSize = 35.sp,  // Keep font size constant
                    fontWeight = FontWeight.Bold,)
            }
            if (deepfakeResponseState.value.data?.score!! < 0.5){
                Text("Score: ${formattedScore}", color = Color.Green,
                    fontSize = 35.sp,  // Keep font size constant
                    fontWeight = FontWeight.Bold)

            }else if (deepfakeResponseState.value.data?.score!! == 0.5){
                Text("Score: ${formattedScore}", color = Color.Red,
                    fontSize = 35.sp,  // Keep font size constant
                    fontWeight = FontWeight.Bold)
            }else{
                Text("Score: ${formattedScore}", color = Color.Red,
                    fontSize = 35.sp,  // Keep font size constant
                    fontWeight = FontWeight.Bold)
            }

        }
        else if(deepfakeResponseState.value.error.isNullOrEmpty()){
//            FancyToast.makeText(context,"Error in processing",FancyToast.LENGTH_SHORT,FancyToast.ERROR,false).show()
        }
        //0.5 ke niche real , 0.5 = fake and up fake
    }

}

