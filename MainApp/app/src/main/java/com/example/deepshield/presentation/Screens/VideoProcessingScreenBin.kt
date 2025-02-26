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
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.wrapContentSize
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.RectangleShape
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.net.toUri
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.navigation.NavController
import coil3.compose.rememberAsyncImagePainter
import com.airbnb.lottie.compose.LottieAnimation
import com.airbnb.lottie.compose.LottieCompositionSpec
import com.airbnb.lottie.compose.LottieConstants
import com.airbnb.lottie.compose.animateLottieCompositionAsState
import com.airbnb.lottie.compose.rememberLottieComposition
import com.example.deepshield.R
import com.example.deepshield.presentation.Navigation.HEATMAPSCREEN
import com.example.deepshield.presentation.Navigation.VIDEOSELECTIONSCREEN
import com.example.deepshield.presentation.Utils.AnimatedText
import com.example.deepshield.presentation.viewModel.MyViewModel
import com.shashank.sony.fancytoastlib.FancyToast



@Composable
fun VideoProcessingScreen(
    viewmodel: MyViewModel = hiltViewModel(),
    imageUri: String,
    videoUri: String,
    navController: NavController
) {
    val navigationFlag = remember { mutableStateOf(false) }
    val deepfakeResponseState = viewmodel.uploadDeepFakeVideoState.collectAsState()
    val context = LocalContext.current
    val data = remember { mutableStateOf<Bitmap?>(null) }

    // Animation transition (Only runs when loading)
    val infiniteTransition = rememberInfiniteTransition()
    val animatedAlpha by infiniteTransition.animateFloat(
        initialValue = 0.2f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 1300, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse
        )
    )
    val lottiecomposition by rememberLottieComposition(LottieCompositionSpec.RawRes(R.raw.button))
    val progress2 by animateLottieCompositionAsState(
        composition = lottiecomposition,
        iterations = LottieConstants.IterateForever,
        speed = 0.75f
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
                shape = RectangleShape,
                elevation = CardDefaults.elevatedCardElevation(8.dp),  // Adds shadow effect
                modifier = Modifier
                    .size(350.dp)
                    .padding(16.dp).background(Color.LightGray)
            ) {
                Image(
                    bitmap = bitmap.asImageBitmap(),
                    contentDescription = "Video Thumbnail",
                    modifier = Modifier
                        .size(300.dp)  // Image size inside the circle
                        .clip(RectangleShape)  // Clips image to circular shape
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
            navigationFlag.value = false
            FancyToast.makeText(context,"processing",FancyToast.LENGTH_SHORT,FancyToast.CONFUSING,false).show()

            Text(
                text = "Processing...",
                color = colorResource(id = R.color.themecolour),
                fontSize = 35.sp,  // Keep font size constant
                fontWeight = FontWeight.Bold,
            )
            Spacer(modifier = Modifier.height(16.dp))
        }
        else if (deepfakeResponseState.value.data != null) {


            navigationFlag.value=true
            val score = deepfakeResponseState.value.data?.score ?: 0.0
            val formattedScore = String.format("%.3f", score)
            if( deepfakeResponseState.value.data?.prediction == "FAKE") {
                FancyToast.makeText(context,"Video may be AI modified",FancyToast.LENGTH_SHORT,FancyToast.ERROR,false).show()
                Box(
                    modifier = Modifier
                        .wrapContentSize()  // Automatically adjusts width & height
                        .background(colorResource(R.color.themesuit1), shape = RoundedCornerShape(8.dp)) // Add rounded corners
                        .padding(8.dp)  // Add padding for better spacing
                ) {
                    Text(
                        text = "Looks like Computer generated OR Modified data ",
                        fontSize = 16.sp,
                        fontWeight = FontWeight.SemiBold

                    )
                }
                Box(
                    contentAlignment = Alignment.Center,  // Centers the text inside the animation
                    modifier = Modifier
                        .fillMaxWidth(0.95f)
                        .height(50.dp)
                        .clickable {
                          //HeatMap
                            navController.navigate(HEATMAPSCREEN)

                        }
                ) {
                    // Lottie Animation
                    LottieAnimation(
                        composition = lottiecomposition,
                        progress = { progress2 },
                        modifier = Modifier.fillMaxWidth()
                    )

                    // Overlayed Text
                    Text(
                        text = "HeatMap",  // Your desired text
                        color = Color.White,  // Adjust color for visibility
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold
                    )
                }


            }else{
                FancyToast.makeText(context,"Video Seems to be REAL",FancyToast.LENGTH_SHORT,FancyToast.SUCCESS,false).show()
                Box(
                    modifier = Modifier
                        .wrapContentSize()  // Automatically adjusts width & height
                        .background(colorResource(R.color.themecolour), shape = RoundedCornerShape(8.dp)) // Add rounded corners
                        .padding(8.dp)  // Add padding for better spacing
                ) {
                    Text(
                        text = "This Video is not modified OR Altered",
                        fontSize = 16.sp,
                        fontWeight = FontWeight.SemiBold,
                          // Ensure text is visible on background
                    )
                }
            }
            Spacer(modifier = Modifier.height(16.dp))
//            if (deepfakeResponseState.value.data?.score!! < 0.5){
//                Text("Score: ${formattedScore}", color = Color.Green,
//                    fontSize = 35.sp,  // Keep font size constant
//                    fontWeight = FontWeight.Bold)
//
//            }else if (deepfakeResponseState.value.data?.score!! == 0.5){
//                Text("Score: ${formattedScore}", color = Color.Red,
//                    fontSize = 35.sp,  // Keep font size constant
//                    fontWeight = FontWeight.Bold)
//            }else{
//                Text("Score: ${formattedScore}", color = Color.Red,
//                    fontSize = 35.sp,  // Keep font size constant
//                    fontWeight = FontWeight.Bold)
//            }
            Spacer(modifier = Modifier.height(16.dp))
            Box(
                contentAlignment = Alignment.Center,  // Centers the text inside the animation
                modifier = Modifier
                    .fillMaxWidth(0.95f)
                    .height(50.dp)
                    .clickable {
                        if(navigationFlag.value){
                            navController.navigate(VIDEOSELECTIONSCREEN){
                                popUpTo(0)
                            }
                        }else{
                            FancyToast.makeText(context,"Let background task finish",FancyToast.LENGTH_SHORT,FancyToast.ERROR,false).show()
                        }

                    }
            ) {
                // Lottie Animation
                LottieAnimation(
                    composition = lottiecomposition,
                    progress = { progress2 },
                    modifier = Modifier.fillMaxWidth()
                )

                // Overlayed Text
                Text(
                    text = "New Video",  // Your desired text
                    color = Color.White,  // Adjust color for visibility
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold
                )
            }

        }
        else if(deepfakeResponseState.value.error.isNullOrEmpty()){
//            FancyToast.makeText(context,"Error in processing",FancyToast.LENGTH_SHORT,FancyToast.ERROR,false).show()
        }


    }

}

