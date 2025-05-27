package com.example.deepshield.presentation.Screens

import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavController
import com.airbnb.lottie.compose.LottieAnimation
import com.airbnb.lottie.compose.LottieCompositionSpec
import com.airbnb.lottie.compose.LottieConstants
import com.airbnb.lottie.compose.animateLottieCompositionAsState
import com.airbnb.lottie.compose.rememberLottieComposition
import com.example.deepshield.R
import com.example.deepshield.presentation.Navigation.VIDEOSELECTIONSCREEN

@Composable
fun SelectDeepFakeTypeScreen(navController: NavController) {
    val lottiecomposition by rememberLottieComposition(LottieCompositionSpec.RawRes(R.raw.button))
    val progress2 by animateLottieCompositionAsState(
        composition = lottiecomposition,
        iterations = LottieConstants.IterateForever,
        speed = 0.75f
    )
    val context = LocalContext.current
    val composition by rememberLottieComposition(
        LottieCompositionSpec.RawRes(R.raw.cloud)
    )
    Column (modifier = Modifier.fillMaxSize() ,horizontalAlignment = Alignment.CenterHorizontally){
        Spacer(modifier = Modifier.height(16.dp))
        LazyColumn(modifier = Modifier.fillMaxSize()) {
            item {
                Box(
                    contentAlignment = Alignment.Center,  // Centers the text inside the animation
                    modifier = Modifier
                        .fillMaxWidth(0.95f)
                        .height(50.dp)
                        .clickable {
                            navController.navigate(VIDEOSELECTIONSCREEN)
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
                        text = "Video",  // Your desired text
                        color = Color.White,  // Adjust color for visibility
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
                Spacer(modifier = Modifier.height(16.dp))
                Box(
                    contentAlignment = Alignment.Center,  // Centers the text inside the animation
                    modifier = Modifier
                        .fillMaxWidth(0.95f)
                        .height(50.dp)
                        .clickable {

                        }
                ) {
                    LottieAnimation(
                        composition = lottiecomposition,
                        progress = { progress2 },
                        modifier = Modifier.fillMaxWidth()  // Makes animation fill the Box
                    )

                    Text(
                        text = "Audio",  // Your desired text
                        color = Color.White,  // Adjust color for visibility
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
                Spacer(modifier = Modifier.height(16.dp))
                Box(
                    contentAlignment = Alignment.Center,  // Centers the text inside the animation
                    modifier = Modifier
                        .fillMaxWidth(0.95f)
                        .height(50.dp)
                        .clickable {

                        }
                ) {
                    LottieAnimation(
                        composition = lottiecomposition,
                        progress = { progress2 },
                        modifier = Modifier.fillMaxWidth()  // Makes animation fill the Box
                    )

                    Text(
                        text = "Image",  // Your desired text
                        color = Color.White,  // Adjust color for visibility
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold
                    )
                }


            }


        }

    }

}